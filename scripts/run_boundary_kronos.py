#!/usr/bin/env python3
"""Knowledge-boundary analysis of the real Kronos model on OHLCVA bars.

Needs the official checkpoints on disk (Hugging Face is not reachable from the
evaluation container), e.g.

  PYTHONPATH=.:scripts python scripts/run_boundary_kronos.py \
      --kronos-src /path/to/shiyu-coder-Kronos \
      --model-dir /path/to/Kronos-mini --tokenizer-dir /path/to/Kronos-Tokenizer-2k \
      --bars results/2026-11-06/kronos/detections.csv.gz --out results/<date>/knowledge_boundary_kronos

`--tiny` runs the same script on a random-weight miniature Kronos built from the pinned
model code, as a dry run of the plumbing only; its numbers mean nothing.

Protocol (frozen here, before any real-weight run):
  - bars: a CSV with timestamps, open, high, low, close, volume, amount, a `split`
    column (calibration / test) and optionally a 0/1 `label` column;
  - windows of 64 bars normalized by the official rule (optionally prescaled);
  - H = embedding output and every block output at the last context bar;
  - error boundary: y = 1 when the teacher-forced NLL of the realised next bar is above
    the calibration 90th percentile; label boundary: y = 1 on labelled bars (needs >= 20
    calibration and >= 10 test positives);
  - probes fit on calibration rows, evaluated on balanced test rows; knob search with
    delta = 1%, k <= 10, 50 random controls; interventions at the eight tau values of the
    paper with tangent and random controls; effect = change in log p(s1*, s2* | context).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from boundary_pipeline import analyse_boundary, batched, save_results, thin
from kronos_ad.boundary import split_balanced
from kronos_ad.hidden import KronosHiddenStates, normalize_windows
from run_fusion import recall_at_fpr

COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']


def stamps(timestamps):
    t = pd.DatetimeIndex(timestamps)
    return np.column_stack([t.minute, t.hour, t.weekday, t.day, t.month]).astype(np.float32)


def load_models(args):
    if args.tiny:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests.vendor.kronos import Kronos, KronosTokenizer
        torch.manual_seed(0)
        model = Kronos(s1_bits=4, s2_bits=4, n_layers=3, d_model=32, n_heads=4, ff_dim=64, ffn_dropout_p=0.0,
                       attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True)
        tokenizer = KronosTokenizer(d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
                                    ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=4, s2_bits=4,
                                    beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=1)
        return model, tokenizer
    sys.path.insert(0, args.kronos_src)
    from model import Kronos, KronosTokenizer
    return Kronos.from_pretrained(args.model_dir), KronosTokenizer.from_pretrained(args.tokenizer_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bars', default='results/2026-11-06/kronos/detections.csv.gz')
    p.add_argument('--label-column', default='weak_label')
    p.add_argument('--kronos-src')
    p.add_argument('--model-dir')
    p.add_argument('--tokenizer-dir')
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--lookback', type=int, default=64)
    p.add_argument('--prescale', action='store_true')
    p.add_argument('--row-cap', type=int, default=24000)
    p.add_argument('--k-max', type=int, default=10)
    p.add_argument('--random-trials', type=int, default=50)
    p.add_argument('--out', default='results/2026-11-07/knowledge_boundary_kronos')
    a = p.parse_args()
    warnings.filterwarnings('ignore')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    bars = pd.read_csv(a.bars, parse_dates=['timestamps'])
    values = bars[COLUMNS].to_numpy(dtype=float)
    labels = bars[a.label_column].to_numpy() if a.label_column in bars else np.zeros(len(bars))
    split = bars['split'].to_numpy()
    L = a.lookback
    index = np.arange(L, len(bars))
    windows = np.stack([values[i - L:i] for i in index])
    x = normalize_windows(windows, prescale=a.prescale).astype(np.float32)
    # the realised next bar, normalized with its own window's statistics
    full = normalize_windows(np.stack([values[i - L:i + 1] for i in index]), prescale=a.prescale).astype(np.float32)
    x_next = full[:, -1]
    stamp = np.stack([stamps(bars.timestamps.iloc[i - L:i]) for i in index])
    model, tokenizer = load_models(a)
    states = KronosHiddenStates(model, tokenizer)
    parts = split[index]
    calibration = np.flatnonzero(parts == 'calibration')
    test = np.flatnonzero(parts == 'test')
    calibration = np.union1d(thin(calibration, a.row_cap, 1), calibration[labels[index][calibration] == 1])
    test = np.union1d(thin(test, a.row_cap, 2), test[labels[index][test] == 1])
    log(f'bars {len(bars)}, windows {len(index)}, calibration rows {len(calibration)}, test rows {len(test)}, layers {states.layer_count}')

    def encode(rows):
        s1, s2 = states.tokens(x[rows])
        s1n, s2n = states.next_tokens(x[rows], x_next[rows])
        return s1, s2, torch.tensor(stamp[rows]), s1n, s2n

    def hidden(rows):
        s1, s2, st, _, _ = encode(rows)
        return states.hidden_states(s1, s2, st)

    def loglik(rows, patch=None):
        s1, s2, st, s1n, s2n = encode(rows)
        return states.loglik(s1, s2, st, s1n, s2n, patch=patch)

    H_cal = batched(hidden, calibration, size=512)
    H_test = batched(hidden, test, size=512)
    nll_cal = -batched(loglik, calibration, size=512)
    nll_test = -batched(loglik, test, size=512)
    normal_cal = labels[index][calibration] != 1
    threshold = float(np.quantile(nll_cal[normal_cal], 0.9))
    boundaries = {'error': ((nll_cal > threshold).astype(int), (nll_test > threshold).astype(int)),
                  'label': ((labels[index][calibration] == 1).astype(int), (labels[index][test] == 1).astype(int))}
    log(f'teacher-forced NLL: calibration median {np.median(nll_cal):.3f}, threshold {threshold:.3f}; '
        f'label positives calibration {int(boundaries["label"][0].sum())}, test {int(boundaries["label"][1].sum())}')
    results, detection = [], []
    for name, (y_cal, y_test) in boundaries.items():
        if y_cal.sum() < 20 or y_test.sum() < 10:
            log(f'  {name}: too few positives ({int(y_cal.sum())} / {int(y_test.sum())}); skipped')
            continue
        train_idx, _ = split_balanced(y_cal, 0.0, seed=0)
        test_idx, _ = split_balanced(y_test, 0.0, seed=1)
        rows_test = test[test_idx]

        def loglik_test(patch, parts, rows=rows_test):
            return loglik(rows, patch), None

        result, probes, predictor = analyse_boundary(name, H_cal[train_idx], y_cal[train_idx], H_test[test_idx], y_test[test_idx],
                                                     loglik_test, out, log, k_max=a.k_max, random_trials=a.random_trials,
                                                     cnn_epochs=10 if a.tiny else 80)
        results.append(result)
        y_all = labels[index][test].astype(int)
        known = np.isin(y_all, [0, 1])
        if 0 < y_all[known].sum() < known.sum():
            for key, s in {'nll_surprise': nll_test, 'cnn_probe': probes['cnn'].score(H_test),
                           'logistic_probe': probes['logistic'].score(H_test), 'knob_closed_form': predictor.score(H_test)}.items():
                detection.append(dict(boundary=name, score=key, test_rows=int(known.sum()), positives=int(y_all[known].sum()),
                                      ap=float(average_precision_score(y_all[known], s[known])),
                                      auroc=float(roc_auc_score(y_all[known], s[known])),
                                      recall_1pct_fpr=recall_at_fpr(y_all[known], s[known])))
    header = dict(bars=a.bars, tiny_dry_run=a.tiny, model_dir=a.model_dir, tokenizer_dir=a.tokenizer_dir, lookback=L,
                  prescale=a.prescale, layers=states.layer_count, error_threshold=threshold)
    save_results(out, header, results, detection)
    (out / 'run.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
