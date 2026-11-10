#!/usr/bin/env python3
"""Knowledge-boundary analysis of the real Kronos-mini on a dataset's experiment rows.

Same protocol as `boundary_standin.py` (probes, difference maps, knob search with delta = 1%
and k <= 10, closed-form knob, interventions at the paper's eight tau values against tangent
and random controls), applied to the hidden states and teacher-forced NLLs cached by
`run_kronos_rows.py` -- the real model, loaded offline, on the same rows the stand-in saw.

    PYTHONPATH=.:scripts python scripts/run_boundary_kronos_rows.py --dataset banksim

Boundaries: `error` (NLL above the calibration 90th percentile on normal rows) and the
labelled boundary (`pump` / `fraud`).  Interventions re-run the model with the patch added
at the last context position in every layer (for a proxy row, in each feature's pass; the
effect is the change of the summed log-likelihood) on a balanced subset of at most
`--intervention-cap` test rows.  Outputs: results/2026-11-10/knowledge_boundary_kronos_<dataset>/.
"""
import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from boundary_pipeline import analyse_boundary, save_results
from boundary_standin import K_MAX, LOOKBACK, RANDOM_TRIALS, SEED, label_boundary
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.boundary import split_balanced
from kronos_ad.fusion import robust_reference
from kronos_states import KronosStates, load_kronos, native_sequences, proxy_sequences
from run_fusion import recall_at_fpr
from run_kronos_rows import OUT as ROWS

ROOT = Path(__file__).resolve().parents[1]


def default_out(dataset):
    return ROOT / f'results/2026-11-10/knowledge_boundary_kronos_{dataset}'


def balanced_subset(y, cap, seed):
    """Up to cap rows, half per class, as positions into y."""
    rng = np.random.default_rng(seed)
    pick = []
    for cls in (0, 1):
        idx = np.flatnonzero(np.asarray(y) == cls)
        pick.append(idx if len(idx) <= cap // 2 else np.sort(rng.choice(idx, cap // 2, replace=False)))
    return np.sort(np.concatenate(pick))


def main(dataset, out=None, intervention_cap=None, cnn_epochs=80):
    warnings.filterwarnings('ignore')
    out = Path(out) if out else default_out(dataset)
    out.mkdir(parents=True, exist_ok=True)
    rows_dir = ROWS / dataset
    meta = json.loads((rows_dir / 'meta.json').read_text())
    positive_name = label_boundary(dataset)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    data = load_rows(dataset)
    values, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
    native = 'bars' in data
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    features = [f for f, k in zip(data['features'], keep) if k]
    F = len(features)
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    z_raw = ((values[:, keep] - center) / scale).astype(np.float64)
    positions = pd.Series(np.arange(len(groups))).groupby(groups).cumcount().to_numpy()
    cache = {name: np.load(rows_dir / f'{name}.npz') for name in ('ref', 'cal', 'test')}
    t_cal, t_test = cache['cal']['targets'], cache['test']['targets']
    t_pool = np.r_[cache['ref']['targets'], t_cal]
    h_test = cache['test']['histories']
    H_pool = np.concatenate([cache['ref']['H'], cache['cal']['H']]).astype(np.float32)
    H_test = cache['test']['H'].astype(np.float32)
    nll_pool = np.r_[cache['ref']['nll'], cache['cal']['nll']]
    nll_test = cache['test']['nll']
    cap = intervention_cap if intervention_cap is not None else (2000 if native else 600)
    log(f'{dataset}: Kronos-mini ({meta["representation"]}, prescale {meta["prescale"]}), features {features}; '
        f'pool rows {len(t_pool)} (reference {len(cache["ref"]["targets"])} + calibration {len(t_cal)}), test rows {len(t_test)}; '
        f'hidden states {H_pool.shape[1:]}')
    model, tokenizer, official = load_kronos()
    states = KronosStates(model, tokenizer, official, seed=42)
    calibration_rows = np.arange(len(t_pool) - len(t_cal), len(t_pool))
    threshold = float(np.quantile(nll_pool[calibration_rows][labels[t_cal] != 1], 0.9))
    boundaries = {'error': ((nll_pool > threshold).astype(int), (nll_test > threshold).astype(int)),
                  positive_name: (labels[t_pool].astype(int), labels[t_test].astype(int))}
    log(f'teacher-forced NLL: calibration median {np.median(nll_pool[calibration_rows]):.3f}, error threshold (90th pct, normal rows) {threshold:.3f}; '
        f'{positive_name} positives pool {int((boundaries[positive_name][0] == 1).sum())}, test {int((boundaries[positive_name][1] == 1).sum())}')
    results, detection = [], []
    for name, (y_pool_b, y_test_b) in boundaries.items():
        train_idx, _ = split_balanced(y_pool_b, test_fraction=0.0, seed=SEED)
        test_idx, _ = split_balanced(y_test_b, test_fraction=0.0, seed=SEED + 1)
        if (y_test_b == 1).sum() < 10 or (y_pool_b == 1).sum() < 20:
            log(f'  {name}: too few positives for a balanced probe; skipped')
            continue
        sub = balanced_subset(y_test_b[test_idx], cap, SEED + 2)
        rows, hist = t_test[test_idx][sub], h_test[test_idx][sub]
        chunk = 256 if native else max(8, 1024 // F)

        def loglik_test(patch, parts, rows=rows, hist=hist):
            ll = []
            for start in range(0, len(rows), chunk):
                sl = slice(start, start + chunk)
                p = None if patch is None else patch[sl]
                if native:
                    x, stamp, _ = native_sequences(data['bars'], hist[sl], rows[sl], data['timestamps'], meta['prescale'])
                    ll.append(states.loglik(x, stamp, patch=p))
                else:
                    x, stamp, _ = proxy_sequences(z_raw[hist[sl]], z_raw[rows[sl]], positions[rows[sl]], meta['prescale'])
                    p = None if p is None else np.repeat(p, F, axis=0)
                    ll.append(states.loglik(x, stamp, patch=p).reshape(-1, F).sum(axis=1))
            return np.concatenate(ll), None

        t = time.time()
        state = out / f'{name}_state.pkl'          # per-boundary checkpoint: a killed run resumes at the next boundary
        if state.exists():
            with open(state, 'rb') as fh:
                result, probes, predictor, saved_lines = pickle.load(fh)
            lines.extend(saved_lines)
            log(f'  {name}: loaded from {state.name} (previous run)')
        else:
            start = len(lines)
            result, probes, predictor = analyse_boundary(name, H_pool[train_idx], y_pool_b[train_idx], H_test[test_idx], y_test_b[test_idx],
                                                         loglik_test, out, log, k_max=K_MAX, random_trials=RANDOM_TRIALS, seed=SEED,
                                                         cnn_epochs=cnn_epochs, intervention_index=sub)
            log(f'  {name}: done in {(time.time() - t) / 60:.1f} min (interventions on {len(sub)} balanced rows)')
            with open(state, 'wb') as fh:
                pickle.dump((result, probes, predictor, lines[start:]), fh)
        results.append(result)
        pump = labels[t_test].astype(int)
        known = np.isin(pump, [0, 1])
        scores = {'nll_surprise': nll_test, 'cnn_probe': probes['cnn'].score(H_test), 'logistic_probe': probes['logistic'].score(H_test),
                  'knob_closed_form': predictor.score(H_test)}
        for key, s in scores.items():
            detection.append(dict(boundary=name, score=key, test_rows=int(known.sum()), positives=int(pump[known].sum()),
                                  ap=float(average_precision_score(pump[known], s[known])), auroc=float(roc_auc_score(pump[known], s[known])),
                                  recall_1pct_fpr=recall_at_fpr(pump[known], s[known])))
    header = dict(dataset=dataset, label_boundary=positive_name, features=features,
                  model=dict(name='Kronos-mini + Kronos-Tokenizer-2k (official weights, offline)', d_model=meta['d_model'], layers=meta['layers'],
                             lookback=LOOKBACK, representation=meta['representation'], prescale=meta['prescale'],
                             model_sha256=meta['model_sha256'], tokenizer_sha256=meta['tokenizer_sha256']),
                  rows=dict(pool=int(len(t_pool)), test=int(len(t_test))), error_threshold=threshold, intervention_cap=cap)
    save_results(out, header, results, detection)
    (out / 'run.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out')
    p.add_argument('--intervention-cap', type=int)
    p.add_argument('--cnn-epochs', type=int, default=80)
    a = p.parse_args()
    main(a.dataset, a.out, a.intervention_cap, a.cnn_epochs)
