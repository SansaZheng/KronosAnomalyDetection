#!/usr/bin/env python3
"""Do the proposed remedies for Kronos' weaknesses as an anomaly detector actually work?

Real Kronos-mini, frozen label knob of results/2026-11-10, S-FFSD and Sapienza 15 s.  Six remedies:

  fusion        B (NLL surprise), K (knob) and the ordinary level-magnitude detector L combined (Fisher)
  two_stage     L screens; Kronos scores only the top q % of rows by L (threshold from the reference period)
  rolling       1 % alarm threshold re-estimated on the last W scored rows instead of fixed on calibration
  clean_context rows flagged by L (above its reference 95th / 99th percentile) are dropped from the 32-row context
  joint_input   up to six features share one Kronos sequence (one per OHLCVA channel) instead of one sequence each
  finetune      Kronos-mini's predictor fine-tuned on unlabelled reference-period windows (tokenizer frozen)

The first three need no new forward pass.  The last three re-run Kronos on the evaluation rows (all labelled
S-FFSD test rows; all Sapienza pump rows plus 6,000 random normal test rows) and on 2,000 normal calibration
rows that provide the p-values.  Every remedy is label-free except where the knob itself is used.
Outputs: results/2026-11-24/solutions/<dataset>/.
"""
import argparse
import copy
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from boundary_missrate import conformal_logp, fisher
from boundary_standin import label_boundary
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.fusion import robust_reference
from kronos_states import KronosStates, load_kronos, normalize_full, proxy_pool, proxy_sequences, proxy_stamps
from run_fusion import recall_at_fpr

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'results/2026-11-10'
OUT = ROOT / 'results/2026-11-24/solutions'
LOOK, SEED, NORMAL_CAP, NULL_CAP = 32, 20261116, 6000, 2000
FT_STEPS, FT_ROWS, FT_LR, FT_SECONDS = 600, 32, 4e-5, 1500


def metrics(y, score):
    return dict(ap=float(average_precision_score(y, score)), auroc=float(roc_auc_score(y, score)),
                recall_1pct_fpr=recall_at_fpr(y, score))


def joint_sequences(z_windows, z_targets, positions, prescale=True):
    """Up to six features per Kronos sequence, one per OHLCVA channel.  Returns x (b * C, L + 1, 6), stamps, C."""
    b, L, F = z_windows.shape
    series = np.concatenate([z_windows, z_targets[:, None, :]], axis=1)          # (b, L+1, F)
    chunks = [np.arange(F)[i:i + 6] for i in range(0, F, 6)]
    full = np.zeros((b, len(chunks), L + 1, 6))
    for c, cols in enumerate(chunks):
        full[:, c, :, :len(cols)] = series[:, :, cols]
    x, *_ = normalize_full(full.reshape(b * len(chunks), L + 1, 6), prescale)
    pos = np.asarray(positions)[:, None] - L + np.arange(L)[None, :]
    return x, np.repeat(proxy_stamps(pos), len(chunks), axis=0), len(chunks)


def main(dataset, only=None):
    warnings.filterwarnings('ignore')
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    lines, table = [], []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    def record(remedy, variant, y, score, **extra):
        row = dict(remedy=remedy, variant=variant, **metrics(y, score), **extra)
        table.append(row)
        pd.DataFrame(table).to_csv(out / f'solutions{"_" + only if only else ""}.csv', index=False)
        log(f'  {remedy:14s} {variant:42s} AP {row["ap"]:.3f} AUROC {row["auroc"]:.3f} recall@1% {row["recall_1pct_fpr"]:.3f} '
            + ' '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in extra.items()))

    name = label_boundary(dataset)
    with open(FROZEN / f'knowledge_boundary_kronos_{dataset}/{name}_state.pkl', 'rb') as fh:
        _, _, predictor, _ = pickle.load(fh)
    meta = json.loads((FROZEN / f'kronos_rows/{dataset}/meta.json').read_text())
    caches = {s: np.load(FROZEN / f'kronos_rows/{dataset}/{s}.npz') for s in ('cal', 'test')}
    data = load_rows(dataset)
    values, parts, groups, labels = data['values'], data['parts'], np.asarray(data['groups']), data['labels']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    F = int(keep.sum())
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    z_raw = ((values[:, keep] - center) / scale).astype(np.float64)
    level = np.abs(z_raw).max(axis=1)
    positions = pd.Series(np.arange(len(groups))).groupby(groups).cumcount().to_numpy()

    def channels(split):
        c = caches[split]
        t = c['targets']
        return dict(t=t, h=c['histories'], B=c['nll'].astype(float), L=level[t], K=predictor.score(c['H'].astype(np.float32)),
                    y=labels[t].astype(int))

    cal, test = channels('cal'), channels('test')
    null = cal['y'] == 0                                         # labelled-normal calibration rows: the p-value reference
    known = np.isin(test['y'], [0, 1])
    y = test['y'][known]
    logp = {k: conformal_logp(test[k], cal[k][null]) for k in 'BLK'}
    fuse = lambda keys, lp=logp: fisher(np.stack([lp[k] for k in keys]))
    log(f'{dataset}: {int(known.sum())} labelled test rows, {int(y.sum())} anomalies; {int(null.sum())} normal calibration rows for p-values')

    # ------------------------------------------------------------------ fusion (no new forward pass)
    if only in (None, 'offline'):
        for key in 'BLK':
            record('fusion', {'B': 'B: Kronos NLL surprise', 'L': 'L: ordinary level magnitude', 'K': 'K: Kronos knob'}[key], y, test[key][known])
        for keys in ('BK', 'BL', 'LK', 'BLK'):
            record('fusion', ' + '.join(keys) + ' (Fisher)', y, fuse(keys)[known])

        # -------------------------------------------------------------- two-stage screening
        ref_level = level[parts == 'reference']
        for q in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
            screened = test['L'] > np.quantile(ref_level, 1 - q)
            lp = {k: np.where(screened, logp[k], 0.0) if k != 'L' else logp[k] for k in 'BLK'}   # unscreened rows: p_B = p_K = 1
            record('two_stage', f'Kronos on rows above the reference {100 * (1 - q):.0f}th pct of L', y, fuse('BLK', lp)[known],
                   kronos_share_of_test_rows=float(screened.mean()), anomalies_screened_in=float(screened[known][y == 1].mean()))
        record('two_stage', 'Kronos on every row', y, fuse('BLK')[known], kronos_share_of_test_rows=1.0, anomalies_screened_in=1.0)

        # -------------------------------------------------------------- rolling threshold (label-free)
        roll = []
        for label_, score_cal, score_test in (('B', cal['B'], test['B']), ('B + L + K (Fisher)', None, fuse('BLK'))):
            if score_cal is None:
                lpc = {k: conformal_logp(cal[k], cal[k][null]) for k in 'BLK'}
                score_cal = fisher(np.stack([lpc[k] for k in 'BLK']))
            fixed = float(np.quantile(score_cal, 0.99))
            for window in (None, 500, 2000, 5000):
                if window is None:
                    alarm = score_test > fixed
                else:
                    stream = np.r_[score_cal[-window:], score_test]
                    thr = pd.Series(stream).rolling(window).quantile(0.99).shift(1).to_numpy()[window:]
                    alarm = score_test > thr
                a = alarm[known]
                roll.append(dict(score=label_, threshold='fixed on calibration' if window is None else f'rolling, last {window} rows',
                                 false_alarm_rate=float(a[y == 0].mean()), recall=float(a[y == 1].mean()), alarm_share=float(alarm.mean())))
                log(f'  rolling        {label_:20s} {roll[-1]["threshold"]:26s} false-alarm rate {roll[-1]["false_alarm_rate"]:.4f} '
                    f'recall {roll[-1]["recall"]:.3f} alarm share {roll[-1]["alarm_share"]:.4f}')
        pd.DataFrame(roll).to_csv(out / 'rolling_threshold.csv', index=False)
    if only == 'offline':
        (out / 'run_offline.log').write_text('\n'.join(lines) + '\n')
        return

    # ------------------------------------------------------------------ remedies that re-run Kronos
    rng = np.random.default_rng(SEED)
    idx_known = np.flatnonzero(known)
    normal_idx = idx_known[y == 0]
    if len(normal_idx) > NORMAL_CAP:
        normal_idx = np.sort(rng.choice(normal_idx, NORMAL_CAP, replace=False))
    E = np.sort(np.r_[idx_known[y == 1], normal_idx])
    N = np.flatnonzero(null)
    if len(N) > NULL_CAP:
        N = np.sort(rng.choice(N, NULL_CAP, replace=False))
    yE = test['y'][E]
    sets = {'eval': (test['t'][E], test['h'][E]), 'null': (cal['t'][N], cal['h'][N])}
    base = {k: (test[k][E], cal[k][N]) for k in 'BLK'}
    lpE = {k: conformal_logp(*base[k]) for k in 'BLK'}
    log(f'  re-scoring {len(E)} evaluation rows ({int(yE.sum())} anomalies) and {len(N)} normal calibration rows')
    for keys in ('B', 'BK', 'BLK'):
        record('baseline', (' + '.join(keys) + ' (Fisher)') if len(keys) > 1 else 'B: Kronos NLL surprise',
               yE, fisher(np.stack([lpE[k] for k in keys])) if len(keys) > 1 else base['B'][0])

    model, tokenizer, official = load_kronos()
    states = KronosStates(model, tokenizer, official, seed=42)
    chunk = max(8, 1024 // F)

    def score_rows(st, rows, hist, capture=False, joint=False):
        """NLL surprise (and layer-pooled hidden states) of rows given explicit 32-row histories."""
        nll, hidden = [], []
        for start in range(0, len(rows), chunk):
            sl = slice(start, start + chunk)
            if joint:
                x, stamp, C = joint_sequences(z_raw[hist[sl]], z_raw[rows[sl]], positions[rows[sl]], meta['prescale'])
                nll.append(-st.loglik(x, stamp).reshape(-1, C).sum(axis=1))
            else:
                x, stamp, _ = proxy_sequences(z_raw[hist[sl]], z_raw[rows[sl]], positions[rows[sl]], meta['prescale'])
                pooled = proxy_pool(st.run(x, stamp, capture=capture), len(rows[sl]), F)
                nll.append(-pooled['loglik'])
                if capture:
                    hidden.append(pooled['H'])
        return np.concatenate(nll), (np.concatenate(hidden) if capture else None)

    def evaluate(remedy, variant, st, histories=None, capture=False, joint=False, **extra):
        t0 = time.time()
        got = {s: score_rows(st, rows, hist if histories is None else histories[s], capture, joint) for s, (rows, hist) in sets.items()}
        seconds = time.time() - t0
        per_1000 = 1000 * seconds / (len(E) + len(N))
        b_eval, b_null = got['eval'][0], got['null'][0]
        record(remedy, f'{variant}: B', yE, b_eval, seconds_per_1000_rows=float(per_1000), **extra)
        lp = dict(lpE, B=conformal_logp(b_eval, b_null))
        if capture:
            lp['K'] = conformal_logp(predictor.score(got['eval'][1]), predictor.score(got['null'][1]))
            record(remedy, f'{variant}: B + K (Fisher)', yE, fisher(np.stack([lp[k] for k in 'BK'])))
            record(remedy, f'{variant}: B + L + K (Fisher)', yE, fisher(np.stack([lp[k] for k in 'BLK'])))
        else:
            record(remedy, f'{variant}: B + L (Fisher)', yE, fisher(np.stack([lp[k] for k in 'BL'])))
        return b_eval

    t0 = time.time()
    b_check, _ = score_rows(states, *sets['eval'])
    log(f'  unchanged Kronos recomputed in {time.time() - t0:.0f}s ({1000 * (time.time() - t0) / len(E):.1f}s per 1,000 rows); '
        f'max |difference to cached NLL| {np.abs(b_check - base["B"][0]).max():.4f}')

    # ---- clean context: drop rows the ordinary detector flags from the history
    members = pd.Series(np.arange(len(groups))).groupby(groups).indices
    ref_level = level[parts == 'reference']
    for q in (0.95, 0.99):
        flagged = level > np.quantile(ref_level, q)
        clean = {g: idx[~flagged[idx]] for g, idx in members.items()}
        histories, replaced = {}, {}
        for s, (rows, hist) in sets.items():
            new, changed = hist.copy(), 0
            for i, r in enumerate(rows):
                pool = clean[groups[r]]
                before = pool[: np.searchsorted(pool, r)]
                if len(before) >= LOOK and flagged[hist[i]].any():
                    new[i] = before[-LOOK:]
                    changed += 1
            histories[s], replaced[s] = new, changed / len(rows)
        evaluate('clean_context', f'history without rows above L {100 * q:.0f}th pct', states, histories, capture=True,
                 rows_with_changed_history=float(replaced['eval']), flagged_share_of_all_rows=float(flagged.mean()))

    # ---- joint input: features share a sequence
    evaluate('joint_input', f'{F} features in {int(np.ceil(F / 6))} sequence(s) per row instead of {F}', states, joint=True)

    # ---- fine-tuning on unlabelled reference-period windows
    ref_rows = np.flatnonzero((parts == 'reference') & (positions >= LOOK))
    tuned = copy.deepcopy(model).train()
    optimizer = torch.optim.AdamW(tuned.parameters(), lr=FT_LR, weight_decay=0.1)
    train_rng, losses, t0 = np.random.default_rng(SEED + 1), [], time.time()
    for step in range(FT_STEPS):
        rows = train_rng.choice(ref_rows, FT_ROWS, replace=False)
        hist = np.stack([members[groups[r]][positions[r] - LOOK: positions[r]] for r in rows])
        x, stamp, _ = proxy_sequences(z_raw[hist], z_raw[rows], positions[rows], meta['prescale'])
        with torch.no_grad():
            s1, s2 = tokenizer.encode(torch.as_tensor(x, dtype=torch.float32), half=True)
        s1_logits, s2_logits = tuned(s1[:, :-1], s2[:, :-1], torch.as_tensor(stamp, dtype=torch.float32),
                                     use_teacher_forcing=True, s1_targets=s1[:, 1:])
        loss, _, _ = tuned.head.compute_loss(s1_logits, s2_logits, s1[:, 1:], s2[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuned.parameters(), 3.0)
        optimizer.step()
        losses.append(float(loss))
        if step % 100 == 0 or step == FT_STEPS - 1:
            log(f'  finetune step {step}: loss {np.mean(losses[-50:]):.4f} ({time.time() - t0:.0f}s)')
        if time.time() - t0 > FT_SECONDS:
            log(f'  finetune stopped at step {step} (time limit)')
            break
    tuned_states = KronosStates(tuned.eval(), tokenizer, official, seed=42)
    evaluate('finetune', f'predictor fine-tuned {len(losses)} steps on unlabelled reference rows', tuned_states,
             train_seconds=float(time.time() - t0), first_loss=float(np.mean(losses[:20])), last_loss=float(np.mean(losses[-20:])))
    (out / f'run{"_" + only if only else ""}.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--only', choices=['offline', 'inference'])
    args = parser.parse_args()
    main(args.dataset, args.only)
