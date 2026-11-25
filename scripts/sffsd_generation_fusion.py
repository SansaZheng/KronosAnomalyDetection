#!/usr/bin/env python3
"""S-FFSD: does day-by-day generation help once it is combined with the other label-free Kronos channels?

G  = row distance from the block ("day") that Kronos generated from the 32 rows before the block
Gb = mean of G over the block (a "this whole day looks unlike what Kronos expected" score)
B  = ordinary rolling surprise (NLL), L = level magnitude, K = frozen fraud knob.

Every channel is turned into a p-value against the normal calibration rows and channels are combined with
Fisher's method, exactly as in the earlier label-free experiments (no detector is trained on labels here).
Scores are reported on all labelled test rows and separately on test rows outside the fraud burst.
Needs: scripts/sffsd_block_generation.py --mode generate for both --part calibration and --part test.
Output: results/2026-11-26/sffsd_best/generation_fusion.csv
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from boundary_missrate import conformal_logp, fisher
from boundary_standin import label_boundary
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.fusion import robust_reference

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'results/2026-11-10'
OUT = ROOT / 'results/2026-11-26/sffsd_best'
D = 32


def metrics(y, score, prefix=''):
    out = {prefix + 'ap': float(average_precision_score(y, score)), prefix + 'auroc': float(roc_auc_score(y, score))}
    for fpr in (0.01, 0.05):
        out[f'{prefix}recall_{int(100 * fpr)}pct'] = float((score[y == 1] > np.quantile(score[y == 0], 1 - fpr)).mean())
    return out


def main():
    data = load_rows('s_ffsd')
    values, parts, labels = data['values'], data['parts'], data['labels'].astype(int)
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    level = np.abs((values[:, keep] - center) / scale).max(axis=1)
    with open(FROZEN / f'knowledge_boundary_kronos_s_ffsd/{label_boundary("s_ffsd")}_state.pkl', 'rb') as fh:
        _, _, predictor, _ = pickle.load(fh)
    n = len(labels)
    chan = {k: np.full(n, np.nan) for k in ('B', 'K', 'G', 'Gb')}
    for part, tag in (('calibration', 'cal'), ('test', 'test')):
        c = np.load(FROZEN / f'kronos_rows/s_ffsd/{tag}.npz')
        chan['B'][c['targets']] = c['nll']
        chan['K'][c['targets']] = predictor.score(c['H'].astype(np.float32))
        g = np.load(OUT / f'generated_D{D}_{part}.npz')
        chan['G'][g['targets']] = g['score_feature'].sum(axis=1)
        chan['Gb'][g['targets']] = g['block_mean']
    chan['L'] = level
    have = ~np.any([np.isnan(v) for v in chan.values()], axis=0)
    cal = have & (parts == 'calibration') & (labels == 0)
    test = np.flatnonzero(have & (parts == 'test') & np.isin(labels, [0, 1]))
    y = labels[test]
    t0 = np.flatnonzero(parts == 'test')[0]
    bins = (test - t0) // 1000
    share = pd.Series(y).groupby(bins).mean()
    outside = ~np.isin(bins, share.index[share > 0.5])
    print(f'calibration normals {int(cal.sum())}; test rows {len(test)} ({int(y.sum())} fraud); outside the burst {int(outside.sum())} rows '
          f'({int(y[outside].sum())} fraud)')
    logp = {k: conformal_logp(v[test], v[cal]) for k, v in chan.items()}
    combos = [('B',), ('G',), ('Gb',), ('G', 'Gb'), ('K',), ('L', 'K'), ('B', 'L', 'K'), ('G', 'K'), ('Gb', 'K'), ('G', 'Gb', 'K'),
              ('G', 'L', 'K'), ('Gb', 'L', 'K'), ('G', 'Gb', 'L', 'K'), ('B', 'G', 'Gb', 'L', 'K')]
    rows = []
    for combo in combos:
        score = fisher(np.stack([logp[k] for k in combo])) if len(combo) > 1 else -logp[combo[0]]
        rows.append(dict(channels=' + '.join(combo), **metrics(y, score), **metrics(y[outside], score[outside], 'outside_')))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'generation_fusion.csv', index=False)
    print(frame.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
