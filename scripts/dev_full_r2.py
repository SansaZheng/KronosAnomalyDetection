#!/usr/bin/env python3
"""Round-2 design on every calibration row (fit = earlier half, dev = later half), no test rows.

The subsampled calibration targets hold 0-11 labelled positives per dataset, so the
design ladder is evaluated here on all calibration rows with a 32-row group history,
scored by the pipeline without Kronos (whose stacking weight is 0-0.2 everywhere).
Selection rule, fixed before running: among pure configurations (no row-level
detector), choose the best mean rank over AP and recall at 1% FPR across datasets
with >= 20 dev positives; the injection ladder on the subsampled partitions breaks ties.
The ensemble with the Isolation Forest row channel is a separate arm.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from dev_ladder_r2 import BASE, LADDER
from fusion_data import (NOT_DETECTION_TARGETS, PROXY_NAMES, SD, iforest_scores, load_rows, long_context,
                         split_calibration)
from kronos_ad.fusion import EvidenceFusion, FusionConfig
from kronos_ad.proxy import select_targets
from run_fusion import recall_at_fpr

OUT = 'results/2026-11-07/kronos_fusion_r2/dev_full.csv'


def thin(index, cap=4000):
    return index if len(index) <= cap else index[np.linspace(0, len(index) - 1, cap, dtype=int)]


def main():
    warnings.filterwarnings('ignore')
    names = sys.argv[1:] or PROXY_NAMES
    only = [c for c in LADDER if c.startswith(tuple(f'C{k} ' for k in (7, 8, 9, 10)))] if '--new' in names else ([c for c in LADDER if c.startswith('C11 ')] if '--c11' in names else list(LADDER))
    names = [n for n in names if not n.startswith('--')]
    rows = []
    for name in names:
        data = load_rows(name)
        values, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
        is_context = np.array([f in NOT_DETECTION_TARGETS for f in data['features']])
        targets, histories, _ = select_targets(parts, groups, 32, 0, 0)
        calibration = targets[parts[targets] == 'calibration']
        hist_cal = histories[parts[targets] == 'calibration']
        fit_mask, dev_mask = split_calibration(calibration, groups[calibration])
        fit, dev = thin(np.flatnonzero(fit_mask)), np.flatnonzero(dev_mask)
        reference = values[parts == 'reference']
        center, scale = long_context(values, groups)
        forest = iforest_scores(reference, values[calibration])
        x_fit, w_fit = values[calibration[fit]], values[hist_cal[fit]]
        y_dev = labels[calibration[dev]]
        known = np.isin(y_dev, [0, 1])
        positives = int((y_dev == 1).sum())
        for label, spec in LADDER.items():
            if label not in only:
                continue
            spec = dict(spec)
            contextual = spec.pop('contextual')
            config = FusionConfig(**{**BASE, **spec})
            global_mask = np.full(values.shape[1], name not in SD)
            if contextual == 'exclude':
                global_mask &= ~is_context
            forecasts = {}
            if 'long' in config.channels:
                forecasts['long:center'], forecasts['long:scale'] = center[calibration], scale[calibration]
            if config.row_evidence:
                forecasts['row:iforest'] = forest
            model = EvidenceFusion(config).fit(x_fit, w_fit, {k: v[fit] for k, v in forecasts.items()}, reference,
                                               global_mask, ~is_context)
            score = np.concatenate([model.score(values[calibration[chunk]], values[hist_cal[chunk]],
                                                {k: v[chunk] for k, v in forecasts.items()})
                                    for chunk in np.array_split(dev, max(1, len(dev) // 20000))])
            rows.append(dict(dataset=name, config=label, dev_rows=len(dev), dev_positives=positives,
                             ap=average_precision_score(y_dev[known] == 1, score[known]) if 0 < positives < known.sum() else np.nan,
                             auroc=roc_auc_score(y_dev[known] == 1, score[known]) if 0 < positives < known.sum() else np.nan,
                             r1=recall_at_fpr(y_dev[known].astype(int), score[known]) if positives else np.nan,
                             active_pairs=int(model.active_mask().sum())))
        print(name, 'dev rows', len(dev), 'positives', positives, flush=True)
        partial = pd.DataFrame(rows)
        suffix = ('__new' if any(c.startswith('C7 ') for c in only) else '__c11') if only != list(LADDER) else ''
        partial.to_csv(OUT.replace('.csv', f'__{name}{suffix}.csv'), index=False)
        rows = []
    import glob
    table = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(OUT.replace('.csv', '__*.csv')))], ignore_index=True)
    table.to_csv(OUT, index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    for metric in ['ap', 'r1', 'auroc']:
        print('\n==', metric)
        print(table.pivot(index='dataset', columns='config', values=metric)[list(LADDER)].round(3))
    usable = table[table.dev_positives >= 20]
    ranks = pd.concat([usable.pivot(index='dataset', columns='config', values=m).rank(axis=1, ascending=False).mean()
                       for m in ['ap', 'r1']], axis=1, keys=['ap_rank', 'r1_rank'])
    ranks['mean_rank'] = ranks.mean(axis=1)
    print('\n== selection summary over datasets with >= 20 dev positives:', sorted(usable.dataset.unique()))
    print(ranks.round(2).sort_values('mean_rank'))


if __name__ == '__main__':
    main()
