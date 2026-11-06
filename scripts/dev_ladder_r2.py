#!/usr/bin/env python3
"""Round-2 design ladder, evaluated on the calibration partitions only (fit -> validation).

Selection rule, fixed before running: among the pure configurations C0-C5 (no row-level
detector), pick the one with the best mean rank over (validation AP on datasets with
>= 5 validation positives, injection power at delta 3 and 6), subject to the held-out
alarm check staying within [0.3%, 3%] on every dataset. The ensemble C6 is a separate
arm and is tested alongside whatever pure configuration wins. No test row is touched.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fusion_data import NATIVE, PROXY_NAMES, native_bundle, proxy_bundle
from kronos_ad.fusion import EvidenceFusion, FusionConfig, is_forecaster
from run_fusion import proxy_shift, pump_bar, recall_at_fpr, subset

BASE = dict(channels=('forecast', 'local', 'global'), naive=('last', 'median'), aggregate='max',
            gate_skill=0.0, global_rarity='scale', row_evidence=False)
LADDER = {
    'C0 round1': dict(contextual='exclude'),
    'C1 +long': dict(contextual='exclude', channels=('forecast', 'local', 'long', 'global')),
    'C2 +gate': dict(contextual='exclude', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95),
    'C3 +contextual': dict(contextual='global', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95),
    'C4 +rank': dict(contextual='global', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95, global_rarity='rank'),
    'C5 +multi': dict(contextual='global', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95, global_rarity='rank', aggregate='multi'),
    'C3 +multi': dict(contextual='global', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95, aggregate='multi'),
    'C1 +multi': dict(contextual='exclude', channels=('forecast', 'local', 'long', 'global'), aggregate='multi'),
    'C7 gate+long+multi4': dict(contextual='exclude', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95, aggregate='multi', statistics=('tippett', 'fisher', 'meanz', 'maxz')),
    'C8 C7+contextual': dict(contextual='global', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95, aggregate='multi', statistics=('tippett', 'fisher', 'meanz', 'maxz')),
    'C9 gate+multi4 nolong': dict(contextual='exclude', channels=('forecast', 'local', 'global'), gate_skill=0.95, aggregate='multi', statistics=('tippett', 'fisher', 'meanz', 'maxz')),
    'C10 C7 magnitude only': dict(contextual='exclude', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95, aggregate='multi', statistics=('meanz', 'maxz')),
    'C11 C1+multi+iforest': dict(contextual='exclude', channels=('forecast', 'local', 'long', 'global'), aggregate='multi', row_evidence=True),
    'C6 +iforest': dict(contextual='global', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95, global_rarity='rank', aggregate='multi', row_evidence=True),
    'C6 max': dict(contextual='global', channels=('forecast', 'local', 'long', 'global'), gate_skill=0.95, global_rarity='rank', aggregate='max', row_evidence=True),
}


def fit_config(b, spec):
    spec = dict(spec)
    contextual = spec.pop('contextual')
    config = FusionConfig(**{**BASE, **spec})
    forecasts = {k: v for k, v in b['forecasts'].items() if not k.startswith('long:') or 'long' in config.channels}
    forecasts = {k: v for k, v in forecasts.items() if not k.startswith('row:') or config.row_evidence}
    global_mask = b['global_mask'].copy()
    if contextual == 'exclude':
        global_mask &= b['sequential_mask']
    fit = b['fit']
    model = EvidenceFusion(config).fit(b['x'][fit], b['windows'][fit], subset(forecasts, fit), b['reference'],
                                       global_mask, b['sequential_mask'])
    return model, forecasts


def main():
    warnings.filterwarnings('ignore')
    bundles = [native_bundle(long=True, iforest=True)] + [proxy_bundle(n, contextual='global', long=True, iforest=True) for n in PROXY_NAMES]
    rows = []
    for b in bundles:
        va = b['validation']
        y = b['y'][va]
        known = np.isin(y, [0, 1])
        positives = int((y == 1).sum())
        normal = y != 1
        for label, spec in LADDER.items():
            model, forecasts = fit_config(b, spec)
            score = model.score(b['x'][va], b['windows'][va], subset(forecasts, va))
            out = dict(dataset=b['name'], config=label, validation_positives=positives,
                       ap=average_precision_score(y[known] == 1, score[known]) if 0 < positives < known.sum() else np.nan,
                       r1=recall_at_fpr(y[known].astype(int), score[known]) if positives else np.nan,
                       active_pairs=int(model.active_mask().sum()))
            cut = np.quantile(score[normal], .99)
            for delta in (3, 6):
                if b['name'] == NATIVE:
                    shocked = pump_bar(b, va, delta, 1)
                    forecasts_shocked = subset(forecasts, va)
                else:
                    shocked = b['x'][va] + proxy_shift(b, va, delta, 1)
                    forecasts_shocked = subset(forecasts, va)
                if 'row:iforest' in forecasts_shocked:
                    from fusion_data import iforest_scores
                    forecasts_shocked = dict(forecasts_shocked, **{'row:iforest': iforest_scores(b['reference'], shocked)})
                out[f'power{delta}'] = float((model.score(shocked, b['windows'][va], forecasts_shocked)[normal] > cut).mean())
            half = len(score) // 2
            ordered = np.sort(score[:half])
            threshold = ordered[min(int(np.ceil((half + 1) * .99)), half) - 1]
            out['alarm_B'] = float((score[half:] > threshold).mean())
            rows.append(out)
        print(b['name'], 'validation positives', positives, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv('results/2026-11-07/kronos_fusion_r2/dev_ladder.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    for metric in ['ap', 'r1', 'power3', 'power6', 'alarm_B', 'active_pairs']:
        print('\n==', metric)
        print(table.pivot(index='dataset', columns='config', values=metric)[list(LADDER)].round(3))
    usable = table[table.validation_positives >= 5]
    ranks = []
    for metric in ['ap', 'power3', 'power6']:
        source = usable if metric == 'ap' else table
        pivot = source.pivot(index='dataset', columns='config', values=metric)
        ranks.append(pivot.rank(axis=1, ascending=False).mean())
    summary = pd.concat(ranks, axis=1, keys=['ap_rank', 'power3_rank', 'power6_rank'])
    summary['mean_rank'] = summary.mean(axis=1)
    alarm = table.pivot(index='dataset', columns='config', values='alarm_B')
    summary['alarm_ok'] = ((alarm >= 0.003) & (alarm <= 0.03)).all()
    print('\n== selection summary (lower rank is better)')
    print(summary.round(2))


if __name__ == '__main__':
    main()
