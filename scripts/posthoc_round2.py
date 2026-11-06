#!/usr/bin/env python3
"""Post-hoc for round 2: paired bootstrap of the ensemble arm against Isolation Forest and
against the pure round-2 method, from the saved row files (same resampling scheme and seed)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import run_fusion
from fusion_data import PROXY_NAMES
from run_fusion import recall_at_fpr, resamples, resampling_units


def main():
    OUT = run_fusion.use_round('r2')
    metrics = pd.read_csv(OUT / 'metrics.csv').set_index(['dataset', 'method'])
    minimum = run_fusion.PRIMARY_MIN
    rows = []
    for name in PROXY_NAMES:
        if metrics.loc[(name, 'fusion'), 'positives'] < minimum:
            continue
        table = pd.read_csv(OUT / f'{name}__rows.csv.gz')
        test = table[(table.split == 'test') & table.label.isin([0, 1])]
        y = test.label.to_numpy().astype(int)
        scores = {'fusion_ensemble': test.fusion_ensemble_score.to_numpy(), 'fusion': test.score.to_numpy(),
                  'isolation_forest': test.isolation_forest_score.to_numpy(), 'robust_mad': test.robust_mad_score.to_numpy()}
        rng = np.random.default_rng(run_fusion.SEED)
        samples = {k: [] for k in scores}
        for index in resamples(test.group.to_numpy(), rng, 64, 2000):
            yy = y[index]
            if not 0 < yy.sum() < len(yy):
                continue
            for k, v in scores.items():
                samples[k].append((average_precision_score(yy, v[index]), recall_at_fpr(yy, v[index])))
        base = np.array(samples['fusion_ensemble'])
        for k in ('fusion', 'isolation_forest', 'robust_mad'):
            d = base - np.array(samples[k])
            rows.append(dict(dataset=name, comparison=f'ensemble minus {k}', units=resampling_units(test.group.to_numpy(), 64),
                             ap_diff_lo=np.quantile(d[:, 0], .025), ap_diff_hi=np.quantile(d[:, 0], .975),
                             r1_diff_lo=np.nanquantile(d[:, 1], .025), r1_diff_hi=np.nanquantile(d[:, 1], .975)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'posthoc_ensemble_bootstrap.csv', index=False)
    print(out.round(3).to_string())


if __name__ == '__main__':
    main()
