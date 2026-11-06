#!/usr/bin/env python3
"""Recompute the evidence-fusion tables from saved row-level outputs. Needs no Torch."""
import hashlib
import json

import numpy as np
import pandas as pd

from fusion_data import NATIVE, PROXY, PROXY_NAMES, RESULTS, ROOT
from run_fusion import ALPHA, OUT, evaluate

FROZEN = json.loads((OUT / 'frozen_config.json').read_text())


def main():
    manifest = json.loads((OUT / 'frozen_manifest.json').read_text())
    current = manifest['revisions'][-1]['sha256'] if manifest.get('revisions') else manifest['sha256']
    for path, digest in current.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    config = 'results/2026-11-06/kronos_fusion/frozen_config.json'
    assert current[config] == manifest['sha256'][config], 'frozen configuration changed after the freeze'
    metrics = pd.read_csv(OUT / 'metrics.csv').set_index(['dataset', 'method'])
    checked = 0
    for name in [NATIVE] + PROXY_NAMES:
        rows = pd.read_csv(OUT / f'{name}__rows.csv.gz')
        fit, validation, test = (rows[rows.split == s] for s in ('fit', 'validation', 'test'))
        assert fit.position.max() < validation.position.min() < test.position.min()
        if rows.group.nunique() > 1:
            assert set(fit.group).isdisjoint(validation.group) and set(validation.group).isdisjoint(test.group)
        for column, threshold in [('score', 'threshold'), ('score_no_kronos', 'threshold_no_kronos')]:
            ordered = np.sort(validation[column].to_numpy())
            rank = min(int(np.ceil((len(ordered) + 1) * (1 - ALPHA))), len(ordered))
            assert rows[threshold].nunique() == 1 and ordered[rank - 1] == rows[threshold].iloc[0]
        if name != NATIVE:
            old = pd.read_csv(PROXY / f'{name}__detections.csv.gz')
            np.testing.assert_array_equal(old.loc[old.split == 'test', 'source_row'].to_numpy(), test.source_row.to_numpy())
            np.testing.assert_array_equal(old.loc[old.split == 'test', 'label'].to_numpy(), test.label.to_numpy())
        y = test.label.to_numpy()
        methods = {'fusion': (test.score.to_numpy(), test.score.to_numpy() > test.threshold.iloc[0]),
                   'fusion_no_kronos': (test.score_no_kronos.to_numpy(), test.score_no_kronos.to_numpy() > test.threshold_no_kronos.iloc[0])}
        for column in test.columns:
            if column.endswith('_score') and column not in ('score', 'score_no_kronos'):
                method = column[:-6]
                methods[method] = (test[column].to_numpy(), test[method + '_alarm'].to_numpy(bool))
        for method, (s, f) in methods.items():
            expected = metrics.loc[(name, method)]
            for key, value in evaluate(y, s, f).items():
                if isinstance(value, float) and np.isnan(value):
                    assert np.isnan(expected[key]), (name, method, key)
                else:
                    assert np.isclose(value, expected[key]), (name, method, key, value, expected[key])
                checked += 1
    summary = json.loads((OUT / 'native_summary.json').read_text())
    native = pd.read_csv(OUT / f'{NATIVE}__rows.csv.gz')
    native_test = native[native.split == 'test']
    assert int((native_test.score > native_test.threshold.iloc[0]).sum()) == summary['fusion_alarms']
    pump = native_test[native_test.label == 1]
    assert len(pump) == 1 and int((native_test.score > pump.score.iloc[0]).sum()) + 1 == summary['pump_rank_fusion']
    bootstrap = pd.read_csv(OUT / 'bootstrap.csv')
    primary = [n for n in PROXY_NAMES if metrics.loc[(n, 'fusion'), 'positives'] >= FROZEN['endpoints']['primary_datasets_min_test_positives']]
    assert set(bootstrap.dataset) == set(primary), (set(bootstrap.dataset), primary)
    print('verified', checked, 'metric values across', len(PROXY_NAMES) + 1, 'datasets; primary datasets:', primary)


if __name__ == '__main__':
    main()
