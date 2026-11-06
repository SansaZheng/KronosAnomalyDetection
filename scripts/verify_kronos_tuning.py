#!/usr/bin/env python3
"""Verify published tuning outputs without loading Torch or model weights."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from kronos_ad.joint import JointResidualCalibrator
from run_benchmarks import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/2026-11-06/kronos_tuning'


def main():
    summaries = json.loads((OUT / 'all_metrics.json').read_text())
    assert len(summaries) == 20
    table = pd.read_csv(OUT / 'comparison.csv')
    search = pd.read_csv(OUT / 'validation_search.csv')
    assert len(table) == 20 and len(search) == 1080
    for source in json.loads((OUT / 'input_manifest.json').read_text()):
        assert hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() == source['sha256']
    for r in summaries:
        frame = pd.read_csv(OUT / (r['dataset'] + '__' + r['method'] + '__detections.csv.gz'))
        np.testing.assert_array_equal(frame.anomaly, frame.score > frame.threshold)
        np.testing.assert_allclose(frame.threshold, r['fitted_state']['threshold'])
        fit = frame[frame.split == 'calibration_fit']
        val = frame[frame.split == 'validation']
        test = frame[frame.split == 'test']
        assert len(fit) == r['fit_rows'] and len(val) == r['validation_rows'] and len(test) == r['test_rows']
        assert fit.source_row.max() < val.source_row.min() < test.source_row.min()
        if frame.group.nunique() > 1:
            assert set(fit.group).isdisjoint(val.group)
        expected_threshold = np.sort(fit.score)[int(np.ceil((len(fit) + 1) * (1-r['config']['risk']))) - 1]
        np.testing.assert_allclose(expected_threshold, r['fitted_state']['threshold'], rtol=1e-12)
        for section, expected in [(test, r), (val, r['validation_metrics'])]:
            actual = metrics(section.label, section.score, section.anomaly)
            for key in actual:
                if expected[key] is None:
                    assert actual[key] is None or (key in ['f1', 'recall'] and actual['positives'] == 0)
                else:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-12)
        if r['method'] == 'joint_unsupervised' or not r['validation_selection_eligible']:
            assert r['config'] == dict(floor=.1, aggregate='top3', bias=True, risk=.01)
            assert not r['uses_validation_labels']
        else:
            rows = search[(search.dataset == r['dataset']) & (search.status == 'evaluated')]
            chosen = rows.sort_values(['f1', 'average_precision', 'alarm_rate', 'floor', 'risk'],
                                     ascending=[False, False, True, True, True], kind='stable').iloc[0]
            for key in ['floor', 'aggregate', 'bias', 'risk']:
                assert chosen[key] == r['config'][key]
        old = pd.read_csv(ROOT / ('results/2026-11-06/kronos_proxy/' + r['dataset'] + '__detections.csv.gz'))
        details = json.loads((ROOT / ('results/2026-11-06/kronos_proxy/' + r['dataset'] + '__details.json')).read_text())
        with np.load(OUT / (r['dataset'] + '__score_inputs.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['targets'], frame.source_row)
            replay = JointResidualCalibrator(**r['config']).fit(z['residual'][z['fit_indices']],
                         z['local_std'][z['fit_indices']], z['reference_scale'])
            assert replay.state() == r['fitted_state']
            np.testing.assert_allclose(replay.score(z['residual'], z['local_std']), frame.score, rtol=1e-12)
            np.testing.assert_allclose(z['reference_scale'], details['reference_scale'])
            features = details['summary']['features']
            observed = old[['observed_' + f for f in features]].to_numpy()
            predicted = old[['prediction_' + f for f in features]].to_numpy()
            np.testing.assert_allclose(z['residual'], observed-predicted, rtol=1e-8, atol=1e-10)
            original_score = old[['score_' + f for f in features]].to_numpy()
            recomputed = np.square(z['residual'] / (z['local_std'] + np.asarray(details['epsilon'])))
            np.testing.assert_allclose(recomputed, original_score, rtol=1e-9, atol=1e-8)
        np.testing.assert_array_equal(test.source_row, old[old.split == 'test'].source_row)
        np.testing.assert_array_equal(test.label, old[old.split == 'test'].label)
        comparison = table[(table.dataset == r['dataset']) & (table.method == r['method'])].iloc[0]
        for key in ['precision', 'recall', 'f1', 'average_precision', 'alarm_rate', 'tp', 'fp', 'fn', 'tn']:
            if r[key] is None:
                assert pd.isna(comparison['after_' + key])
            else:
                np.testing.assert_allclose(comparison['after_' + key], r[key], rtol=1e-10, atol=1e-12)
    print('Verified 20 tuned experiments, 1080 candidate records, source hashes, validation selection, thresholds and test-position identity.')


if __name__ == '__main__':
    main()
