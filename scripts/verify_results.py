#!/usr/bin/env python3
"""Offline checks that published tables reproduce saved row-level predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_benchmarks import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2026-11-06'


def main():
    baseline=pd.read_csv(OUT/'baselines/summary.csv')
    if len(baseline)!=20:raise ValueError('Expected 20 completed baseline rows')
    for _,r in baseline.iterrows():
        x=pd.read_csv(OUT/f'baselines/{r.dataset}__{r.model}__detections.csv.gz')
        np.testing.assert_array_equal(x.anomaly,x.score>=x.threshold)
        actual=metrics(x.label,x.score,x.anomaly)
        for key in ['evaluated','positives','precision','recall','f1','average_precision','tp','fp','fn','tn']:
            np.testing.assert_allclose(actual[key],r[key],rtol=1e-10,atol=1e-12)
        if len(x)!=r.test_rows:raise ValueError('Baseline row count mismatch')
    kronos=pd.read_csv(OUT/'kronos/detections.csv.gz')
    summary=json.loads((OUT/'kronos/summary.json').read_text())
    features=summary['features']
    flags=[]
    for feature in features:
        threshold=summary['pot'][feature]['threshold']
        np.testing.assert_allclose(kronos['threshold_'+feature],threshold)
        np.testing.assert_array_equal(kronos['anomaly_'+feature],kronos['score_'+feature]>=threshold)
        flags.append(kronos['anomaly_'+feature].to_numpy())
    np.testing.assert_array_equal(kronos.anomaly,np.stack(flags).any(axis=0))
    test=kronos[kronos.split=='test']
    assert len(test)==summary['test_rows'] and int(test.anomaly.sum())==summary['test_alarms']
    computed=metrics(test.weak_label,test.max_score_ratio,test.anomaly)
    for key,value in computed.items():
        np.testing.assert_allclose(value,summary['weak_annotation_metrics'][key])
    assert pd.to_datetime(kronos.loc[kronos.split=='calibration','timestamps']).max() < pd.to_datetime(test.timestamps).min()
    proxy_count=0
    if (OUT/'kronos_proxy/all_metrics.json').exists():
        from verify_kronos_proxy import main as verify_proxy
        verify_proxy()
        proxy_count=10
        assert len(pd.read_csv(OUT/'all_results.csv'))==31
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='result_checksums.json':
            files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (OUT/'result_checksums.json').write_text(json.dumps(dict(row_level_consistency='passed',baseline_experiments=20,kronos_cases=1,kronos_proxy_experiments=proxy_count,files=files),indent=2)+'\n')
    print(f'Verified 20 baseline, 1 native Kronos and {proxy_count} Kronos-Proxy experiments against saved outputs.')


if __name__=='__main__':main()
