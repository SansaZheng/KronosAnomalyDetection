#!/usr/bin/env python3
"""Verify saved proxy decisions and metrics without downloading or running weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_benchmarks import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2026-11-06/kronos_proxy'


def main():
    results=json.loads((OUT/'all_metrics.json').read_text())
    assert len(results)==10,'Expected all ten non-OHLCVA dataset variants'
    assert json.loads((OUT/'run.json').read_text())['failures']==[]
    for r in results:
        rows=pd.read_csv(OUT/(r['dataset']+'__detections.csv.gz'))
        details=json.loads((OUT/(r['dataset']+'__details.json')).read_text())
        flags=[];ratios=[]
        for f,tail in details['tail_fits'].items():
            np.testing.assert_allclose(rows['threshold_'+f],tail['threshold'])
            flag=rows['score_'+f]>=rows['threshold_'+f]
            np.testing.assert_array_equal(flag,rows['anomaly_'+f])
            flags.append(flag);ratios.append(rows['score_'+f]/rows['threshold_'+f])
        np.testing.assert_array_equal(np.array(flags).any(axis=0),rows.anomaly)
        np.testing.assert_allclose(np.max(ratios,axis=0),rows.max_score_ratio,rtol=1e-10)
        with np.load(OUT/(r['dataset']+'__predictions.npz'),allow_pickle=False) as cache:
            np.testing.assert_array_equal(rows.source_row,cache['targets'])
            for j,f in enumerate(r['features']):np.testing.assert_allclose(rows['prediction_'+f],cache['predictions'][:,j],rtol=1e-8,atol=1e-12)
        test=rows[rows.split=='test']
        actual=metrics(test.label,test.max_score_ratio,test.anomaly)
        for key,value in actual.items():
            if value is None:assert r[key] is None
            else:np.testing.assert_allclose(value,r[key],rtol=1e-10,atol=1e-12)
        assert len(test)==r['test_rows']
        assert sum(rows.split=='calibration')==r['calibration_scored']
    files=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!='checksums.json':files.append(dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (OUT/'checksums.json').write_text(json.dumps(dict(status='passed',proxy_experiments=10,files=files),indent=2)+'\n')
    print('Verified ten real-weight proxy experiments against saved per-feature forecasts, decisions and test labels.')


if __name__=='__main__':main()
