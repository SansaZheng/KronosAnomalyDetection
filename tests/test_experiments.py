import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from run_benchmarks import encode, metrics, run_one, temporal_parts


def test_timestamp_ties_stay_in_one_split():
    frame=pd.DataFrame({'Time':np.repeat(np.arange(10),3),'x':np.arange(30)})
    sorted_frame,parts=temporal_parts(frame,'Time')
    for t in sorted_frame.Time.unique():
        assert len(set(parts[sorted_frame.Time==t]))==1
    assert set(sorted_frame.loc[parts=='reference','Time']).isdisjoint(sorted_frame.loc[parts=='test','Time'])


def test_categorical_encoding_uses_reference_only():
    frame=pd.DataFrame({'category':['a','a','b','new','new']})
    x,names=encode(frame,np.array(['reference']*3+['test']*2))
    assert x[:,0].tolist()==pytest.approx([2/3,2/3,1/3,0,0])


def test_unknown_labels_excluded_from_accuracy():
    result=metrics([0,1,2],[.1,.9,1.0],[False,True,True])
    assert result['evaluated']==2 and result['unknown_excluded']==1
    assert result['f1']==1.0 and result['positives']==1


def test_label_permutation_does_not_change_scores_or_thresholds(tmp_path):
    rng=np.random.default_rng(12);n=3000
    x=pd.DataFrame({'value':rng.normal(size=n)})
    parts=np.repeat(['reference','calibration','test'],[1800,600,600])
    labels=rng.integers(0,2,size=n);ids=np.arange(n).astype(str)
    for out in ['one','two']:(tmp_path/out).mkdir()
    first=run_one('fixture',(x,labels,parts,ids,'fixture'),tmp_path/'one')
    second=run_one('fixture',(x,1-labels,parts,ids,'fixture'),tmp_path/'two')
    for a,b in zip(first,second):
        assert a['pot']==b['pot']
        f=pd.read_csv(tmp_path/'one'/f"fixture__{a['model']}__detections.csv.gz")
        s=pd.read_csv(tmp_path/'two'/f"fixture__{a['model']}__detections.csv.gz")
        np.testing.assert_array_equal(f.score,s.score)
        np.testing.assert_array_equal(f.anomaly,s.anomaly)
