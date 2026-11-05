#!/usr/bin/env python3
"""Real Kronos-mini rolling one-step forecasts on original BRD/BTC 5-minute bars."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch

from kronos_ad.pot import fit_pot

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['open','high','low','close','volume','amount']


def load_bars(path):
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    checksum=path.with_name(path.name+'.CHECKSUM').read_text().split()[0]
    if source_hash!=checksum:raise ValueError('Binance published SHA256 mismatch')
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise ValueError('ZIP CRC mismatch')
        raw=pd.read_csv(z.open(z.namelist()[0]),header=None)
    frame=raw.iloc[:,[0,1,2,3,4,5,7]].copy()
    frame.columns=['timestamps',*FEATURES]
    frame.timestamps=pd.to_datetime(frame.timestamps,unit='ms')
    if frame.timestamps.duplicated().any() or not frame.timestamps.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise ValueError('Missing, duplicate or irregular market bars')
    if not np.isfinite(frame[FEATURES]).all().all():raise ValueError('Nonfinite market observations')
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='mps');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--sample-count',type=int,default=3);p.add_argument('--lookback',type=int,default=64)
    p.add_argument('--output',type=Path,default=ROOT/'results/2026-11-06/kronos')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);torch.set_num_threads(4)
    sys.path.insert(0,str(ROOT/'vendor/Kronos'))
    from model import Kronos,KronosTokenizer,KronosPredictor
    tokenizer=KronosTokenizer.from_pretrained(str(ROOT/'weights/Kronos-Tokenizer-2k')).eval()
    model=Kronos.from_pretrained(str(ROOT/'weights/Kronos-mini')).eval()
    predictor=KronosPredictor(model,tokenizer,device=a.device,max_context=512)
    frame=load_bars(ROOT/'runs/market/BRDBTC-5m-2018-12.zip')
    selected=np.flatnonzero(frame.timestamps>=pd.Timestamp('2018-12-22'))
    checkpoint=a.output/'forecast_cache.csv'
    cache_spec=dict(lookback=a.lookback,sample_count=a.sample_count,batch_size=a.batch_size,device=a.device,seed=42,T=1.0,top_p=.9,top_k=0,first_scored='2018-12-22',model_sha256=hashlib.sha256((ROOT/'weights/Kronos-mini/model.safetensors').read_bytes()).hexdigest(),tokenizer_sha256=hashlib.sha256((ROOT/'weights/Kronos-Tokenizer-2k/model.safetensors').read_bytes()).hexdigest())
    cache_metadata=a.output/'forecast_cache_parameters.json'
    if checkpoint.exists():
        if not cache_metadata.exists() or json.loads(cache_metadata.read_text())!=cache_spec:
            raise ValueError('Forecast cache parameters differ; select a fresh output directory')
        forecasts=pd.read_csv(checkpoint,parse_dates=['timestamps'])
        if not forecasts.timestamps.equals(frame.loc[selected,'timestamps'].reset_index(drop=True)):
            raise ValueError('Forecast cache timestamp mismatch')
        pred=forecasts[FEATURES].to_numpy()
    else:
        predicted=[];start=time.monotonic()
        for start_i in range(0,len(selected),a.batch_size):
            indices=selected[start_i:start_i+a.batch_size]
            windows=[frame.iloc[i-a.lookback:i].copy().reset_index(drop=True) for i in indices]
            with torch.inference_mode():
                batch=predictor.predict_batch(
                    df_list=[w[FEATURES] for w in windows],
                    x_timestamp_list=[w.timestamps for w in windows],
                    y_timestamp_list=[pd.Series([frame.timestamps.iloc[i]]) for i in indices],
                    pred_len=1,T=1.0,top_k=0,top_p=.9,sample_count=a.sample_count,verbose=False)
            predicted.extend([v.iloc[0][FEATURES].to_numpy(dtype=float) for v in batch])
            print('Kronos',min(start_i+a.batch_size,len(selected)),'/',len(selected),round(time.monotonic()-start,1),'sec',flush=True)
        pred=np.array(predicted)
        forecasts=pd.DataFrame(pred,columns=FEATURES);forecasts.insert(0,'timestamps',frame.timestamps.iloc[selected].to_numpy())
        forecasts.to_csv(checkpoint,index=False)
        cache_metadata.write_text(json.dumps(cache_spec,indent=2)+'\n')
    epsilon=np.array([1e-8]*4+[1e-3,1e-8])
    std=np.array([frame.iloc[i-a.lookback:i][FEATURES].std(ddof=0).to_numpy() for i in selected])
    scores=np.square((frame.loc[selected,FEATURES].to_numpy()-pred)/(std+epsilon))
    calibration=(frame.timestamps.iloc[selected]<pd.Timestamp('2018-12-28')).to_numpy()
    tails={f:fit_pot(scores[calibration,j],quantile=.9,risk=.01,min_exceedances=30) for j,f in enumerate(FEATURES)}
    # Weak event/chunk annotations are used only AFTER forecasts and thresholds.
    chunks=pd.read_csv(ROOT/'data/raw/sapienza_pump_dump/labeled_features/features_25S.csv.gz')
    chunks=chunks.loc[(chunks.symbol=='BRD')&(chunks.pump_index==0)].copy()
    chunks['bar']=pd.to_datetime(chunks.date).dt.floor('5min')
    weak_labels=chunks.groupby('bar')['gt'].max()
    rows=frame.loc[selected].reset_index(drop=True).copy();rows['split']=np.where(calibration,'calibration','test')
    rows['weak_label']=rows.timestamps.map(weak_labels).fillna(-1).astype(int)
    for j,f in enumerate(FEATURES):
        rows['prediction_'+f]=pred[:,j];rows['score_'+f]=scores[:,j]
        rows['threshold_'+f]=tails[f]['threshold'];rows['anomaly_'+f]=scores[:,j]>=tails[f]['threshold']
    rows['anomaly']=rows[['anomaly_'+f for f in FEATURES]].any(axis=1)
    rows['max_score_ratio']=np.max(scores/np.array([tails[f]['threshold'] for f in FEATURES]),axis=1)
    test=rows.loc[~calibration]
    from run_benchmarks import metrics
    summary=dict(dataset='BRDBTC_5m_2018_12',model='Kronos-mini',lookback=a.lookback,sample_count=a.sample_count,
                 temperature=1.0,top_p=.9,top_k=0,seed=42,batch_size=a.batch_size,device=a.device,
                 calibrated_rows=int(calibration.sum()),test_rows=len(test),features=FEATURES,epsilon=epsilon.tolist(),
                 pot=tails,calibration_end='2018-12-27 23:55:00',test_start='2018-12-28 00:00:00',
                 test_alarms=int(test.anomaly.sum()),test_alarm_rate=float(test.anomaly.mean()),
                 calibration_joint_alarm_rate=float(rows.loc[calibration,'anomaly'].mean()),
                 per_feature_test_alarms={f:int(test['anomaly_'+f].sum()) for f in FEATURES},
                 weak_annotation_metrics=metrics(test.weak_label,test.max_score_ratio,test.anomaly),
                 warning='Single retrospective case. Labels are mapped from published 25-second feature chunks, not exhaustive market-fraud ground truth. Unannotated bars remain unknown. Pretrained model may have seen 2018 data.',
                 versions={n:version(n) for n in ['torch','huggingface_hub','numpy','pandas','scipy']})
    rows.to_csv(a.output/'detections.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    pd.DataFrame([dict(feature=f,**tails[f],test_alarms=summary['per_feature_test_alarms'][f]) for f in FEATURES]).to_csv(a.output/'feature_summary.csv',index=False)
    print('Kronos finished:',summary['test_alarms'],'alerts /',len(test),'test bars',flush=True)


if __name__=='__main__':main()
