#!/usr/bin/env python3
"""Label-free baseline evaluation on downloaded release files; no Kronos claims."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from kronos_ad.pot import fit_pot

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, score, alarms):
    y = np.asarray(y)
    known = np.isin(y, [0, 1])
    yy, ss, aa = y[known].astype(int), np.asarray(score)[known], np.asarray(alarms)[known]
    p, r, f, _ = precision_recall_fscore_support(yy, aa, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yy, aa, labels=[0, 1]).ravel()
    return dict(evaluated=len(yy), unknown_excluded=int((~known).sum()), positives=int(yy.sum()),
                prevalence=float(yy.mean()), precision=float(p), recall=float(r), f1=float(f),
                average_precision=float(average_precision_score(yy, ss)) if len(np.unique(yy))==2 else None,
                roc_auc=float(roc_auc_score(yy, ss)) if len(np.unique(yy))==2 else None,
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                false_positive_rate=float(fp/(fp+tn)) if fp+tn else None,
                alarm_rate=float(np.asarray(alarms).mean()))


def temporal_parts(frame, time_col):
    # Whole timestamp units stay together, so day/hour/index ties never cross splits.
    frame = frame.sort_values(time_col, kind="stable").reset_index(drop=True)
    units = np.sort(frame[time_col].unique())
    a, b = int(len(units)*0.6), int(len(units)*0.8)
    parts = np.where(frame[time_col] < units[a], "reference", np.where(frame[time_col] < units[b], "calibration", "test"))
    return frame, parts


def encode(frame, parts):
    ref = frame.loc[parts=="reference"]
    out = {}
    for name in frame:
        if pd.api.types.is_numeric_dtype(frame[name]):
            out[name] = pd.to_numeric(frame[name]).to_numpy(dtype=float)
        else:
            # Training-only frequencies, not ordinal codes or label target encoding.
            frequencies = ref[name].astype(str).value_counts(normalize=True)
            out[name+"_reference_frequency"] = frame[name].astype(str).map(frequencies).fillna(0).to_numpy()
    x = pd.DataFrame(out)
    medians = x.loc[parts=="reference"].median()
    x = x.replace([np.inf,-np.inf], np.nan).fillna(medians).to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("A feature has no finite reference values")
    return x, list(out)


def tabular(raw, dataset):
    if dataset=="ulb_creditcard":
        with zipfile.ZipFile(raw/'ulb_creditcard/ulb_creditcard.zip') as z:
            frame=pd.read_csv(z.open('creditcard.csv'))
        frame, parts=temporal_parts(frame,'Time')
        return frame.drop(columns=['Class','Time']),frame['Class'].to_numpy(),parts,frame.Time.astype(str).to_numpy(),'chronological whole-Time 60/20/20'
    if dataset=="banksim":
        with zipfile.ZipFile(raw/'banksim/banksim.zip') as z:
            frame=pd.read_csv(z.open('bs140513_032310.csv'))
        frame, parts=temporal_parts(frame,'step')
        return frame.drop(columns=['fraud','step','zipcodeOri','zipMerchant']),frame.fraud.to_numpy(),parts,frame.step.astype(str).to_numpy(),'chronological whole-day 60/20/20'
    if dataset=="s_ffsd":
        with zipfile.ZipFile(raw/'s_ffsd/data/S-FFSD.zip') as z:
            frame=pd.read_csv(z.open('S-FFSD.csv'))
        frame, parts=temporal_parts(frame,'Time')
        return frame.drop(columns=['Labels','Time']),frame.Labels.to_numpy(),parts,frame.Time.astype(str).to_numpy(),'transaction-index 60/20/20; label 2 unknown'
    raise ValueError(dataset)


def pump(raw, resolution):
    frame=pd.read_csv(raw/f'sapienza_pump_dump/labeled_features/features_{resolution}S.csv.gz')
    frame['date']=pd.to_datetime(frame['date'])
    events=frame.groupby('pump_index').date.agg(['min','max']).sort_values('min')
    first=events['min'].iloc[int(len(events)*.6)]
    second=events['min'].iloc[int(len(events)*.8)]
    # Purge events straddling boundaries; an event never appears in two splits.
    group={i:('reference' if row['max']<first else 'calibration' if row['min']>=first and row['max']<second else 'test' if row['min']>=second else 'purged') for i,row in events.iterrows()}
    parts=frame.pump_index.map(group).to_numpy()
    keep=parts!='purged';frame=frame.loc[keep].reset_index(drop=True);parts=parts[keep]
    frame=frame.sort_values('date',kind='stable')
    parts=parts[frame.index];frame=frame.reset_index(drop=True)
    x=frame.drop(columns=['date','pump_index','gt'])
    return x,frame['gt'].to_numpy(),parts,(frame.pump_index.astype(str)+':'+frame.date.astype(str)).to_numpy(),'event-group chronological 60/20/20; boundary-overlap events purged; upstream engineered features'


def windows(raw, folder, data_name, label_name):
    x=np.load(raw/'sd_fmm'/folder/data_name,allow_pickle=False)
    y=np.load(raw/'sd_fmm'/folder/label_name,allow_pickle=False)
    n, length=x.shape[:2];a,b=int(.6*n),int(.8*n)
    parts=np.repeat(np.where(np.arange(n)<a,'reference',np.where(np.arange(n)<b,'calibration','test')),length)
    ids=np.array([f'case{i}:point{j}' for i in range(n) for j in range(length)])
    return pd.DataFrame(x.reshape(-1,x.shape[-1]),columns=['value']),y.reshape(-1),parts,ids,'whole-case file-order 60/20/20; timestamps/source overlap unverified, NOT chronological evidence'


def run_one(name, bundle, out):
    frame,y,parts,ids,protocol=bundle
    x,features=encode(frame,parts)
    train=x[parts=='reference'];cal=x[parts=='calibration'];test=x[parts=='test']
    truth=y[parts=='test']
    baseline=IsolationForest(n_estimators=100,max_samples=256,contamination='auto',random_state=42,n_jobs=2)
    start=time.monotonic();baseline.fit(train)
    median=np.median(train,axis=0);mad=np.median(np.abs(train-median),axis=0)*1.4826
    scale=np.where(mad>1e-10,mad,np.std(train,axis=0));scale=np.where(scale>1e-10,scale,1.0)
    output=[]
    for model in ['IsolationForest','RobustMAD']:
        if model=='IsolationForest':
            cal_score=-baseline.score_samples(cal);test_score=-baseline.score_samples(test)
        else:
            cal_score=np.mean(np.abs((cal-median)/scale),axis=1)
            test_score=np.mean(np.abs((test-median)/scale),axis=1)
        tail=fit_pot(cal_score,quantile=.9,risk=.01,min_exceedances=30)
        alarms=test_score>=tail['threshold']
        result=dict(dataset=name,model=model,reference_rows=len(train),calibration_rows=len(cal),test_rows=len(test),
                    protocol=protocol,features=features,seed=42,pot=tail,
                    **metrics(truth,test_score,alarms))
        output.append(result)
        rows=pd.DataFrame(dict(row_id=ids[parts=='test'],label=truth,evaluation_known=np.isin(truth,[0,1]),score=test_score,threshold=tail['threshold'],anomaly=alarms))
        rows.to_csv(out/f'{name}__{model}__detections.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (out/f'{name}__details.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(name,'completed',len(test),'test rows',round(time.monotonic()-start,2),'sec',flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/2026-11-06/baselines')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);raw=ROOT/'data/raw'
    jobs=[(d,lambda d=d:tabular(raw,d)) for d in ['ulb_creditcard','banksim','s_ffsd']]
    jobs += [(f'sapienza_{s}s',lambda s=s:pump(raw,s)) for s in [5,15,25]]
    for name,folder,data,label in [('sd_crypto','cryptocurrency cases','test_data.npy','test_label.npy'),('sd_stock','stock cases','test_data_bs.npy','test_label_bs.npy'),('sd_prices','prices','test_data_02.npy','test_label_02.npy'),('sd_synthetic','systhesis','all_DTWsim_data.npy','all_DTWsim_label.npy')]:
        jobs.append((name,lambda folder=folder,data=data,label=label:windows(raw,folder,data,label)))
    results=[];failures=[]
    for name,load in jobs:
        try:results.extend(run_one(name,load(),a.output))
        except Exception as exc:
            failures.append(dict(dataset=name,error=repr(exc)));print('FAILED',name,repr(exc),flush=True)
    (a.output/'all_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['features','pot']}|{'threshold':r['pot']['threshold']} for r in results]).to_csv(a.output/'summary.csv',index=False)
    (a.output/'run.json').write_text(json.dumps(dict(failures=failures,versions={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']},models=['IsolationForest 100 trees max_samples256 seed42','mean absolute robust-MAD standardized deviation'],pot_quantile=.9,pot_risk=.01,min_exceedances=30,labels_used_for_training_or_thresholds=False),indent=2)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
