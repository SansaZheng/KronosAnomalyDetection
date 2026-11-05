#!/usr/bin/env python3
"""Run actual frozen Kronos weights with a disclosed non-OHLCVA proxy adapter."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

from kronos_ad.proxy import select_targets,proxy_input,restore_prediction,ordinal_stamps,input_fingerprint
from kronos_ad.pot import fit_pot
from run_benchmarks import tabular,pump,windows,encode,metrics

ROOT=Path(__file__).resolve().parents[1]


def load_job(name):
    raw=ROOT/'data/raw'
    if name in ['ulb_creditcard','banksim','s_ffsd']:return tabular(raw,name),None
    if name.startswith('sapienza_'):return pump(raw,int(name.split('_')[1][:-1])),'event'
    files={'sd_crypto':('cryptocurrency cases','test_data.npy','test_label.npy'),
           'sd_stock':('stock cases','test_data_bs.npy','test_label_bs.npy'),
           'sd_prices':('prices','test_data_02.npy','test_label_02.npy'),
           'sd_synthetic':('systhesis','all_DTWsim_data.npy','all_DTWsim_label.npy')}
    return windows(raw,*files[name]),'case'


def run_job(name,predictor,a,out):
    torch.manual_seed(42)
    bundle,group_type=load_job(name)
    frame,labels,parts,ids,protocol=bundle
    groups=np.array([str(v).split(':',1)[0] for v in ids]) if group_type else np.repeat('stream',len(frame))
    targets,history_indices,local=select_targets(parts,groups,a.lookback,a.calibration_cap,a.test_cap)
    values,features=encode(frame,parts)
    reference=values[parts=='reference']
    center=np.median(reference,axis=0);scale=np.std(reference,axis=0);scale=np.where(scale>1e-10,scale,1.)
    epsilon=np.maximum(scale*1e-6,1e-8)
    count,dimensions=len(targets),values.shape[1]
    cache=out/(name+'__predictions.npz')
    spec=dict(dataset=name,lookback=a.lookback,sample_count=1,batch_size=a.batch_size,seed=42,device=a.device,calibration_cap=a.calibration_cap,test_cap=a.test_cap,features=features,
              input_sha256=input_fingerprint(values[history_indices],values[targets],center,scale,local[targets],targets),
              model_sha256=hashlib.sha256((ROOT/'weights/Kronos-mini/model.safetensors').read_bytes()).hexdigest(),
              tokenizer_sha256=hashlib.sha256((ROOT/'weights/Kronos-Tokenizer-2k/model.safetensors').read_bytes()).hexdigest())
    meta=out/(name+'__cache_parameters.json')
    if cache.exists():
        if json.loads(meta.read_text())!=spec:raise ValueError('Cached parameters mismatch')
        with np.load(cache,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['targets'],targets);predictions=z['predictions']
    else:
        predictions=np.empty((count,dimensions));started=time.monotonic()
        for start in range(0,count*dimensions,a.batch_size):
            flat=np.arange(start,min(start+a.batch_size,count*dimensions));ri=flat//dimensions;fi=flat%dimensions
            history=values[history_indices[ri],fi[:,None]]
            x,mean,std=proxy_input(history,center[fi],scale[fi])
            stamp=ordinal_stamps(local[targets[ri]][:,None]-np.arange(a.lookback,0,-1))
            future=ordinal_stamps(local[targets[ri]][:,None])
            with torch.inference_mode():
                forecast=predictor.generate(x,stamp,future,1,1.,0,.9,1,False)
            pred=restore_prediction(forecast[:,0,3],mean,std,center[fi],scale[fi])
            predictions[ri,fi]=pred
            if start==0 or start%(a.batch_size*10)==0 or flat[-1]==count*dimensions-1:
                print(name,int(flat[-1]+1),'/',count*dimensions,'feature forecasts',round(time.monotonic()-started,1),'sec',flush=True)
        if not np.isfinite(predictions).all():raise ValueError('Nonfinite model output')
        np.savez_compressed(cache,targets=targets,predictions=predictions)
        meta.write_text(json.dumps(spec,indent=2)+'\n')
    observed=values[targets]
    std=np.std(values[history_indices],axis=1,ddof=0)
    scores=np.square((observed-predictions)/(std+epsilon))
    cal=parts[targets]=='calibration';test=~cal
    tails={};failures=[]
    for j,f in enumerate(features):
        try:tails[f]=fit_pot(scores[cal,j],quantile=.90,risk=.01,min_exceedances=30)
        except Exception as exc:failures.append(dict(feature=f,error=repr(exc)))
    valid=[features.index(f) for f in tails]
    if not valid:raise ValueError('All per-feature POT fits failed')
    thresholds=np.array([tails[features[j]]['threshold'] for j in valid])
    ratio=scores[:,valid]/thresholds
    flags=(ratio>=1).any(axis=1);score=ratio.max(axis=1)
    # Test labels enter the pipeline only now, after predictions and calibration.
    result=dict(dataset=name,model='Kronos-mini-Proxy',representation='out-of-domain per-feature numeric proxy, not real OHLCVA',
                protocol=protocol,lookback=a.lookback,reference_rows=int((parts=='reference').sum()),
                original_calibration_rows=int((parts=='calibration').sum()),original_test_rows=int((parts=='test').sum()),
                calibration_scored=int(cal.sum()),test_rows=int(test.sum()),features=features,dimensions=dimensions,
                active_dimensions=len(valid),failed_dimensions=failures,
                selection='deterministic evenly-spaced valid positions; labels not used; history resets within event/case',
                calibration_cap=a.calibration_cap,test_cap=a.test_cap,per_feature_risk=.01,pot_quantile=.9,
                max_score_ratio_definition='max over successfully calibrated feature scores divided by their POT thresholds',
                calibration_joint_alarm_rate=float(flags[cal].mean()),
                **metrics(labels[targets][test],score[test],flags[test]))
    records=pd.DataFrame(dict(source_row=targets,row_id=ids[targets],group=groups[targets],split=parts[targets],label=labels[targets],evaluation_known=np.isin(labels[targets],[0,1]),max_score_ratio=score,anomaly=flags))
    for j,f in enumerate(features):
        records['prediction_'+f]=predictions[:,j];records['observed_'+f]=observed[:,j];records['score_'+f]=scores[:,j]
        records['threshold_'+f]=tails[f]['threshold'] if f in tails else np.nan
        records['anomaly_'+f]=(scores[:,j]>=tails[f]['threshold']) if f in tails else pd.NA
    records.to_csv(out/(name+'__detections.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
    (out/(name+'__details.json')).write_text(json.dumps(dict(summary=result,tail_fits=tails,epsilon=epsilon.tolist(),reference_center=center.tolist(),reference_scale=scale.tolist()),indent=2,allow_nan=False)+'\n')
    print('FINISHED',name,'test',result['test_rows'],'known',result['evaluated'],'positives',result['positives'],'alarms',int(flags[test].sum()),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='mps');p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lookback',type=int,default=32);p.add_argument('--calibration-cap',type=int,default=512);p.add_argument('--test-cap',type=int,default=1024)
    p.add_argument('--ids',nargs='+');p.add_argument('--output',type=Path,default=ROOT/'results/2026-11-06/kronos_proxy')
    a=p.parse_args()
    if not 2<=a.lookback<=512 or a.batch_size<1 or a.calibration_cap<0 or a.test_cap<0:p.error('Require lookback 2..512, positive batch size, nonnegative caps')
    a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'vendor/Kronos'))
    from model import Kronos,KronosTokenizer,KronosPredictor
    torch.set_num_threads(4);torch.manual_seed(42)
    net=Kronos.from_pretrained(str(ROOT/'weights/Kronos-mini')).eval();tok=KronosTokenizer.from_pretrained(str(ROOT/'weights/Kronos-Tokenizer-2k')).eval()
    predictor=KronosPredictor(net,tok,device=a.device,max_context=512)
    names=a.ids or ['ulb_creditcard','banksim','s_ffsd','sapienza_5s','sapienza_15s','sapienza_25s','sd_crypto','sd_stock','sd_prices','sd_synthetic']
    previous=a.output/'all_metrics.json'
    retained=json.loads(previous.read_text()) if a.ids and previous.exists() else []
    results=[r for r in retained if r['dataset'] not in names];failures=[]
    for name in names:
        try:results.append(run_job(name,predictor,a,a.output))
        except Exception as exc:failures.append(dict(dataset=name,error=repr(exc)));print('FAILED',name,repr(exc),flush=True)
        (a.output/'all_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    simple=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results]
    pd.DataFrame(simple).to_csv(a.output/'summary.csv',index=False)
    (a.output/'run.json').write_text(json.dumps(dict(failures=failures,arguments={k:v for k,v in vars(a).items() if k!='output'},output_directory_name=a.output.name,versions={n:version(n) for n in ['torch','numpy','pandas','scipy','scikit-learn']},native_Kronos_results_are_separate='../kronos/summary.json'),indent=2)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
