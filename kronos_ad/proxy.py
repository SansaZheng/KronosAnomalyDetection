"""Explicitly out-of-domain, per-feature Kronos proxy; NOT native OHLCVA inference."""
import hashlib

import numpy as np
import pandas as pd


def select_targets(parts, groups, lookback=32, calibration_cap=512, test_cap=1024):
    """Label-free, deterministic spread over each split; histories stay in group."""
    groups=np.asarray(groups);parts=np.asarray(parts)
    local=np.empty(len(groups),dtype=np.int64)
    histories={}
    for group in pd.unique(groups):
        positions=np.flatnonzero(groups==group)
        local[positions]=np.arange(len(positions))
        histories[group]=positions
    targets=[]
    for split,cap in [('calibration',calibration_cap),('test',test_cap)]:
        eligible=np.flatnonzero((parts==split)&(local>=lookback))
        if cap and len(eligible)>cap:
            eligible=eligible[np.linspace(0,len(eligible)-1,cap,dtype=int)]
        targets.extend(eligible.tolist())
    targets=np.asarray(targets,dtype=np.int64)
    windows=np.array([histories[groups[i]][local[i]-lookback:local[i]] for i in targets],dtype=np.int64)
    if not len(targets):raise ValueError('No valid within-group histories')
    if not (windows<targets[:,None]).all():raise ValueError('A history includes its target or a future row')
    return targets,windows,local


def proxy_input(history, location, scale):
    """Copy one standardized feature into four numeric price slots; V/A are zeros.

    These slots are an artificial representation. No real OHLC/volume is claimed.
    Returns exactly the normalization quantities needed to invert a close forecast.
    """
    values=(np.asarray(history,dtype=float)-np.asarray(location)[:,None])/np.asarray(scale)[:,None]
    values=values.astype(np.float32)
    mean=values.mean(axis=1);std=values.std(axis=1)
    standardized=np.clip((values-mean[:,None])/(std[:,None]+1e-5),-5,5)
    x=np.zeros((len(values),values.shape[1],6),dtype=np.float32)
    x[:,:,:4]=standardized[:,:,None]
    return x,mean,std


def restore_prediction(normalized_close, mean, std, location, scale):
    return (np.asarray(normalized_close)*(std+1e-5)+mean)*scale+location


def ordinal_stamps(positions):
    """Artificial ordinal calendar, independently reset within each case/event."""
    positions=np.asarray(positions,dtype=np.int64)
    flat=pd.DatetimeIndex(pd.Timestamp('2000-01-01')+pd.to_timedelta(positions.reshape(-1),unit='min'))
    stamp=np.column_stack([flat.minute,flat.hour,flat.weekday,flat.day,flat.month]).astype(np.float32)
    return stamp.reshape(*positions.shape,5)


def input_fingerprint(*arrays):
    """Bind cached predictions to selected contexts, scales and ordinal positions."""
    digest=hashlib.sha256()
    for value in arrays:
        value=np.ascontiguousarray(value)
        digest.update(str((value.shape,value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
