"""Rebuild the published splits around the cached real Kronos forecasts; needs no torch.

Every bundle separates rows from labels. `y` is read by evaluation code only, and the
fit and calibration partitions are the ones the tuning experiment already used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from kronos_ad.proxy import select_targets
from run_benchmarks import encode, pump, tabular, windows

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/2026-11-06'
PROXY = RESULTS / 'kronos_proxy'
TABULAR = ['ulb_creditcard', 'banksim', 's_ffsd']
PUMP = ['sapienza_5s', 'sapienza_15s', 'sapienza_25s']
SD = ['sd_crypto', 'sd_stock', 'sd_prices', 'sd_synthetic']
PROXY_NAMES = TABULAR + PUMP + SD
NATIVE = 'native_brdbtc'
SD_FILES = {'sd_crypto': ('cryptocurrency cases', 'test_data.npy', 'test_label.npy'),
            'sd_stock': ('stock cases', 'test_data_bs.npy', 'test_label_bs.npy'),
            'sd_prices': ('prices', 'test_data_02.npy', 'test_label_02.npy'),
            'sd_synthetic': ('systhesis', 'all_DTWsim_data.npy', 'all_DTWsim_label.npy')}
# Calendar encodings and the symbol code describe when and what, not market behavior.
NOT_DETECTION_TARGETS = {'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'symbol_reference_frequency'}


def load_rows(name):
    raw = ROOT / 'data/raw'
    if name in TABULAR:
        bundle, grouped = tabular(raw, name), False
    elif name in PUMP:
        bundle, grouped = pump(raw, int(name.split('_')[1][:-1])), True
    else:
        bundle, grouped = windows(raw, *SD_FILES[name]), True
    frame, labels, parts, ids, protocol = bundle
    groups = np.array([str(v).split(':', 1)[0] for v in ids]) if grouped else np.repeat('stream', len(frame))
    values, features = encode(frame, parts)
    return dict(name=name, values=values, features=features, labels=labels, parts=parts,
                groups=groups, ids=ids, protocol=protocol)


def split_calibration(positions, groups):
    """Midpoint split that keeps groups whole, matching tune_kronos_proxy.py."""
    unique = list(dict.fromkeys(groups))
    if len(unique) > 1:
        boundary = positions[len(positions) // 2]
        early = [g for g in unique if positions[groups == g].max() < boundary]
        late = [g for g in unique if positions[groups == g].min() >= boundary]
        return np.isin(groups, early), np.isin(groups, late)
    first = np.zeros(len(positions), bool)
    first[:len(positions) // 2 + len(positions) % 2] = True
    return first, ~first


def proxy_bundle(name):
    rows = load_rows(name)
    targets, histories, _ = select_targets(rows['parts'], rows['groups'], 32, 512, 1024)
    with np.load(PROXY / (name + '__predictions.npz'), allow_pickle=False) as cached:
        np.testing.assert_array_equal(cached['targets'], targets)
        predictions = cached['predictions'].copy()
    values = rows['values']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in rows['features']])
    split = rows['parts'][targets]
    calibration, test = np.flatnonzero(split == 'calibration'), np.flatnonzero(split == 'test')
    fit, validation = split_calibration(targets[calibration], rows['groups'][targets[calibration]])
    return dict(name=name, x=values[targets][:, keep], windows=values[histories][:, :, keep],
                forecasts={'kronos': predictions[:, keep]}, fit=calibration[fit],
                validation=calibration[validation], test=test,
                reference=values[rows['parts'] == 'reference'][:, keep], y=rows['labels'][targets],
                groups=rows['groups'][targets], features=[f for f, k in zip(rows['features'], keep) if k],
                # SD cases are min-max scaled over their whole length, so level is not causal.
                global_mask=np.full(int(keep.sum()), name not in SD), block=64,
                raw=dict(x=values[targets], windows=values[histories], forecast=predictions,
                         calibration=calibration, features=rows['features'], parts=rows['parts'],
                         values=values, targets=targets))


def bar_features(frame, previous_close, volume_unit, amount_unit):
    """Stationary, nearly uncorrelated views of a bar."""
    return np.column_stack([np.log(frame[:, 3] / previous_close),
                            np.log(frame[:, 1] / frame[:, 2]),
                            np.log1p(np.maximum(frame[:, 4], 0) / volume_unit),
                            np.log1p(np.maximum(frame[:, 5], 0) / amount_unit)])


def native_bundle(lookback=64):
    rows = pd.read_csv(RESULTS / 'kronos/detections.csv.gz', parse_dates=['timestamps'])
    names = ['open', 'high', 'low', 'close', 'volume', 'amount']
    bars = rows[names].to_numpy()
    forecast = rows[['prediction_' + n for n in names]].to_numpy()
    calibration = (rows.split == 'calibration').to_numpy()
    previous = np.r_[bars[0, 3], bars[:-1, 3]]
    early = bars[:864]
    volume_unit = np.median(early[early[:, 4] > 0, 4])
    amount_unit = np.median(early[early[:, 5] > 0, 5])
    forecast_bar = forecast.copy()
    forecast_bar[:, 1] = forecast[:, [0, 1, 3]].max(axis=1)
    forecast_bar[:, 2] = forecast[:, [0, 2, 3]].min(axis=1)
    observed = bar_features(bars, previous, volume_unit, amount_unit)
    predicted = bar_features(forecast_bar, previous, volume_unit, amount_unit)
    index = np.arange(lookback + 1, len(rows))
    history = np.stack([observed[i - lookback:i] for i in index])
    calibration_rows = np.flatnonzero(calibration[index])
    half = len(calibration_rows) // 2
    fit = calibration_rows[:half]
    return dict(name=NATIVE, x=observed[index], windows=history, forecasts={'kronos': predicted[index]},
                fit=fit, validation=calibration_rows[half:], test=np.flatnonzero(~calibration[index]),
                reference=observed[index][fit], y=rows.weak_label.to_numpy()[index],
                groups=np.repeat('stream', len(index)), features=['log_return', 'log_range', 'log_volume', 'log_amount'],
                global_mask=np.ones(4, bool), block=64, timestamps=rows.timestamps.to_numpy()[index],
                bars=rows.iloc[index].reset_index(drop=True), offset=lookback + 1)
