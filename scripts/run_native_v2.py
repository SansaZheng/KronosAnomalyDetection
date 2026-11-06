#!/usr/bin/env python3
"""BRD/BTC rerun with real Kronos weights: prescale on/off, draws kept, fusion scoring.

Pre-registered in results/2026-11-06/kronos_native_v2/PREREGISTRATION.md. Inputs are
the files pinned by results/2026-11-06/input_manifest.json (official code, weights,
Binance archive); nothing is downloaded here. Each arm's draws are cached with the
weight hashes, so scoring can be replayed without Torch.

    PYTHONPATH=.:scripts python scripts/run_native_v2.py --device cpu
    PYTHONPATH=.:scripts python scripts/run_native_v2.py --score-only
"""
import argparse
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd

from fusion_data import RESULTS, ROOT, bar_features
from kronos_ad.fusion import EvidenceFusion, FusionConfig, mad, track_threshold
from run_fusion import ALPHA, DELTAS, FROZEN as FUSION_FROZEN, ONLINE, evaluate, step_scale, subset

OUT = RESULTS / 'kronos_native_v2'
NAMES = ['open', 'high', 'low', 'close', 'volume', 'amount']
ARMS = {'prescale': True, 'raw': False}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_inputs():
    manifest = json.loads((RESULTS / 'input_manifest.json').read_text())
    missing = []
    for record in manifest['files']:
        path = ROOT / record['path']
        if not path.exists():
            missing.append(record)
        elif sha256(path) != record['sha256']:
            raise ValueError(f'{path} differs from the pinned experiment input')
    if missing:
        lines = '\n'.join(f"  {r['path']}  <-  {r['url']}" for r in missing)
        raise SystemExit('Missing pinned inputs (place them at these paths; SHA256 is checked):\n' + lines)
    return manifest


def load_bars(path=ROOT / 'runs/market/BRDBTC-5m-2018-12.zip'):
    """Same loader as run_kronos_case.py, without importing Torch."""
    import zipfile
    checksum = path.with_name(path.name + '.CHECKSUM').read_text().split()[0]
    if sha256(path) != checksum:
        raise ValueError('Binance published SHA256 mismatch')
    with zipfile.ZipFile(path) as z:
        if z.testzip():
            raise ValueError('ZIP CRC mismatch')
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
    frame = raw.iloc[:, [0, 1, 2, 3, 4, 5, 7]].copy()
    frame.columns = ['timestamps', *NAMES]
    frame.timestamps = pd.to_datetime(frame.timestamps, unit='ms')
    if frame.timestamps.duplicated().any() or not frame.timestamps.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise ValueError('Missing, duplicate or irregular market bars')
    if not np.isfinite(frame[NAMES]).all().all():
        raise ValueError('Nonfinite market observations')
    return frame


def run_arm(name, prescale, frame, selected, a, spec):
    from kronos_ad.sampling import forecast_samples, load_official
    import torch
    cache = OUT / f'draws_{name}.npz'
    meta = OUT / f'draws_{name}.json'
    if cache.exists():
        if json.loads(meta.read_text()) != spec:
            raise ValueError(f'{cache} was produced with other parameters; move it away')
        with np.load(cache, allow_pickle=False) as z:
            return z['draws']
    predictor, official = load_official(ROOT / 'vendor/Kronos', ROOT / 'weights/Kronos-mini',
                                        ROOT / 'weights/Kronos-Tokenizer-2k', a.device, 512)
    torch.manual_seed(spec['seed'])
    draws, started = [], time.monotonic()
    for start in range(0, len(selected), a.batch_size):
        index = selected[start:start + a.batch_size]
        windows = [frame.iloc[i - a.lookback:i].reset_index(drop=True) for i in index]
        draws.append(forecast_samples(predictor, official, windows, [w.timestamps for w in windows],
                                      [pd.Series([frame.timestamps.iloc[i]]) for i in index],
                                      spec['sample_count'], prescale=prescale, T=spec['T'], top_k=0, top_p=spec['top_p']))
        done = min(start + a.batch_size, len(selected))
        if start == 0 or done % (a.batch_size * 10) == 0 or done == len(selected):
            print(name, done, '/', len(selected), round(time.monotonic() - started, 1), 'sec', flush=True)
    draws = np.concatenate(draws)
    if not np.isfinite(draws).all():
        raise ValueError('Nonfinite draws')
    np.savez_compressed(cache, draws=draws, timestamps=frame.timestamps.iloc[selected].to_numpy().astype('datetime64[ns]').astype(np.int64))
    meta.write_text(json.dumps(spec, indent=2) + '\n')
    return draws


def features_from_draws(draws, previous_close, volume_unit, amount_unit):
    """Median and robust spread (IQR / 1.349) over draws of the transformed features.

    Draws are made consistent before transforming (high >= open, close; low <= open,
    close; prices positive) so that a stray draw cannot turn into a NaN.
    """
    n, s, _ = draws.shape
    flat = draws.reshape(n * s, 6).copy()
    flat[:, 1] = flat[:, [0, 1, 3]].max(axis=1)
    flat[:, 2] = flat[:, [0, 2, 3]].min(axis=1)
    floor = 1e-3 * np.repeat(previous_close, s)
    flat[:, 3] = np.maximum(flat[:, 3], floor)
    flat[:, 1] = np.maximum(flat[:, 1], flat[:, 3])
    flat[:, 2] = np.clip(flat[:, 2], floor, flat[:, 1])
    transformed = bar_features(flat, np.repeat(previous_close, s), volume_unit, amount_unit).reshape(n, s, 4)
    quartiles = np.quantile(transformed, [0.25, 0.5, 0.75], axis=1)
    return quartiles[1], (quartiles[2] - quartiles[0]) / 1.349


def bundle(frame, selected, draws, lookback):
    bars = frame[NAMES].to_numpy()
    previous = np.r_[bars[0, 3], bars[:-1, 3]]
    calibration_bars = bars[selected][:864]
    volume_unit = np.median(calibration_bars[calibration_bars[:, 4] > 0, 4])
    amount_unit = np.median(calibration_bars[calibration_bars[:, 5] > 0, 5])
    observed = bar_features(bars, previous, volume_unit, amount_unit)
    point, spread = features_from_draws(draws, previous[selected], volume_unit, amount_unit)
    forecast = np.full_like(observed, np.nan)
    forecast[selected] = point
    spread_full = np.full_like(observed, np.nan)
    spread_full[selected] = spread
    index = selected  # histories come from the bars before 2018-12-22, so every scored bar is kept
    history = np.stack([observed[i - lookback:i] for i in index])
    calibration = (frame.timestamps.iloc[index] < pd.Timestamp('2018-12-28')).to_numpy()
    cal_rows = np.flatnonzero(calibration)
    half = len(cal_rows) // 2
    return dict(name='native_brdbtc_v2', x=observed[index], windows=history,
                forecasts={'kronos': forecast[index], 'kronos:spread': spread_full[index]},
                fit=cal_rows[:half], validation=cal_rows[half:], test=np.flatnonzero(~calibration),
                reference=observed[index][cal_rows[:half]], groups=np.repeat('stream', len(index)),
                features=['log_return', 'log_range', 'log_volume', 'log_amount'], global_mask=np.ones(4, bool),
                timestamps=frame.timestamps.to_numpy()[index], bars=frame.iloc[index].reset_index(drop=True), index=index)


def fusion_config():
    return FusionConfig(channels=tuple(FUSION_FROZEN['channels']), naive=tuple(FUSION_FROZEN['naive_forecasts']),
                        tail_quantile=FUSION_FROZEN['tail_quantile'], scale_quantile=FUSION_FROZEN['scale_quantile'],
                        aggregate=FUSION_FROZEN['aggregate'], alpha=ALPHA)


def fit_variant(b, forecasts):
    fit, validation = b['fit'], b['validation']
    model = EvidenceFusion(fusion_config()).fit(b['x'][fit], b['windows'][fit], subset(forecasts, fit), b['reference'], b['global_mask'])
    return model.calibrate(b['x'][validation], b['windows'][validation], subset(forecasts, validation))


def skill_table(frame, selected, draws_by_arm, lookback):
    """E1/E2: forecast error against persistence, raw bars and transformed features."""
    bars = frame[NAMES].to_numpy()
    rows = []
    calibration = (frame.timestamps.iloc[selected] < pd.Timestamp('2018-12-28')).to_numpy()
    old = pd.read_csv(RESULTS / 'kronos/detections.csv.gz')
    arms = {name: draws.mean(axis=1) for name, draws in draws_by_arm.items()}
    arms['published_s3'] = old[['prediction_' + n for n in NAMES]].to_numpy()
    for arm, mean in arms.items():
        for part, mask in [('calibration', calibration), ('test', ~calibration)]:
            idx = selected[mask]
            kronos = np.abs(bars[idx] - mean[mask]).mean(axis=0)
            persistence = np.abs(bars[idx] - bars[idx - 1]).mean(axis=0)
            rows.append(dict(arm=arm, part=part, bars=len(idx), **{f'mae_ratio_{n}': kronos[j] / persistence[j] for j, n in enumerate(NAMES)}))
    return pd.DataFrame(rows)


def pump_shock(b, index, delta):
    scale = step_scale(b)
    x = b['x'][index].copy()
    x += delta * scale
    return x


def score_arm(name, b):
    variants = {'fusion_kronos_spread': b['forecasts'],
                'fusion_kronos_mean': {'kronos': b['forecasts']['kronos']},
                'fusion_no_kronos': {}}
    rows, native_rows = [], None
    test = b['test']
    pump = np.flatnonzero(b['bars'].timestamps.to_numpy() == np.datetime64('2018-12-30T17:00:00'))[0]
    pump_in_test = pump - test[0]
    for variant, forecasts in variants.items():
        model = fit_variant(b, forecasts)
        score = model.score(b['x'][test], b['windows'][test], subset(forecasts, test))
        flags = score > model.threshold_
        online, _ = track_threshold(score, model.calibration_scores_, ALPHA, ONLINE['gamma'], ONLINE['window'])
        base_cut = np.quantile(score, .99)
        power = {f'power_delta{d}': float((model.score(pump_shock(b, test, d), b['windows'][test], subset(forecasts, test)) > base_cut).mean()) for d in (3, 6, 12)}
        zero_volume = (b['bars'].volume.to_numpy()[test] == 0)
        state = model.state()
        rows.append(dict(arm=name, variant=variant, test_bars=len(test), alarms=int(flags.sum()), alarm_rate=float(flags.mean()),
                         online_alarms=int(online.sum()), alarms_on_zero_volume=int((flags & zero_volume).sum()),
                         pump_alarm=bool(flags[pump_in_test]), pump_rank=int((score > score[pump_in_test]).sum() + 1),
                         kronos_weight_mean=float(model.weights_[:, -1].mean()) if 'kronos' in model.forecast_names_ else 0.0,
                         kronos_weights=json.dumps(dict(zip(b['features'], np.round(model.weights_[:, -1], 3).tolist()))) if 'kronos' in model.forecast_names_ else '',
                         spread_coefficient_forecast=json.dumps(np.round(model.scale_coef_['forecast'][:, -2], 4).tolist()) if 'kronos:spread' in forecasts else '',
                         threshold=model.threshold_, **power))
        if variant == 'fusion_kronos_spread':
            feature, channel = model.explain(b['x'][test], b['windows'][test], subset(forecasts, test))
            native_rows = pd.DataFrame(dict(timestamp=b['bars'].timestamps.to_numpy()[test], close=b['bars'].close.to_numpy()[test],
                                            volume=b['bars'].volume.to_numpy()[test], score=score, alarm=flags, online_alarm=online,
                                            top_feature=np.array(b['features'])[feature], top_channel=np.array(model.config.channels)[channel]))
    return rows, native_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lookback', type=int, default=64)
    p.add_argument('--sample-count', type=int, default=20)
    p.add_argument('--score-only', action='store_true', help='replay cached draws without Torch')
    a = p.parse_args()
    warnings.filterwarnings('ignore')
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((OUT / 'frozen_config.json').read_text())
    if (a.lookback, a.sample_count) != (frozen['lookback'], frozen['sample_count']):
        raise SystemExit('Arguments differ from the frozen configuration')
    manifest = pinned_inputs()
    frame = load_bars()
    selected = np.flatnonzero(frame.timestamps >= pd.Timestamp('2018-12-22'))
    weights = {r['path']: r['sha256'] for r in manifest['files'] if r['path'].startswith('weights/')}
    draws_by_arm = {}
    for name, prescale in ARMS.items():
        spec = dict(arm=name, prescale=prescale, lookback=a.lookback, sample_count=a.sample_count, seed=42, T=1.0, top_p=0.9,
                    top_k=0, first_scored='2018-12-22', device=a.device, batch_size=a.batch_size, weights=weights)
        if a.score_only:
            with np.load(OUT / f'draws_{name}.npz', allow_pickle=False) as z:
                draws_by_arm[name] = z['draws']
        else:
            draws_by_arm[name] = run_arm(name, prescale, frame, selected, a, spec)
    skill = skill_table(frame, selected, draws_by_arm, a.lookback)
    skill.to_csv(OUT / 'forecast_skill.csv', index=False)
    rows = []
    for name, draws in draws_by_arm.items():
        b = bundle(frame, selected, draws, a.lookback)
        arm_rows, native_rows = score_arm(name, b)
        rows.extend(arm_rows)
        native_rows.to_csv(OUT / f'native_rows_{name}.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'detection.csv', index=False)
    pd.set_option('display.width', 250)
    print(skill.round(3).to_string())
    print(table.drop(columns=['kronos_weights', 'spread_coefficient_forecast']).round(3).to_string())


if __name__ == '__main__':
    main()
