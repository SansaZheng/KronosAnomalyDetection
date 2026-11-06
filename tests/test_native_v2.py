"""Plumbing of the real-weight rerun, exercised with synthetic bars and draws."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from run_native_v2 import NAMES, bundle, features_from_draws, score_arm


def month(seed=0, bars=31 * 288):
    rng = np.random.default_rng(seed)
    close = 5e-5 * np.exp(np.cumsum(rng.normal(scale=0.003, size=bars)))
    frame = pd.DataFrame(dict(timestamps=pd.date_range('2018-12-01', periods=bars, freq='5min')))
    frame['open'] = np.r_[close[0], close[:-1]]
    frame['close'] = close
    frame['high'] = frame[['open', 'close']].max(axis=1) * (1 + np.abs(rng.normal(scale=1e-3, size=bars)))
    frame['low'] = frame[['open', 'close']].min(axis=1) * (1 - np.abs(rng.normal(scale=1e-3, size=bars)))
    frame['volume'] = np.where(rng.uniform(size=bars) < 0.3, 0.0, rng.exponential(2000, size=bars).round())
    frame['amount'] = frame.volume * frame.close
    return frame


def fake_draws(frame, selected, samples=5, seed=1):
    rng = np.random.default_rng(seed)
    last = frame[NAMES].to_numpy()[selected - 1]
    noise = 1 + rng.normal(scale=0.002, size=(len(selected), samples, 6))
    return last[:, None, :] * noise


def test_features_from_draws_are_finite_for_degenerate_draws():
    draws = np.array([[[5e-5, 5e-5, 5e-5, 5e-5, 10.0, 5e-4], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    point, spread = features_from_draws(draws, np.array([5e-5]), 100.0, 1e-3)
    assert point.shape == (1, 4) and spread.shape == (1, 4) and np.isfinite(point).all() and np.isfinite(spread).all()


def test_bundle_partitions_and_scoring_run_end_to_end():
    frame = month()
    selected = np.flatnonzero(frame.timestamps >= pd.Timestamp('2018-12-22'))
    draws = fake_draws(frame, selected)
    b = bundle(frame, selected, draws, lookback=64)
    assert b['x'].shape == (len(selected), 4) and b['windows'].shape == (len(selected), 64, 4)
    assert len(b['fit']) == len(b['validation']) == 864 and len(b['test']) == 1152
    assert b['fit'].max() < b['validation'].min() < b['test'].min()
    assert np.isfinite(b['forecasts']['kronos']).all() and (b['forecasts']['kronos:spread'] >= 0).all()
    # scored bars never see their own row: the window ends one bar earlier
    np.testing.assert_array_equal(b['windows'][10, -1], b['x'][9])
    rows, native = score_arm('synthetic', b)
    table = pd.DataFrame(rows)
    assert set(table.variant) == {'fusion_kronos_spread', 'fusion_kronos_mean', 'fusion_no_kronos'}
    assert (table.alarm_rate < 0.1).all() and table.pump_rank.between(1, 1152).all()
    assert len(native) == 1152 and native.score.notna().all()
