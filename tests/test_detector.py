import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from kronos_ad import Config, Detector, fit_pot


class LastValue:
    max_context = 512

    def __init__(self):
        self.calls = []

    def forecast(self, history, timestamp):
        assert history.timestamps.max() < timestamp
        self.calls.append(history.copy())
        return history.iloc[-1]


def series(n=1300):
    rng = np.random.default_rng(42)
    price = 100 + rng.normal(size=n)
    return pd.DataFrame(dict(timestamps=pd.date_range("2020-01-01", periods=n, freq="h"),
                             open=price, high=price+1, low=price-1, close=price))


def config(**kw):
    return Config(features=("close",), epsilon=(1e-5,), lookback=16, **kw)


def test_scores_causal_and_raw_extremes():
    frame = series(50)
    b = LastValue()
    d = Detector(b, config())
    first = d.score(frame)
    changed = frame.copy()
    changed.loc[30:, "close"] += 1e6
    second = d.score(changed)
    pd.testing.assert_frame_equal(first.iloc[:30], second.iloc[:30])
    expected = ((changed.close[30]-changed.close[29]) /
                (changed.close[14:30].std(ddof=0)+1e-5))**2
    assert second.score_close[30] == pytest.approx(expected)
    assert len(b.calls[0]) == 16
    assert first.scored.iloc[:16].eq(False).all()
    assert first.score_close.iloc[:16].isna().all()


def test_frozen_thresholds_ordering_and_warmup(tmp_path):
    frame = series()
    d = Detector(LastValue(), config()).fit(frame.iloc[20:1100], frame.iloc[:20])
    before = d.state["tails"]["close"]["threshold"]
    result = d.predict(frame.iloc[1100:])
    assert result.anomaly.iloc[:16].isna().all()
    assert result.threshold_close.eq(before).all()
    assert d.state["tails"]["close"]["threshold"] == before
    assert 0 <= d.state["calibration_joint_alarm_rate"] <= 1
    d.save(tmp_path / "state.json")
    with pytest.raises(ValueError, match="strictly after"):
        d.predict(frame.iloc[1000:])
    with pytest.raises(ValueError, match="strictly earlier"):
        d.score(frame.iloc[100:], frame.iloc[:101])


def test_missing_or_synthetic_features_are_excluded():
    frame = series(40)
    with pytest.raises(ValueError, match="real input"):
        Detector(LastValue()).score(frame)
    result = Detector(LastValue(), config()).score(frame)
    assert "score_volume" not in result
    assert "score_amount" not in result


def test_validation_and_constant_windows():
    frame = series(40)
    frame["close"] = 1.0
    frame.loc[25, "close"] = 100
    result = Detector(LastValue(), config()).score(frame)
    assert np.isfinite(result.score_close.dropna()).all()
    assert result.score_close[25] == pytest.approx((99/1e-5)**2)
    with pytest.raises(ValueError, match="max_context"):
        Detector(LastValue(), Config(lookback=513))
    with pytest.raises(ValueError, match="unique"):
        Detector(LastValue(), config()).score(pd.concat([frame, frame]))
    frame.loc[5, "close"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        Detector(LastValue(), config()).score(frame)


def test_pot_known_exponential_tail():
    x = np.random.default_rng(7).exponential(size=100000)
    tail = fit_pot(x, risk=0.001)
    assert tail["threshold"] == pytest.approx(-np.log(0.001), rel=0.08)
    expected = tail["initial_threshold"] + genpareto.isf(
        0.001/(tail["n_exceedances"]/tail["n"]), tail["shape"], scale=tail["scale"])
    assert tail["threshold"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="tail exceedances"):
        fit_pot(np.ones(1000))
    with pytest.raises(ValueError, match="risk"):
        fit_pot(x, risk=0.1)


def test_injected_spike_detected_and_localized():
    frame = series()
    d = Detector(LastValue(), config()).fit(frame.iloc[20:1100], frame.iloc[:20])
    test = frame.iloc[1100:].copy().reset_index(drop=True)
    test.loc[10, "close"] += 1000
    result = d.predict(test, frame.iloc[:1100])
    assert result.anomaly_close[10]
    assert result.anomaly[10]


def test_multivariate_or_and_threshold_equality():
    frame = series()
    c = Config(features=("open", "close"), epsilon=(1e-5, 1e-5), lookback=16)
    d = Detector(LastValue(), c).fit(frame.iloc[20:1100], frame.iloc[:20])
    test = frame.iloc[1100:].copy().reset_index(drop=True)
    test.loc[0, "close"] += 1e5
    result = d.predict(test, frame.iloc[:1100])
    assert result.anomaly_close[0] and not result.anomaly_open[0]
    assert result.anomaly[0]
    # An exact tie must trigger >=, not >.
    d.state["tails"]["open"]["threshold"] = result.score_open[1]
    assert d.predict(test, frame.iloc[:1100]).anomaly_open[1]
