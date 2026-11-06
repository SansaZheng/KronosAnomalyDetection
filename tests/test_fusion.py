import numpy as np
import pandas as pd
import pytest

from kronos_ad.backend import prescale_window
from kronos_ad.fusion import (EvidenceFusion, FusionConfig, TailPValue, fit_pinball, fit_stack,
                              simplex_grid, track_threshold)


def stream(n=900, d=3, L=32, seed=0):
    rng = np.random.default_rng(seed)
    series = np.cumsum(rng.normal(size=(n + L, d)), axis=0)
    windows = np.stack([series[i - L:i] for i in range(L, n + L)])
    return series[L:], windows


def test_simplex_grid_sums_to_one():
    grid = simplex_grid(3, 0.05)
    assert len(grid) == 231 and np.allclose(grid.sum(axis=1), 1) and (grid >= 0).all()


def test_stack_ignores_a_useless_forecast():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=500)
    good, noise = truth + rng.normal(scale=0.1, size=500), rng.normal(scale=5, size=500)
    assert fit_stack(truth, np.column_stack([good, noise]))[1] <= 0.05


def test_pinball_recovers_a_positive_scale():
    rng = np.random.default_rng(2)
    scale = rng.uniform(1, 3, 2000)
    target = np.abs(rng.normal(size=2000)) * scale
    coefficient = fit_pinball(scale[:, None], target, 0.75)
    assert coefficient[0] == pytest.approx(1.15, abs=0.15)


def test_pvalue_is_monotone_and_extends_past_one_over_n():
    calibration = np.random.default_rng(3).exponential(size=400)
    tail = TailPValue(calibration)
    grid = np.linspace(0, 50, 200)
    log_p = tail(grid)
    assert (np.diff(log_p) <= 1e-12).all() and (log_p <= 0).all() and tail(np.array([30.0]))[0] < np.log(1 / 401)


def test_pvalues_are_close_to_uniform_on_exchangeable_rows():
    rng = np.random.default_rng(4)
    tail = TailPValue(rng.exponential(size=1000))
    assert (np.exp(tail(rng.exponential(size=20000))) <= 0.05).mean() == pytest.approx(0.05, abs=0.01)


def test_extreme_scores_stay_finite_and_ordered():
    tail = TailPValue(np.random.default_rng(8).exponential(size=300))
    log_p = tail(np.array([1e3, 1e4, 1e6]))
    assert np.isfinite(log_p).all() and log_p[0] > log_p[1] > log_p[2]


def fitted(x, windows, reference=None, **kwargs):
    model = EvidenceFusion(FusionConfig(**kwargs))
    model.fit(x[:300], windows[:300], {}, x[:300] if reference is None else reference)
    return model.calibrate(x[300:600], windows[300:600], {})


def test_scores_are_causal_and_rowwise():
    x, windows = stream()
    model = fitted(x, windows)
    baseline = model.score(x[600:], windows[600:], {})
    changed = x[600:].copy()
    changed[100:] += 1e6
    after = model.score(changed, windows[600:], {})
    assert np.array_equal(baseline[:100], after[:100])


def test_alarm_rate_does_not_inflate_with_dimension():
    x, windows = stream(n=4000, d=20)
    model = EvidenceFusion(FusionConfig(channels=("forecast", "local")))
    model.fit(x[:800], windows[:800], {}, x[:800]).calibrate(x[800:1600], windows[800:1600], {})
    rate = (model.score(x[1600:], windows[1600:], {}) > model.threshold_).mean()
    assert rate < 0.04


def test_injected_shock_ranks_first_and_is_localized():
    x, windows = stream()
    model = fitted(x, windows)
    shocked = x[600:].copy()
    shocked[50, 2] += 40
    score = model.score(shocked, windows[600:], {})
    feature, _ = model.explain(shocked, windows[600:], {})
    assert score.argmax() == 50 and feature[50] == 2


def test_global_mask_removes_that_channel_from_the_vote():
    x, windows = stream()
    model = EvidenceFusion().fit(x[:300], windows[:300], {}, x[:300], global_mask=np.zeros(3, bool))
    assert not model.active_mask()[:, 2].any() and model.active_mask()[:, :2].all()


def test_online_threshold_holds_the_budget_under_drift():
    rng = np.random.default_rng(5)
    scores = rng.exponential(size=6000) * np.linspace(1, 5, 6000)
    flags, _ = track_threshold(scores, rng.exponential(size=512), alpha=0.01, gamma=0.005)
    fixed = scores > np.quantile(rng.exponential(size=512), 0.99)
    assert flags.mean() == pytest.approx(0.01, abs=0.005) and fixed.mean() > 0.1


def test_online_threshold_is_causal():
    rng = np.random.default_rng(6)
    scores, calibration = rng.exponential(size=800), rng.exponential(size=256)
    flags, _ = track_threshold(scores, calibration)
    changed = scores.copy()
    changed[400:] *= 100
    assert np.array_equal(flags[:400], track_threshold(changed, calibration)[0][:400])


def test_prescale_defeats_the_predictor_constant_on_tiny_prices():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"close": 5.3e-5 + rng.normal(scale=2.6e-7, size=64),
                          "volume": rng.normal(1000, 100, size=64)})
    collapsed = frame.close.std(ddof=0) / (frame.close.std(ddof=0) + 1e-5)
    scaled, factor = prescale_window(frame)
    assert collapsed < 0.05
    assert scaled.close.std(ddof=0) / (scaled.close.std(ddof=0) + 1e-5) > 0.9999
    np.testing.assert_allclose(scaled * factor, frame)


def test_prescale_keeps_a_flat_column_finite():
    scaled, factor = prescale_window(pd.DataFrame({"close": [2.0] * 8}))
    assert factor.close == 1.0 and (scaled.close == 2.0).all()


def test_score_is_log_of_inverse_minimum_pvalue():
    x, windows = stream()
    model = fitted(x, windows)
    log_p = model.log_pvalues(x[600:], windows[600:], {})[:, model.active_mask()]
    np.testing.assert_allclose(model.score(x[600:], windows[600:], {}), -log_p.min(axis=1))
