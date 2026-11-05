"""Offline peaks-over-threshold calibration; never updates on test observations."""
import warnings

import numpy as np
from scipy.stats import genpareto


def fit_pot(scores, quantile=0.95, risk=0.001, min_exceedances=30):
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("POT needs a nonempty finite nonnegative score vector")
    if not 0 < quantile < 1 or not 0 < risk < 1 - quantile:
        raise ValueError("Require 0 < risk < 1 - quantile < 1")
    if min_exceedances < 3:
        raise ValueError("min_exceedances must be >= 3")
    u = float(np.quantile(x, quantile))
    excess = x[x > u] - u
    n_tail = len(excess)
    if n_tail < min_exceedances:
        raise ValueError(f"Only {n_tail} tail exceedances; need {min_exceedances}. "
                         "Use more calibration data or a lower initial quantile.")
    tail_fraction = n_tail / len(x)
    if risk >= tail_fraction:
        raise ValueError("risk must be below the empirical exceedance fraction")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        shape, _, scale = genpareto.fit(excess, floc=0)
    if not np.isfinite([shape, scale]).all() or scale <= 0:
        raise ValueError("Invalid GPD fit; no threshold was produced")
    # P(S > D) = P(S > u) * P(Z > D-u); isf is stable near shape=0.
    threshold = float(u + genpareto.isf(risk / tail_fraction, shape, loc=0, scale=scale))
    if not np.isfinite(threshold) or threshold <= u:
        raise ValueError("Invalid POT threshold; inspect the calibration tail")
    return dict(threshold=threshold, initial_threshold=u, shape=float(shape),
                scale=float(scale), n=len(x), n_exceedances=n_tail,
                quantile=quantile, risk=risk)
