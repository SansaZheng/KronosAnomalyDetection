"""Unit tests for the miss-rate experiment helpers (scripts/boundary_missrate.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from boundary_missrate import bootstrap_units, conformal_logp, fisher, paired_bootstrap, threshold_at_fpr, tippett  # noqa: E402


def test_conformal_logp_is_monotone_and_finite_beyond_the_null():
    rng = np.random.default_rng(0)
    null = rng.standard_normal(2000)
    scores = np.array([-3.0, 0.0, 2.0, 5.0, 50.0])
    logp = conformal_logp(scores, null)
    assert np.all(np.diff(logp) < 0)                 # more anomalous -> smaller p
    assert np.all(np.isfinite(logp)) and logp[-1] < logp[-2]   # GPD tail keeps separating beyond the null maximum
    assert abs(np.exp(logp[1]) - 0.5) < 0.05        # the median scores p ~ 0.5


def test_tippett_ignores_the_second_channel_and_fisher_does_not():
    strong_only = np.log(np.array([[1e-4], [0.5]]))
    strong_and_moderate = np.log(np.array([[1e-4], [0.02]]))
    assert np.isclose(tippett(strong_only), tippett(strong_and_moderate))   # min p is the same
    assert fisher(strong_and_moderate) > fisher(strong_only)                 # the second channel adds evidence
    assert np.isclose(fisher(strong_and_moderate), -2 * (np.log(1e-4) + np.log(0.02)))


def test_threshold_at_fpr_hits_the_budget():
    rng = np.random.default_rng(1)
    score = rng.standard_normal(10000)
    normal = np.ones(10000, bool)
    thr = threshold_at_fpr(score, normal, 0.01)
    assert abs((score > thr).mean() - 0.01) < 0.002


def test_bootstrap_units_fall_back_to_blocks_on_a_single_stream():
    rows = np.arange(0, 1000, 4)
    units = bootstrap_units(np.repeat('stream', len(rows)), rows)
    assert len(np.unique(units)) == 50
    many = np.array([f'g{i}' for i in range(30)])
    assert bootstrap_units(many, np.arange(30)) is many


def test_paired_bootstrap_brackets_a_known_difference():
    rng = np.random.default_rng(2)
    groups = np.repeat([f'g{i}' for i in range(40)], 5)
    a = rng.random(200) < 0.5
    b = a | (rng.random(200) < 0.3)
    lo, hi, units = paired_bootstrap(a, b, groups, resamples=500, seed=0)
    assert units == 40 and lo > 0 and lo < (b.mean() - a.mean()) < hi
