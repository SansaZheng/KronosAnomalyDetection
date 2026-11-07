import numpy as np
import pytest

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from kronos_ad.boundary import (CNNProbe, KnobPredictor, LogisticProbe, MLPProbe, balanced_accuracy,
                                difference_map, dimension_energy, hidden_patch, intervention_directions, knee,
                                layer_patches, nested_knob_search, rank_dimensions, split_balanced, summary_metrics)


def planted(n=600, L=6, d=40, seed=0, knobs=(7, 23), strength=1.5):
    """Hidden states whose label is carried by two dimensions in the later layers."""
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, L, d))
    y = rng.integers(0, 2, n)
    for j in knobs:
        H[:, L // 2:, j] += strength * (2 * y - 1)[:, None]
    return H, y


def test_split_is_balanced_and_disjoint():
    y = np.r_[np.ones(100), np.zeros(400)].astype(int)
    train, test = split_balanced(y, 0.2, seed=1)
    assert len(train) + len(test) == 200 and not set(train) & set(test)
    assert y[np.r_[train, test]].mean() == pytest.approx(0.5)


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_probes_learn_a_planted_boundary():
    H, y = planted()
    train, test = split_balanced(y, 0.25)
    for probe in (LogisticProbe(), MLPProbe(max_iter=200), CNNProbe(epochs=60)):
        probe.fit(H[train], y[train])
        assert balanced_accuracy(probe, H[test], y[test]) > 0.8


def test_difference_map_localizes_the_planted_dimensions_and_layers():
    H, y = planted()
    delta = difference_map(H, y)
    ranking = rank_dimensions(delta)
    assert set(ranking[:2]) == {7, 23}
    metrics = summary_metrics(delta)
    assert metrics['late_ratio'] > 0.6 and metrics['snr_db'] > 10
    energy = np.sort(dimension_energy(delta))[::-1]
    assert knee(energy) <= 3


def test_knee_of_a_long_tailed_profile():
    profile = np.r_[np.array([10, 8, 6]), np.full(97, 0.1)]
    assert knee(profile) in (2, 3)


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_knob_search_finds_a_two_dimensional_near_optimal_set():
    H, y = planted()
    train, test = split_balanced(y, 0.25)
    full = LogisticProbe().fit(H[train], y[train])
    full_accuracy = balanced_accuracy(full, H[test], y[test])
    ranking = rank_dimensions(difference_map(H[train], y[train]))
    result = nested_knob_search(H[train], y[train], H[test], y[test], ranking, full_accuracy, delta=0.03, k_max=4,
                                make_predictor=lambda: KnobPredictor(epochs=150), random_trials=5)
    assert result.knob_k <= 2 and set(result.knob_dims) <= {7, 23}
    assert result.random_control['mean'] < result.accuracies[result.knob_k] - 0.1


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_knob_predictor_closed_form_and_attention():
    H, y = planted()
    predictor = KnobPredictor(epochs=150).fit(H, y, [7, 23])
    assert balanced_accuracy(predictor, H, y) > 0.85
    alpha = predictor.alpha()
    assert alpha.shape == (6,) and alpha.sum() == pytest.approx(1.0, abs=1e-5)
    text = predictor.closed_form(['z1', 'z2'])
    assert text.startswith('Phi = ') and text.count('log(') == 4


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_layer_patch_reproduces_the_requested_pooled_shift():
    H, y = planted()
    predictor = KnobPredictor(epochs=50).fit(H, y, [7, 23])
    z = predictor.pooled(H[:5])
    directions, norm = intervention_directions(predictor, z)
    assert np.allclose(np.linalg.norm(directions['normal'], axis=1), 1, atol=1e-4)
    assert np.allclose((directions['normal'] * directions['tangent']).sum(axis=1), 0, atol=1e-5)
    patch = hidden_patch(predictor, H[:5], tau=0.5, direction_name='normal')
    assert patch.shape == H[:5].shape and np.count_nonzero(patch[:, :, [0, 1, 2]]) == 0
    shifted = predictor.pooled(H[:5] + patch)
    expected = z + 0.5 / (norm[:, None] + 1e-6) * directions['normal']
    assert np.allclose(shifted, expected, atol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_normal_direction_moves_the_score_most():
    H, y = planted()
    predictor = KnobPredictor(epochs=150).fit(H, y, [7, 23])
    rows = H[:40]
    base = predictor.score(rows)
    effects = {}
    for name in ('normal', 'tangent', 'random'):
        effects[name] = (predictor.score(rows + hidden_patch(predictor, rows, 0.3, name)) - base).mean()
    assert effects['normal'] > abs(effects['tangent']) + 0.05 and effects['normal'] > abs(effects['random'])
