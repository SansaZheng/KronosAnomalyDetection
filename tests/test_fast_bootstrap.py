"""kronos_ad.fast_bootstrap must reproduce sklearn's AP and run_fusion.recall_at_fpr on every resample."""
import sys
from pathlib import Path

import numpy as np
import pytest

sklearn = pytest.importorskip('sklearn')
from sklearn.metrics import average_precision_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kronos_ad.fast_bootstrap import paired_bootstrap  # noqa: E402
from run_fusion import recall_at_fpr, resamples  # noqa: E402


def test_matches_sklearn_and_recall_at_fpr_with_ties_and_repeats():
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.03).astype(int)
    scores = {'ties': np.round(rng.standard_normal(n) + 2 * y, 1), 'cont': rng.standard_normal(n) + y, 'coarse': np.round(rng.random(n) * 4)}
    groups = np.repeat(np.arange(60), 50)
    idxs = list(resamples(groups, np.random.default_rng(1), 64, 100))
    fast = paired_bootstrap(y, scores, iter(idxs))
    for k, v in scores.items():
        slow = []
        for idx in idxs:
            yy = y[idx]
            if 0 < yy.sum() < len(yy):
                slow.append((average_precision_score(yy, v[idx]), recall_at_fpr(yy, v[idx])))
        assert np.allclose(np.array(slow), fast[k], atol=1e-12)


def test_resamples_without_both_classes_are_skipped_for_every_score():
    y = np.array([0, 0, 0, 1])
    scores = {'a': np.array([0.1, 0.2, 0.3, 0.9]), 'b': np.array([3.0, 2.0, 1.0, 0.0])}
    idxs = [np.array([0, 1, 2]), np.array([0, 1, 2, 3]), np.array([3, 3])]
    out = paired_bootstrap(y, scores, iter(idxs))
    assert out['a'].shape == (1, 2) and out['b'].shape == (1, 2)
    assert out['a'][0, 0] == 1.0 and out['b'][0, 0] == 0.25
