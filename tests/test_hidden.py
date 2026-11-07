import numpy as np
import pytest

torch = pytest.importorskip('torch')

from kronos_ad.boundary import KnobPredictor, hidden_patch
from kronos_ad.hidden import KronosHiddenStates, intervention_effect, normalize_windows
from tests.vendor.kronos import Kronos, KronosTokenizer


@pytest.fixture(scope='module')
def tiny():
    torch.manual_seed(0)
    model = Kronos(s1_bits=4, s2_bits=4, n_layers=3, d_model=32, n_heads=4, ff_dim=64, ffn_dropout_p=0.0,
                   attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True)
    tokenizer = KronosTokenizer(d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
                                ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=4, s2_bits=4,
                                beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=1)
    states = KronosHiddenStates(model, tokenizer)
    rng = np.random.default_rng(1)
    windows = np.cumsum(rng.normal(size=(8, 12, 6)), axis=1)
    x = normalize_windows(windows[:, :11])
    stamp = torch.zeros(8, 11, 5, dtype=torch.long)
    s1, s2 = states.tokens(x)
    x_next = normalize_windows(windows)[:, -1]
    s1n, s2n = states.next_tokens(x, x_next)
    return states, x, stamp, s1, s2, s1n, s2n


def test_normalize_windows_matches_the_official_rule():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(2, 10, 6)) * 3 + 1
    z = normalize_windows(w)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-6) and np.abs(z).max() <= 5


def test_hidden_states_have_one_row_per_layer_plus_embedding(tiny):
    states, x, stamp, s1, s2, _, _ = tiny
    H = states.hidden_states(s1, s2, stamp)
    assert H.shape == (8, 4, 32) and np.isfinite(H).all()
    assert not np.allclose(H[:, 0], H[:, 1])


def test_zero_patch_leaves_the_likelihood_unchanged_and_nonzero_patch_changes_it(tiny):
    states, x, stamp, s1, s2, s1n, s2n = tiny
    base = states.loglik(s1, s2, stamp, s1n, s2n)
    assert base.shape == (8,) and (base <= 0).all()
    assert np.allclose(states.loglik(s1, s2, stamp, s1n, s2n, patch=np.zeros((8, 4, 32))), base)
    patch = np.zeros((8, 4, 32))
    patch[:, 2, :] = 3.0
    assert not np.allclose(states.loglik(s1, s2, stamp, s1n, s2n, patch=patch), base)


def test_patch_only_touches_the_last_position(tiny):
    states, x, stamp, s1, s2, _, _ = tiny
    patch = np.zeros((8, 4, 32))
    patch[:, 1, :] = 5.0
    handles = states._hooks(patch=patch)
    with torch.no_grad():
        logits_patched, _ = states.model.decode_s1(s1, s2, stamp)
    for h in handles:
        h.remove()
    with torch.no_grad():
        logits, _ = states.model.decode_s1(s1, s2, stamp)
    assert torch.allclose(logits[:, :-1], logits_patched[:, :-1])
    assert not torch.allclose(logits[:, -1], logits_patched[:, -1])


def test_intervention_pipeline_runs_end_to_end(tiny):
    states, x, stamp, s1, s2, s1n, s2n = tiny
    H = states.hidden_states(s1, s2, stamp)
    y = (np.arange(8) % 2)
    predictor = KnobPredictor(epochs=20).fit(H, y, [3, 9])
    patch = hidden_patch(predictor, H, tau=0.2, direction_name='normal')
    effect = intervention_effect(states, s1, s2, stamp, s1n, s2n, patch)
    assert effect.shape == (8,) and np.isfinite(effect).all()
