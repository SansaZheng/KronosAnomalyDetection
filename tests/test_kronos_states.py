"""Unit tests for scripts/kronos_states.py on a random-weight miniature Kronos (tests/vendor)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.vendor import kronos as official  # noqa: E402
from kronos_states import (EPS, KronosStates, denormalize, native_sequences, normalize_full, proxy_pool, proxy_sequences,  # noqa: E402
                           proxy_stamps, self_test)


@pytest.fixture(scope='module')
def states():
    torch.manual_seed(0)
    model = official.Kronos(s1_bits=4, s2_bits=4, n_layers=3, d_model=32, n_heads=4, ff_dim=64, ffn_dropout_p=0.0,
                            attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True)
    tokenizer = official.KronosTokenizer(d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
                                         ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=4, s2_bits=4,
                                         beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=1)
    return KronosStates(model, tokenizer, official, seed=1)


def test_normalize_full_uses_context_statistics_and_round_trips():
    rng = np.random.default_rng(0)
    full = (1 + 0.2 * rng.standard_normal((3, 33, 6))) * 1e-5   # BTC-quoted scale, no extreme values
    x, mean, std, factor = normalize_full(full, prescale=True)
    assert x.shape == (3, 33, 6) and np.abs(x).max() <= 5
    context = x[:, :-1]
    assert np.allclose(context.mean(axis=1), 0, atol=1e-3) and np.allclose(context.std(axis=1), 1, atol=0.05)
    back = denormalize(x[:, -1], (mean, std, factor))
    clipped = np.abs(x[:, -1]) >= 5
    assert np.allclose(back[~clipped], full[:, -1][~clipped], rtol=1e-4)


def test_prescale_keeps_a_low_priced_series_from_flattening():
    rng = np.random.default_rng(1)
    full = (1 + 0.01 * rng.standard_normal((2, 33, 6))) * 5e-5
    x_raw, *_ = normalize_full(full, prescale=False)
    x_pre, *_ = normalize_full(full, prescale=True)
    assert x_raw[:, :-1].std() < 0.1 * x_pre[:, :-1].std()   # the 1e-5 constant dominates a std of ~5e-7


def test_proxy_sequences_put_the_feature_in_the_price_slots_only():
    rng = np.random.default_rng(2)
    zw, zt, pos = rng.standard_normal((4, 32, 3)), rng.standard_normal((4, 3)), np.array([40, 41, 42, 43])
    x, stamp, terms = proxy_sequences(zw, zt, pos, prescale=True)
    assert x.shape == (12, 33, 6) and stamp.shape == (12, 32, 5)
    assert np.all(x[:, :, 4:] == 0)                                   # volume / amount slots are zero
    assert np.all(x[:, :, 0] == x[:, :, 3])                           # the four price slots carry the same series
    # sequence i * F + f holds feature f of row i
    expect = np.concatenate([zw[1, :, 2], [zt[1, 2]]])
    seq = x[1 * 3 + 2, :, 3]
    e = (expect - expect[:-1].mean()) / (expect[:-1].std() + EPS)      # prescale factor cancels in the normalized sequence
    assert np.allclose(seq, np.clip(e, -5, 5), atol=1e-4)
    assert np.array_equal(stamp[0], proxy_stamps(np.arange(8, 40)[None])[0])


def test_native_sequences_shapes_and_stamps():
    bars = np.abs(np.random.default_rng(3).standard_normal((100, 6)))
    hist = np.stack([np.arange(i - 32, i) for i in (40, 60)])
    ts = np.array(np.datetime64('2018-09-11T00:00') + np.arange(100) * np.timedelta64(1, 'h'))
    x, stamp, terms = native_sequences(bars, hist, np.array([40, 60]), ts, prescale=True)
    assert x.shape == (2, 33, 6) and stamp.shape == (2, 32, 5)
    assert stamp[0, 0, 1] == 8 and stamp[0, -1, 1] == 15               # hours of the context bars 8..39


def test_run_returns_hidden_states_loglik_and_draws(states):
    assert self_test(states)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 33, 6)).astype(np.float32)
    out = states.run(x, np.zeros((5, 32, 5), np.float32), draws=4)
    assert out['H'].shape == (5, states.layer_count, 32) and out['draws'].shape == (5, 4, 6)
    assert np.all(out['loglik'] <= 0)
    same = states.run(x, np.zeros((5, 32, 5), np.float32), capture=False)
    assert np.allclose(same['loglik'], out['loglik'])                   # likelihood does not depend on capture or draws


def test_patch_changes_the_loglik_and_zero_patch_does_not(states):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 33, 6)).astype(np.float32)
    st = np.zeros((3, 32, 5), np.float32)
    base = states.loglik(x, st)
    zero = states.loglik(x, st, patch=np.zeros((3, states.layer_count, 32)))
    big = states.loglik(x, st, patch=np.full((3, states.layer_count, 32), 3.0))
    assert np.allclose(base, zero) and not np.allclose(base, big)


def test_proxy_pool_folds_features_back_to_rows():
    res = {'loglik': np.arange(6, dtype=float), 'H': np.arange(6 * 2 * 3, dtype=float).reshape(6, 2, 3),
           'draws': np.arange(6 * 2 * 6, dtype=float).reshape(6, 2, 6)}
    pooled = proxy_pool(res, b=2, F=3, draws=2)
    assert np.allclose(pooled['loglik'], [0 + 1 + 2, 3 + 4 + 5])
    assert np.allclose(pooled['H'][0], res['H'][:3].mean(axis=0))
    assert pooled['draws'].shape == (2, 3, 2) and pooled['draws'][1, 2, 1] == res['draws'][5, 1, 3]
