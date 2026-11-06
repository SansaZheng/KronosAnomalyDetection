"""Sample-preserving inference must reproduce the official predictor exactly."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
VENDOR = Path(__file__).resolve().parents[2] / "vendor_kronos"
if not (VENDOR / "model" / "kronos.py").is_file():
    VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "Kronos"
pytestmark = pytest.mark.skipif(not (VENDOR / "model" / "kronos.py").is_file(), reason="official Kronos checkout not present")

from kronos_ad.sampling import COLUMNS, forecast_samples, normalized_inputs, stamps


def tiny_predictor():
    sys.path.insert(0, str(VENDOR))
    from model import kronos as official
    torch.manual_seed(0)
    tokenizer = official.KronosTokenizer(d_in=6, d_model=16, n_heads=2, ff_dim=32, n_enc_layers=2, n_dec_layers=2,
                                         ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=4, s2_bits=4,
                                         beta=1.0, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=4).eval()
    net = official.Kronos(s1_bits=4, s2_bits=4, n_layers=2, d_model=16, n_heads=2, ff_dim=32, ffn_dropout_p=0.0,
                          attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True).eval()
    return official.KronosPredictor(net, tokenizer, device="cpu", max_context=64), official


def bars(n, seed):
    rng = np.random.default_rng(seed)
    close = 5e-5 * np.exp(np.cumsum(rng.normal(scale=0.004, size=n)))
    frame = pd.DataFrame(dict(open=close * (1 + rng.normal(scale=1e-3, size=n)), close=close))
    frame["high"] = frame[["open", "close"]].max(axis=1) * (1 + abs(rng.normal(scale=1e-3, size=n)))
    frame["low"] = frame[["open", "close"]].min(axis=1) * (1 - abs(rng.normal(scale=1e-3, size=n)))
    frame["volume"] = rng.exponential(2000, size=n).round()
    frame["amount"] = frame.volume * frame.close
    frame["timestamps"] = pd.date_range("2018-12-22", periods=n, freq="5min")
    return frame


def test_mean_of_kept_samples_equals_official_prediction():
    predictor, official = tiny_predictor()
    windows = [bars(32, seed) for seed in range(3)]
    targets = [pd.Series([w.timestamps.iloc[-1] + pd.Timedelta(minutes=5)]) for w in windows]
    torch.manual_seed(42)
    reference = predictor.predict_batch(df_list=[w[COLUMNS] for w in windows], x_timestamp_list=[w.timestamps for w in windows],
                                        y_timestamp_list=targets, pred_len=1, T=1.0, top_k=0, top_p=0.9, sample_count=5, verbose=False)
    torch.manual_seed(42)
    draws = forecast_samples(predictor, official, windows, [w.timestamps for w in windows], targets, sample_count=5, prescale=False)
    assert draws.shape == (3, 5, 6)
    np.testing.assert_allclose(draws.mean(axis=1), np.stack([r[COLUMNS].to_numpy()[0] for r in reference]), rtol=1e-5, atol=1e-12)


def test_prescale_changes_what_the_model_sees_but_not_the_interface():
    predictor, official = tiny_predictor()
    windows = [bars(32, 7)]
    targets = [pd.Series([windows[0].timestamps.iloc[-1] + pd.Timedelta(minutes=5)])]
    torch.manual_seed(1)
    plain = forecast_samples(predictor, official, windows, [windows[0].timestamps], targets, 4, prescale=False)
    torch.manual_seed(1)
    scaled = forecast_samples(predictor, official, windows, [windows[0].timestamps], targets, 4, prescale=True)
    assert plain.shape == scaled.shape == (1, 4, 6) and np.isfinite(scaled).all()
    # Without prescale the 1e-5 constant flattens a 5e-5-priced series: the model sees
    # prices inside a band narrower than 0.3 normalized units while volume spans > 2.
    raw_view = normalized_inputs(windows, prescale=False)[0][0]
    scaled_view = normalized_inputs(windows, prescale=True)[0][0]
    assert np.ptp(raw_view[:, :4], axis=0).max() < 0.3 and np.ptp(raw_view[:, 4]) > 2
    assert np.ptp(scaled_view[:, :4], axis=0).min() > 2


def test_samples_are_causal_in_the_window():
    predictor, official = tiny_predictor()
    a = bars(32, 3)
    b = a.copy()
    b.loc[31, "close"] *= 1.5  # last history bar changes
    t = [pd.Series([a.timestamps.iloc[-1] + pd.Timedelta(minutes=5)])]
    torch.manual_seed(5)
    x = forecast_samples(predictor, official, [a], [a.timestamps], t, 3)
    torch.manual_seed(5)
    y = forecast_samples(predictor, official, [b], [b.timestamps], t, 3)
    assert not np.allclose(x, y)
    assert stamps(a.timestamps).shape == (32, 5)
