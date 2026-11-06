"""Validate official predictor call contract without downloading model weights."""
from contextlib import nullcontext
import sys
import types

import pandas as pd

from kronos_ad.backend import KronosBackend


def test_official_adapter_contract(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "kronos.py").touch()
    loaded = []
    instances = []

    class Net:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            loaded.append((name, kwargs))
            return cls()

        def eval(self):
            self.is_eval = True

    class Predictor:
        def __init__(self, net, tok, **kwargs):
            assert net.is_eval and tok.is_eval
            instances.append(self)

        def predict(self, **kwargs):
            self.call = kwargs
            return pd.DataFrame({"close": [42.0], "volume": [0.0], "amount": [0.0]})

    torch = types.SimpleNamespace(manual_seed=lambda _: None,
                                  cuda=types.SimpleNamespace(is_available=lambda: False),
                                  inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "model", types.SimpleNamespace(
        Kronos=Net, KronosTokenizer=Net, KronosPredictor=Predictor))
    monkeypatch.syspath_prepend(str(tmp_path))
    backend = KronosBackend(tmp_path, sample_count=7, model_revision="pinned-model")
    history = pd.DataFrame(dict(timestamps=pd.date_range("2020", periods=2, freq="h"),
                                open=[1, 2], high=[2, 3], low=[0, 1], close=[1, 2]))
    target = pd.Timestamp("2020-01-01 02:00")
    result = backend.forecast(history, target)
    call = instances[0].call
    assert call["pred_len"] == 1 and call["sample_count"] == 7
    assert "timestamps" not in call["df"]
    assert list(call["y_timestamp"]) == [target]
    # The stub answers 42 in the scaled units the model saw; window std of close is 0.5.
    assert result.close == 42 * 0.5
    assert call["df"].close.std(ddof=0) == 1
    assert list(result.index) == ["open", "high", "low", "close"]
    assert loaded[1][1] == {"revision": "pinned-model"}
    backend.prescale = False
    assert backend.forecast(history, target).close == 42
    assert instances[0].call["df"].close.tolist() == [1, 2]
