"""Adapter for the official shiyu-coder/Kronos checkout, loaded only on demand."""
from pathlib import Path
import sys

import pandas as pd


class KronosBackend:
    def __init__(self, repo, model="NeoQuasar/Kronos-small",
                 tokenizer="NeoQuasar/Kronos-Tokenizer-base", device="cpu",
                 max_context=512, sample_count=5, temperature=1.0, top_p=0.9,
                 seed=42, model_revision=None, tokenizer_revision=None):
        if sample_count < 1 or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid sampling parameters")
        repo = Path(repo).resolve()
        if not (repo / "model" / "kronos.py").is_file():
            raise ValueError("repo must point to the official Kronos checkout")
        sys.path.insert(0, str(repo))
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        tok_kw = {"revision": tokenizer_revision} if tokenizer_revision else {}
        mod_kw = {"revision": model_revision} if model_revision else {}
        tok = KronosTokenizer.from_pretrained(tokenizer, **tok_kw)
        net = Kronos.from_pretrained(model, **mod_kw)
        tok.eval()
        net.eval()
        self.predictor = KronosPredictor(net, tok, device=device, max_context=max_context)
        self.max_context = max_context
        self.kwargs = dict(sample_count=sample_count, T=temperature, top_p=top_p,
                           top_k=0, verbose=False)
        self.metadata = dict(model=model, tokenizer=tokenizer, device=device,
                             max_context=max_context, seed=seed,
                             model_revision=model_revision, tokenizer_revision=tokenizer_revision,
                             **self.kwargs)

    def forecast(self, history, timestamp):
        import torch
        with torch.inference_mode():
            result = self.predictor.predict(
                df=history.drop(columns="timestamps").copy(),
                x_timestamp=history.timestamps.reset_index(drop=True),
                y_timestamp=pd.Series([timestamp]), pred_len=1, **self.kwargs)
        return result.iloc[0]
