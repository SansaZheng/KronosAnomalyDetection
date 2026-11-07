"""Hidden states, patches and teacher-forced likelihoods for the official Kronos model.

Works against the shiyu-coder/Kronos classes (`Kronos`, `KronosTokenizer`) without
modifying them: forward hooks capture the residual stream after the embedding
(layer 0) and after every transformer block (layers 1..n_layers) at the last context
position, and the same hooks add a patch there during an intervention.

The tokenizer's encoder and the predictor's self-attention are causal, so the
tokens of the context bars do not change when the realised next bar is appended;
the teacher-forced target is therefore the token pair the tokenizer assigns to the
realised bar, and its log-likelihood is
    log p(s1* | context) + log p(s2* | s1*, context),
the one-bar case of the paper's Eq. 11.
"""
import numpy as np


def _torch():
    import torch
    return torch


def normalize_windows(windows, clip=5.0, prescale=False, eps=1e-5):
    """Per-window normalization as in KronosPredictor.predict: (x - mean) / (std + 1e-5), clipped.

    With `prescale`, columns are first divided by their window std so the 1e-5
    constant cannot flatten series whose scale is far below it (see backend.py).
    """
    x = np.asarray(windows, dtype=float)
    if prescale:
        scale = x.std(axis=1, keepdims=True)
        x = x / np.where(scale > 0, scale, 1.0)
    mean, std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
    return np.clip((x - mean) / (std + eps), -clip, clip)


class KronosHiddenStates:
    """Hidden-state access for a Kronos predictor model and its tokenizer."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.model.eval()
        self.tokenizer.eval()

    @property
    def layer_count(self):
        return len(self.model.transformer) + 1

    def tokens(self, x):
        """(s1, s2) token ids, shape (b, T) each, for normalized windows x of shape (b, T, 6)."""
        torch = _torch()
        with torch.no_grad():
            s1, s2 = self.tokenizer.encode(torch.as_tensor(np.asarray(x), dtype=torch.float32), half=True)
        return s1, s2

    def _hooks(self, patch=None, capture=None):
        """Register hooks; `patch` is (b, L+1, d) added at the last position, `capture` a list."""
        torch = _torch()
        handles = []
        blocks = self.model.transformer

        def pre_hook(module, args, kwargs):
            x = args[0]
            if capture is not None:
                capture.append(('0', x[:, -1].detach().clone()))
            if patch is not None:
                x = x.clone()
                x[:, -1] = x[:, -1] + torch.as_tensor(patch[:, 0], dtype=x.dtype)
            return (x,) + tuple(args[1:]), kwargs

        handles.append(blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
        for index, block in enumerate(blocks):
            def hook(module, args, output, layer=index + 1):
                out = output
                if capture is not None:
                    capture.append((str(layer), out[:, -1].detach().clone()))
                if patch is not None:
                    out = out.clone()
                    out[:, -1] = out[:, -1] + torch.as_tensor(patch[:, layer], dtype=out.dtype)
                return out
            handles.append(block.register_forward_hook(hook))
        return handles

    def hidden_states(self, s1, s2, stamp=None):
        """H with shape (b, L+1, d_model): embedding output then each block output, last position."""
        torch = _torch()
        captured = []
        handles = self._hooks(capture=captured)
        try:
            with torch.no_grad():
                self.model.decode_s1(s1, s2, stamp)
        finally:
            for h in handles:
                h.remove()
        return torch.stack([tensor for _, tensor in captured], dim=1).numpy()

    def loglik(self, s1, s2, stamp, s1_next, s2_next, patch=None):
        """Teacher-forced log p(s1*, s2* | context), optionally with a hidden-state patch."""
        torch = _torch()
        handles = self._hooks(patch=patch)
        try:
            with torch.no_grad():
                s1_logits, context = self.model.decode_s1(s1, s2, stamp)
                lp1 = torch.log_softmax(s1_logits[:, -1], dim=-1).gather(1, s1_next.view(-1, 1))[:, 0]
                s2_logits = self.model.decode_s2(context, s1_next.view(-1, 1))
                lp2 = torch.log_softmax(s2_logits[:, -1], dim=-1).gather(1, s2_next.view(-1, 1))[:, 0]
        finally:
            for h in handles:
                h.remove()
        return (lp1 + lp2).numpy()

    def next_tokens(self, x_context, x_next):
        """Token pair of the realised next bar, tokenized together with its context."""
        torch = _torch()
        full = np.concatenate([np.asarray(x_context), np.asarray(x_next)[:, None, :]], axis=1)
        s1, s2 = self.tokens(full)
        return s1[:, -1], s2[:, -1]


def intervention_effect(states, s1, s2, stamp, s1_next, s2_next, patch):
    """Change in teacher-forced log-likelihood caused by a hidden-state patch (Eq. 11, T=1)."""
    base = states.loglik(s1, s2, stamp, s1_next, s2_next)
    patched = states.loglik(s1, s2, stamp, s1_next, s2_next, patch=patch)
    return patched - base
