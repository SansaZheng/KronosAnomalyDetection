"""The real Kronos-mini as the model under study, loaded offline.

One pass per batch gives everything the knowledge-boundary and fusion experiments need:

  * hidden states H (b, n_layers + 1, d_model) at the last context bar -- the embedding
    output and every transformer block output, captured by forward hooks;
  * the teacher-forced log-likelihood log p(s1*, s2* | context) of the realised next bar
    (one-bar case of Eq. 11 of Huang et al.), with an optional hidden-state patch;
  * S sampled one-step forecasts (the official T = 1, top_p = 0.9 sampler), de-normalized.

The checkpoints come from `vendor/Kronos` (official code, commit 67b630e) and `weights/`
(Kronos-mini, Kronos-Tokenizer-2k), both git-ignored; SHA256s are those pinned in
`results/2026-11-06/input_manifest.json`.  Nothing is downloaded: `HF_HUB_OFFLINE=1` and
`local_files_only=True`.

Two representations of a row:

  native   OHLCVA bars as they are (the K-line dataset);
  proxy    the published Kronos-Proxy mapping of a tabular row: each feature becomes its
           own bar series with the standardized value in the four price slots and zero
           volume / amount (results/2026-11-06/kronos_proxy/README.md).  Per row, hidden
           states are averaged over the feature passes, log-likelihoods are summed, and
           each feature's forecast is read from the close slot of its own pass.

Normalization follows the official predictor -- per window, (x - mean) / (std + 1e-5),
clipped to [-5, 5] -- with `prescale` (kronos_ad.backend) on by default.  The realised
next bar is normalized with its context's statistics, so the teacher-forced target is
the token pair of the bar the forecast is scored against; because the tokenizer's encoder
is causal (checked in `self_test`), the context tokens are identical whether or not the
next bar is appended.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
KRONOS_SRC = ROOT / 'vendor/Kronos'
# Model size: KRONOS_MODEL in {mini, small, base}; mini pairs with Tokenizer-2k, small and base with Tokenizer-base.
KRONOS_MODEL = os.environ.get('KRONOS_MODEL', 'mini')
MODEL_DIR = ROOT / f'weights/Kronos-{KRONOS_MODEL}'
TOKENIZER_DIR = ROOT / ('weights/Kronos-Tokenizer-2k' if KRONOS_MODEL == 'mini' else 'weights/Kronos-Tokenizer-base')
# Where the real-Kronos result directories live (the mini rerun: results/2026-11-10; the size study: results/2026-11-14/<model>).
RESULTS = Path(os.environ['KRONOS_RESULTS']).resolve() if os.environ.get('KRONOS_RESULTS') else ROOT / 'results/2026-11-10'
EPS = 1e-5


def subset_positions(labels, cap, seed):
    """Positions of a row subset: `cap` rows drawn without replacement plus every labelled anomaly (size study)."""
    n = len(labels)
    pick = np.arange(n) if n <= cap else np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
    return np.union1d(pick, np.flatnonzero(np.asarray(labels) == 1))

CLIP = 5.0
COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']
PRICE_SLOTS = [0, 1, 2, 3]


def load_kronos(device='cpu', threads=2):
    """Official code + local weights, offline.  Returns (model, tokenizer, official module)."""
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    if not (KRONOS_SRC / 'model' / 'kronos.py').is_file():
        raise FileNotFoundError(f'{KRONOS_SRC} must hold the official Kronos checkout (model/kronos.py)')
    if str(KRONOS_SRC) not in sys.path:
        sys.path.insert(0, str(KRONOS_SRC))
    torch.set_num_threads(threads)
    from model import kronos as official
    tokenizer = official.KronosTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True).to(device).eval()
    model = official.Kronos.from_pretrained(str(MODEL_DIR), local_files_only=True).to(device).eval()
    return model, tokenizer, official


def normalize_full(full, prescale=True):
    """full: (b, L + 1, 6) raw bars, the last one the realised next bar.

    Returns the normalized sequence (context statistics), and the de-normalization terms
    mean (b, 6), std (b, 6), factor (b, 6) such that raw = (x * (std + EPS) + mean) * factor.
    """
    full = np.asarray(full, dtype=np.float64)
    context = full[:, :-1]
    factor = np.ones((full.shape[0], full.shape[2]))
    if prescale:
        s = context.std(axis=1)
        factor = np.where(s > 0, s, 1.0)
        full = full / factor[:, None, :]
        context = full[:, :-1]
    mean, std = context.mean(axis=1), context.std(axis=1)
    x = np.clip((full - mean[:, None, :]) / (std[:, None, :] + EPS), -CLIP, CLIP)
    return x.astype(np.float32), mean, std, factor


def stamps_from_timestamps(timestamps):
    """(n, 5) float32 time features the model's embedding expects: minute, hour, weekday, day, month."""
    t = pd.DatetimeIndex(pd.to_datetime(timestamps))
    return np.column_stack([t.minute, t.hour, t.day_of_week, t.day, t.month]).astype(np.float32)


def proxy_stamps(positions):
    """The published proxy calendar: artificial minute index from 2000-01-01 by in-group position."""
    t = pd.Timestamp('2000-01-01') + pd.to_timedelta(np.asarray(positions).ravel(), unit='min')
    return stamps_from_timestamps(t).reshape(*np.shape(positions), 5)


class KronosStates:
    """Hidden states, teacher-forced likelihood and sampled forecasts of the official model.

    `hidden_states(x_full, stamp)`, `loglik(x_full, stamp, patch=None)` and
    `run(x_full, stamp, ...)` take normalized sequences of shape (b, L + 1, 6) whose last
    element is the realised next bar, and stamps (b, L, 5) for the context bars.
    """

    def __init__(self, model, tokenizer, official, seed=42, temperature=1.0, top_p=0.9):
        self.model, self.tokenizer, self.official = model.eval(), tokenizer.eval(), official
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.temperature, self.top_p = temperature, top_p

    @property
    def layer_count(self):
        return len(self.model.transformer) + 1

    def _hooks(self, patch=None, capture=None):
        handles, blocks = [], self.model.transformer

        def pre_hook(module, args, kwargs):
            x = args[0]
            if capture is not None:
                capture.append(x[:, -1].detach().clone())
            if patch is not None:
                x = x.clone()
                x[:, -1] = x[:, -1] + torch.as_tensor(patch[:, 0], dtype=x.dtype, device=x.device)
            return (x,) + tuple(args[1:]), kwargs

        handles.append(blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
        for index, block in enumerate(blocks):
            def hook(module, args, output, layer=index + 1):
                out = output
                if capture is not None:
                    capture.append(out[:, -1].detach().clone())
                if patch is not None:
                    out = out.clone()
                    out[:, -1] = out[:, -1] + torch.as_tensor(patch[:, layer], dtype=out.dtype, device=out.device)
                return out
            handles.append(block.register_forward_hook(hook))
        return handles

    def encode(self, x_full):
        """Context tokens (b, L) x 2 and the realised next bar's tokens (b,) x 2."""
        with torch.no_grad():
            s1, s2 = self.tokenizer.encode(torch.as_tensor(np.asarray(x_full), dtype=torch.float32, device=self.device), half=True)
        return s1[:, :-1].contiguous(), s2[:, :-1].contiguous(), s1[:, -1].contiguous(), s2[:, -1].contiguous()

    def _sample(self, logits, count):
        logits = logits / self.temperature
        if self.top_p < 1.0:
            logits = self.official.top_k_top_p_filtering(logits, top_k=0, top_p=self.top_p)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=count, replacement=True, generator=self.generator)

    def run(self, x_full, stamp, patch=None, capture=True, draws=0):
        """One pass.  Returns dict(loglik (b,), H (b, layers, d) if capture, draws (b, S, 6) normalized if draws > 0)."""
        s1, s2, s1n, s2n = self.encode(x_full)
        st = torch.as_tensor(np.asarray(stamp), dtype=torch.float32, device=self.device)
        captured = [] if capture else None
        handles = self._hooks(patch=patch, capture=captured)
        try:
            with torch.no_grad():
                s1_logits, context = self.model.decode_s1(s1, s2, st)
        finally:
            for h in handles:
                h.remove()
        out = {}
        with torch.no_grad():
            last = s1_logits[:, -1]
            lp1 = torch.log_softmax(last, dim=-1).gather(1, s1n.view(-1, 1))[:, 0]
            s2_logits = self.model.decode_s2(context, s1n.view(-1, 1))[:, -1]
            lp2 = torch.log_softmax(s2_logits, dim=-1).gather(1, s2n.view(-1, 1))[:, 0]
            out['loglik'] = (lp1 + lp2).cpu().numpy()
            if capture:
                out['H'] = torch.stack(captured, dim=1).cpu().numpy()
            if draws:
                b, S = s1.shape[0], draws
                samp1 = self._sample(last, S).reshape(b * S, 1)                         # S draws of s1 per row
                ctx = context.repeat_interleave(S, dim=0)
                samp2 = self._sample(self.model.decode_s2(ctx, samp1)[:, -1], 1)       # one s2 draw per s1 draw
                full_pre = torch.cat([s1.repeat_interleave(S, dim=0), samp1], dim=1)
                full_post = torch.cat([s2.repeat_interleave(S, dim=0), samp2], dim=1)
                z = self.tokenizer.decode([full_pre, full_post], half=True)[:, -1, :]
                out['draws'] = z.reshape(b, S, -1).cpu().numpy()
        return out

    def hidden_states(self, x_full, stamp):
        return self.run(x_full, stamp, capture=True)['H']

    def loglik(self, x_full, stamp, patch=None):
        return self.run(x_full, stamp, patch=patch, capture=False)['loglik']


# ------------------------------------------------------------------ row representations


def native_sequences(bars, histories, targets, timestamps, prescale=True):
    """K-lines: (b, L + 1, 6) normalized sequences, context stamps, de-normalization terms."""
    full = np.concatenate([bars[histories], bars[targets][:, None, :]], axis=1)
    x, mean, std, factor = normalize_full(full, prescale)
    stamp = stamps_from_timestamps(timestamps[histories].ravel()).reshape(len(targets), histories.shape[1], 5)
    return x, stamp, (mean, std, factor)


def denormalize(z, terms):
    """z: (b, ..., 6) normalized values -> raw units."""
    mean, std, factor = terms
    shape = (len(mean),) + (1,) * (z.ndim - 2) + (mean.shape[1],)
    return (z * (std.reshape(shape) + EPS) + mean.reshape(shape)) * factor.reshape(shape)


def proxy_sequences(z_windows, z_targets, positions, prescale=True):
    """Tabular rows: one bar series per feature.

    z_windows (b, L, F) standardized context, z_targets (b, F) the realised row, positions (b,)
    in-group index of the target row.  Returns x (b * F, L + 1, 6), stamps (b * F, L, 5) and
    the de-normalization terms; feature f of row i is sequence i * F + f.
    """
    b, L, F = z_windows.shape
    series = np.concatenate([z_windows, z_targets[:, None, :]], axis=1)          # (b, L+1, F)
    series = np.transpose(series, (0, 2, 1)).reshape(b * F, L + 1)               # (b*F, L+1)
    full = np.zeros((b * F, L + 1, 6))
    full[:, :, PRICE_SLOTS] = series[:, :, None]
    x, mean, std, factor = normalize_full(full, prescale)
    pos = np.asarray(positions)[:, None] - L + np.arange(L)[None, :]             # (b, L) context positions
    stamp = np.repeat(proxy_stamps(pos), F, axis=0)
    return x, stamp, (mean, std, factor)


def proxy_pool(result, b, F, draws=0):
    """Fold per-feature outputs of `KronosStates.run` back to rows: mean H, summed loglik, close-slot draws (b, F, S)."""
    out = {'loglik': result['loglik'].reshape(b, F).sum(axis=1)}
    if 'H' in result:
        out['H'] = result['H'].reshape(b, F, *result['H'].shape[1:]).mean(axis=1)
    if draws:
        out['draws'] = result['draws'].reshape(b, F, draws, 6)[..., 3]
    return out


def self_test(states, rng=np.random.default_rng(0)):
    """Causality of the tokenizer encoder and shape checks; raises on failure."""
    full = rng.normal(size=(4, 33, 6)).astype(np.float32)
    s1a, s2a, _, _ = states.encode(full)
    with torch.no_grad():
        s1b, s2b = states.tokenizer.encode(torch.as_tensor(full[:, :-1]), half=True)
    if not (torch.equal(s1a, s1b) and torch.equal(s2a, s2b)):
        raise AssertionError('tokenizer encoder is not causal: context tokens change when the next bar is appended')
    out = states.run(full, np.zeros((4, 32, 5), np.float32), draws=3)
    assert out['H'].shape == (4, states.layer_count, states.model.d_model) and out['draws'].shape == (4, 3, 6)
    assert np.isfinite(out['loglik']).all() and (out['loglik'] <= 0).all()
    return True
