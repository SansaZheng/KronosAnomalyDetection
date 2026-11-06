"""Sample-preserving one-step inference with the official Kronos model.

The official `KronosPredictor.predict_batch` averages its `sample_count` draws
before returning, which throws away the predictive distribution. This module runs
the same tokenize-decode-sample loop (pred_len = 1) and keeps every draw, so the
detector can use the spread of the samples as well as their mean. With the same
seed and arguments, the mean over draws equals the official output exactly; the
unit test checks that against the pinned upstream code.

Everything here is downstream glue. The model, tokenizer and sampling function
are imported from the official checkout; nothing is reimplemented.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .backend import prescale_window

PRICE = ["open", "high", "low", "close"]
COLUMNS = PRICE + ["volume", "amount"]
EPS = 1e-5


def load_official(repo, model_dir, tokenizer_dir, device="cpu", max_context=512):
    """Load the pinned official code and local weights; returns (predictor, module)."""
    repo = Path(repo).resolve()
    if not (repo / "model" / "kronos.py").is_file():
        raise ValueError("repo must point to the official Kronos checkout")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import kronos as official
    tokenizer = official.KronosTokenizer.from_pretrained(str(tokenizer_dir)).eval()
    net = official.Kronos.from_pretrained(str(model_dir)).eval()
    return official.KronosPredictor(net, tokenizer, device=device, max_context=max_context), official


def sample_one_step(predictor, official, x, x_stamp, y_stamp, sample_count, T=1.0, top_k=0, top_p=0.9):
    """All draws for a one-step forecast. x: (B, L, 6) normalized; returns (B, S, 6)."""
    import torch
    device = predictor.device
    x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    x_stamp = torch.from_numpy(np.asarray(x_stamp, dtype=np.float32)).to(device)
    y_stamp = torch.from_numpy(np.asarray(y_stamp, dtype=np.float32)).to(device)
    tokenizer, model, max_context = predictor.tokenizer, predictor.model, predictor.max_context
    with torch.no_grad():
        x = torch.clip(x, -predictor.clip, predictor.clip)
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2))
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2))
        x_token = tokenizer.encode(x, half=True)
        length = x.size(1)
        if length > max_context:
            raise ValueError("History longer than max_context")
        stamp = torch.cat([x_stamp, y_stamp], dim=1)[:, :length, :].contiguous()
        s1_logits, context = model.decode_s1(x_token[0], x_token[1], stamp)
        sample_pre = official.sample_from_logits(s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        s2_logits = model.decode_s2(context, sample_pre)
        sample_post = official.sample_from_logits(s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        full_pre = torch.cat([x_token[0], sample_pre.reshape(-1, 1)], dim=1)
        full_post = torch.cat([x_token[1], sample_post.reshape(-1, 1)], dim=1)
        start = max(0, length + 1 - max_context)
        z = tokenizer.decode([full_pre[:, start:].contiguous(), full_post[:, start:].contiguous()], half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))[:, :, -1, :]
    return z.cpu().numpy()


def stamps(timestamps):
    frame = pd.DataFrame(dict(minute=timestamps.dt.minute, hour=timestamps.dt.hour, weekday=timestamps.dt.day_of_week,
                              day=timestamps.dt.day, month=timestamps.dt.month))
    return frame.to_numpy(dtype=np.float32)


def forecast_samples(predictor, official, windows, window_stamps, target_stamps, sample_count,
                     prescale=True, T=1.0, top_k=0, top_p=0.9):
    """One-step draws in original units for a batch of history windows.

    windows: list of DataFrames with the six columns (oldest first); returns (B, S, 6).
    With `prescale`, each column is divided by its window std before the official
    (std + 1e-5) normalization, so the constant cannot flatten a low-priced series.
    """
    x, means, stds, factors = normalized_inputs(windows, prescale, predictor.clip)
    draws = sample_one_step(predictor, official, x, np.stack([stamps(s) for s in window_stamps]),
                            np.stack([stamps(s) for s in target_stamps]), sample_count, T, top_k, top_p)
    return (draws * (stds[:, None] + EPS) + means[:, None]) * factors[:, None]


def normalized_inputs(windows, prescale, clip=5):
    """What the model sees: (B, L, 6) normalized inputs, plus the de-normalization terms."""
    x_list, means, stds, factors = [], [], [], []
    for frame in windows:
        frame = frame[COLUMNS].astype(float)
        factor = np.ones(len(COLUMNS))
        if prescale:
            frame, scale = prescale_window(frame)
            factor = scale.to_numpy(dtype=float)
        x = frame.to_numpy(dtype=np.float32)
        mean, std = x.mean(axis=0), x.std(axis=0)
        x_list.append(np.clip((x - mean) / (std + EPS), -clip, clip))
        means.append(mean)
        stds.append(std)
        factors.append(factor)
    return np.stack(x_list), np.asarray(means), np.asarray(stds), np.asarray(factors)
