#!/usr/bin/env python3
"""S-FFSD: let Kronos re-create each "day" from what came before it, and score rows against the re-creation.

S-FFSD has no clock time (its Time column is a transaction counter), so a "day" is a block of D consecutive
transactions.  For every block the context is frozen at the block start (the 32 rows before it), which
keeps the block's own fraud out of what Kronos conditions on.  Two modes:

  frozen    reconstruction: the NLL of every row of the block as the next row after the frozen context
            (one forward pass per row, all rows of the dataset; feeds the detector search)
  generate  generation: Kronos samples S continuations of length D per feature from the frozen context;
            each real row is scored by how far it lies in the tails of the generated values of its block
            (sum over features of -log of a two-sided empirical tail probability), and each block by the
            mean of its rows

Outputs: results/2026-11-26/sffsd_best/{frozen_D<D>.npz, generated_D<D>.npz, generation.csv}.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.fusion import robust_reference
from kronos_states import CLIP, EPS, PRICE_SLOTS, KronosStates, load_kronos, normalize_full, proxy_sequences, proxy_stamps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/2026-11-26/sffsd_best'
LOOK = 32


def setup():
    data = load_rows('s_ffsd')
    values, parts, labels = data['values'], data['parts'], data['labels'].astype(int)
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    z_raw = ((values[:, keep] - center) / scale).astype(np.float64)
    model, tokenizer, official = load_kronos()
    return z_raw, parts, labels, KronosStates(model, tokenizer, official, seed=42)


def metrics(y, score):
    out = dict(ap=float(average_precision_score(y, score)), auroc=float(roc_auc_score(y, score)))
    for fpr in (0.01, 0.05, 0.10):
        out[f'recall_{int(100 * fpr)}pct'] = float((score[y == 1] > np.quantile(score[y == 0], 1 - fpr)).mean())
    return out


def frozen(D):
    z_raw, parts, labels, states = setup()
    F = z_raw.shape[1]
    targets = np.arange(max(LOOK, D), len(z_raw))
    starts = np.maximum((targets // D) * D, LOOK)
    nll, chunk, t0 = np.empty((len(targets), F), np.float32), max(8, 1024 // F), time.time()
    for a in range(0, len(targets), chunk):
        rows, s = targets[a:a + chunk], starts[a:a + chunk]
        hist = s[:, None] - LOOK + np.arange(LOOK)[None, :]
        x, stamp, _ = proxy_sequences(z_raw[hist], z_raw[rows], s, True)
        nll[a:a + len(rows)] = -states.loglik(x, stamp).reshape(len(rows), F)
        if (a // chunk) % 60 == 0:
            print(f'frozen D={D}: {a + len(rows)}/{len(targets)} rows, {time.time() - t0:.0f}s', flush=True)
    np.savez(OUT / f'frozen_D{D}.npz', targets=targets, nll_feature=nll)
    test = (parts[targets] == 'test') & np.isin(labels[targets], [0, 1])
    print(f'frozen D={D} standalone on labelled test rows:', {k: round(v, 3) for k, v in metrics(labels[targets][test], nll[test].sum(axis=1)).items()})


def generate(D, S, part):
    z_raw, parts, labels, states = setup()
    F, model, tok = z_raw.shape[1], states.model, states.tokenizer
    part_rows = np.flatnonzero(parts == part)
    starts = np.arange((part_rows[0] // D) * D, part_rows[-1] + 1, D)
    starts = starts[(starts >= LOOK) & (starts + D <= len(z_raw))]
    score, block_of, t0 = np.full((len(z_raw), F), np.nan, np.float32), np.full(len(z_raw), -1), time.time()
    batch = max(1, 4096 // (F * S))                                   # blocks per batch
    for a in range(0, len(starts), batch):
        s = starts[a:a + batch]
        b = len(s)
        hist = s[:, None] - LOOK + np.arange(LOOK)[None, :]
        future = s[:, None] + np.arange(D)[None, :]
        series = np.transpose(z_raw[hist], (0, 2, 1)).reshape(b * F, LOOK)                  # one series per block and feature
        full = np.zeros((b * F, LOOK + 1, 6))
        full[:, :LOOK, PRICE_SLOTS] = series[:, :, None]
        x, mean, std, factor = normalize_full(full, True)                                   # context statistics only
        stamps = np.repeat(proxy_stamps(s[:, None] - LOOK + np.arange(LOOK + D)[None, :]), F, axis=0)
        with torch.no_grad():
            xt = torch.as_tensor(x[:, :LOOK], dtype=torch.float32).repeat_interleave(S, dim=0)
            st = torch.as_tensor(stamps, dtype=torch.float32).repeat_interleave(S, dim=0)
            s1, s2 = tok.encode(xt, half=True)
            for i in range(D):
                logits1, ctx = model.decode_s1(s1, s2, st[:, :LOOK + i])
                pre = states._sample(logits1[:, -1], 1)
                post = states._sample(model.decode_s2(ctx, pre)[:, -1], 1)
                s1, s2 = torch.cat([s1, pre], dim=1), torch.cat([s2, post], dim=1)
            gen = tok.decode([s1, s2], half=True)[:, LOOK:, 3].reshape(b, F, S * D).cpu().numpy()   # normalized close slot
        actual = np.transpose(z_raw[future], (0, 2, 1)) / factor[:, 0].reshape(b, F, 1)
        actual = np.clip((actual - mean[:, 0].reshape(b, F, 1)) / (std[:, 0].reshape(b, F, 1) + EPS), -CLIP, CLIP)
        centre = np.median(gen, axis=2, keepdims=True)
        dev_gen = np.sort(np.abs(gen - centre), axis=2)
        dev_act = np.abs(actual - centre)
        for i in range(b):
            for f in range(F):
                beyond = S * D - np.searchsorted(dev_gen[i, f], dev_act[i, f], side='left')
                score[future[i], f] = -np.log((1 + beyond) / (1 + S * D))
            block_of[future[i]] = a + i
        print(f'generate D={D} S={S}: {a + b}/{len(starts)} blocks, {time.time() - t0:.0f}s', flush=True)
    done = np.flatnonzero(block_of >= 0)
    row = score[done].sum(axis=1)
    block = pd.Series(row).groupby(block_of[done]).transform('mean').to_numpy()
    np.savez(OUT / f'generated_D{D}_{part}.npz', targets=done, score_feature=score[done], block_mean=block)
    lab = np.isin(labels[done], [0, 1])
    y = labels[done][lab]
    rows = [dict(block=D, samples=S, part=part, score='row: distance from the generated block', **metrics(y, row[lab])),
            dict(block=D, samples=S, part=part, score='block: mean distance of the block', **metrics(y, block[lab])),
            dict(block=D, samples=S, part=part, score='row + block', **metrics(y, (row + block)[lab])),
            dict(block=D, samples=S, part=part, score='row: negative (closer than generated = suspicious)', **metrics(y, -row[lab]))]
    path = OUT / 'generation.csv'
    frame = pd.concat(([pd.read_csv(path)] if path.exists() else []) + [pd.DataFrame(rows)])
    frame.to_csv(path, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['frozen', 'generate'], required=True)
    parser.add_argument('--block', type=int, default=32)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--part', default='test')
    a = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen(a.block) if a.mode == 'frozen' else generate(a.block, a.samples, a.part)
