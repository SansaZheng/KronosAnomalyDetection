#!/usr/bin/env python3
"""Kronos-mini states for EVERY S-FFSD row (not the capped row sets of run_kronos_rows.py).

For each row with at least `lookback` predecessors: teacher-forced NLL of the row (total and per feature)
and the layer-wise hidden state at the last context position, averaged over the per-feature sequences.
A complete, contiguous stream is what rolling context features of the knob need.
Output: results/2026-11-26/sffsd_best/states_L<lookback>.npz (not committed; regenerate with this script).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.fusion import robust_reference
from kronos_states import KronosStates, load_kronos, proxy_pool, proxy_sequences

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/2026-11-26/sffsd_best'


def main(lookback, limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_rows('s_ffsd')
    values, parts = data['values'], data['parts']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    F = int(keep.sum())
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    z_raw = ((values[:, keep] - center) / scale).astype(np.float64)
    targets = np.arange(lookback, len(values))[:limit]
    model, tokenizer, official = load_kronos()
    states = KronosStates(model, tokenizer, official, seed=42)
    n, chunk = len(targets), max(8, (1024 * 32 // lookback) // F)
    nll_feature, H = np.empty((n, F), np.float32), np.empty((n, states.layer_count, states.model.d_model), np.float16)
    t0 = time.time()
    for start in range(0, n, chunk):
        rows = targets[start:start + chunk]
        hist = rows[:, None] - lookback + np.arange(lookback)[None, :]
        x, stamp, _ = proxy_sequences(z_raw[hist], z_raw[rows], rows, True)
        res = states.run(x, stamp, capture=True)
        nll_feature[start:start + len(rows)] = -res['loglik'].reshape(len(rows), F)
        H[start:start + len(rows)] = proxy_pool(res, len(rows), F)['H']
        if (start // chunk) % 40 == 0:
            print(f'{start + len(rows)}/{n} rows, {time.time() - t0:.0f}s', flush=True)
    np.savez(OUT / f'states_L{lookback}.npz', targets=targets, nll_feature=nll_feature, H=H)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lookback', type=int, default=32)
    parser.add_argument('--limit', type=int)
    a = parser.parse_args()
    main(a.lookback, a.limit)
