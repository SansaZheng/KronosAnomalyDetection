#!/usr/bin/env python3
"""Figure for docs/knob_values_explained.md: knob coordinate on normal vs anomalous rows, and what steering does."""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boundary_standin import label_boundary

ROOT = Path(__file__).resolve().parents[1]
FROZEN, OUT = ROOT / 'results/2026-11-10', ROOT / 'results/2026-11-16/knob_values'
GREY, BLUE, ORANGE = '#8a8f98', '#1f6f8b', '#d9822b'
SETS = (('s_ffsd', 'S-FFSD', 0), ('sapienza_15s', 'Sapienza 15 s', 1))      # third entry: coordinate to plot

fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
for row, (dataset, title, j) in enumerate(SETS):
    with open(FROZEN / f'knowledge_boundary_kronos_{dataset}/{label_boundary(dataset)}_state.pkl', 'rb') as fh:
        _, _, predictor, _ = pickle.load(fh)
    cache = np.load(FROZEN / f'kronos_rows/{dataset}/test.npz')
    y = cache['labels'].astype(int) if 'labels' in cache.files else None
    z = predictor.pooled(cache['H'].astype(np.float32))[:, j]
    ax = axes[row, 0]
    bins = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), 45)
    ax.hist(z[y == 0], bins, density=True, color=GREY, alpha=0.75, label='normal rows')
    ax.hist(z[y == 1], bins, density=True, color=ORANGE, alpha=0.75, label='anomalous rows')
    ax.set_title(f'{title}: knob value, hidden dimension {int(predictor.dims_[j])}', fontsize=10)
    ax.set_xlabel('knob coordinate z (standardised, layer-weighted)')
    ax.set_ylabel('density')
    ax.legend(fontsize=8, frameon=False)
    s = pd.read_csv(OUT / dataset / 'steering.csv')
    n = pd.concat([s[s.direction == 'none'], s[s.direction == 'normal']]).sort_values('tau')
    r = s[s.direction == 'random']
    ax = axes[row, 1]
    ax.plot(n.tau, n.mean_dB_anomaly, 'o-', color=ORANGE, label='anomalous rows')
    ax.plot(n.tau, n.mean_dB_normal, 'o-', color=GREY, label='normal rows')
    ax.plot(r.tau, r.mean_dB_anomaly, 'x', color=ORANGE, ms=9, label='random push, anomalous')
    ax.plot(r.tau, r.mean_dB_normal, 'x', color=GREY, ms=9, label='random push, normal')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{title}: change of forecast surprise B', fontsize=10)
    ax.set_xlabel('push τ along the knob (τ > 0 = toward "anomaly")')
    ax.set_ylabel('mean change of B')
    ax.legend(fontsize=7, frameon=False)
    ax = axes[row, 2]
    ax.plot(n.tau, n.auroc, 'o-', color=BLUE, label='AUROC of B after push')
    ax.plot(n.tau, n.ap, 's-', color=ORANGE, label='AP of B after push')
    ax.plot(r.tau, r.auroc, 'x', color=BLUE, ms=9, label='random push')
    ax.plot(r.tau, r.ap, 'x', color=ORANGE, ms=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{title}: detection quality after the push', fontsize=10)
    ax.set_xlabel('push τ along the knob')
    ax.legend(fontsize=7, frameon=False, loc='lower right' if row else 'upper center')
for ax in axes.ravel():
    ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'knob_values.png', dpi=150)
print(OUT / 'knob_values.png')
