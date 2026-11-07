#!/usr/bin/env python3
"""Figures for a knowledge-boundary results directory (stand-in or Kronos)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6b66', '#e6e6e2'
DIVERGING = LinearSegmentedColormap.from_list('bwo', [BLUE, '#f4f4f1', ORANGE])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5, 'legend.frameon': False})


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=180)
    fig.savefig(out / f'{name}.pdf')
    plt.close(fig)


def maps(out, results):
    names = [r['boundary'] for r in results]
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.6 + 1.9 * len(names)), squeeze=False, layout='constrained')
    for row, name in enumerate(names):
        for col, label in enumerate(('cnn', 'raw')):
            delta = np.load(out / f'{name}_delta_{label}.npy')
            ax = axes[row, col]
            limit = np.abs(delta).max()
            image = ax.imshow(delta, aspect='auto', cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit), interpolation='nearest')
            metrics = next(r for r in results if r['boundary'] == name)[f'structure_{label}']
            ax.set_title(f'{name} boundary, {"CNN vertical-branch activations" if label == "cnn" else "standardized hidden states"}\n'
                         f'V_corr {metrics["v_corr"]:.2f}   H_corr {metrics["h_corr"]:.2f}   SNR {metrics["snr_db"]:.1f} dB   LateRatio {metrics["late_ratio"]:.2f}', fontsize=8.5)
            ax.set_ylabel('layer (0 = embedding)')
            ax.set_xlabel('hidden dimension')
            fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    fig.suptitle('Class-conditional activation difference, mean(y=1) - mean(y=0)', fontsize=10)
    save(fig, out, 'boundary_difference_maps')


def energy(out, results):
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 3.4), squeeze=False, layout='constrained')
    for ax, r in zip(axes[0], results):
        delta = np.load(out / f'{r["boundary"]}_delta_cnn.npy')
        profile = np.sort(np.abs(delta).mean(axis=0))[::-1]
        profile = profile / profile[0]
        ax.plot(profile, color=BLUE, lw=1.8)
        k = r['knee_cnn']
        ax.axvline(k, color=YELLOW, lw=1.2)
        ax.text(k + 0.5, 0.9, f'knee at {k}', color='#8a5d00', fontsize=8)
        ax.set_title(f'{r["boundary"]} boundary: ranked dimension energy')
        ax.set_xlabel('dimensions, ranked')
        ax.set_ylabel('normalized energy')
        ax.grid(color=GRID, lw=0.6)
    save(fig, out, 'boundary_energy')


def knobs(out, results):
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4), squeeze=False, layout='constrained')
    for ax, r in zip(axes[0], results):
        data = np.load(out / f'{r["boundary"]}_knob_space.npz')
        z, y, score = data['z'], data['y'], data['score']
        if z.shape[1] >= 2:
            for label, color, text in ((0, BLUE, 'y = 0'), (1, ORANGE, 'y = 1')):
                m = y == label
                ax.scatter(z[m, 0], z[m, 1], s=9, color=color, alpha=0.55, lw=0, label=text)
            wrong = (score > 0) != (y == 1)
            ax.scatter(z[wrong, 0], z[wrong, 1], s=22, facecolors='none', edgecolors=INK, lw=0.5, label='misclassified')
            ax.set_xlabel(f'pooled knob dim {r["knob"]["dims"][0]}')
            ax.set_ylabel(f'pooled knob dim {r["knob"]["dims"][1]}')
        else:
            bins = np.linspace(z.min(), z.max(), 40)
            ax.hist(z[y == 0, 0], bins, color=BLUE, alpha=0.6, label='y = 0')
            ax.hist(z[y == 1, 0], bins, color=ORANGE, alpha=0.6, label='y = 1')
            ax.set_xlabel(f'pooled knob dim {r["knob"]["dims"][0]}')
            ax.set_ylabel('rows')
        acc = r['knob']['distilled_accuracy']
        ax.set_title(f'{r["boundary"]} boundary: k = {r["knob"]["k"]}, closed-form accuracy {acc:.3f}')
        ax.legend(fontsize=8)
        ax.grid(color=GRID, lw=0.6)
    save(fig, out, 'boundary_knob_space')


def intervention(out, results):
    table = pd.read_csv(out / 'intervention.csv')
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 3.6), squeeze=False, layout='constrained')
    for ax, r in zip(axes[0], results):
        sub = table[table.boundary == r['boundary']].sort_values('tau')
        ax.axhline(0, color=GRID, lw=1)
        ax.plot(sub.tau, sub.mean_normal, '-o', color=BLUE, lw=1.8, ms=4, mec='white', label='normal direction')
        ax.plot(sub.tau, sub.mean_tangent, '-s', color=ORANGE, lw=1.4, ms=4, mec='white', label='tangent control')
        ax.plot(sub.tau, sub.mean_random, '-^', color=AQUA, lw=1.4, ms=4, mec='white', label='random control')
        if 'mean_normal_y0' in sub:
            ax.plot(sub.tau, sub.mean_normal_y0, ':', color=BLUE, lw=1.2, label='normal, y = 0 rows')
            ax.plot(sub.tau, sub.mean_normal_y1, '--', color=BLUE, lw=1.2, label='normal, y = 1 rows')
        ax.set_title(f'{r["boundary"]} boundary: change in log p(realised next bar)')
        ax.set_xlabel('step size tau in knob space')
        ax.set_ylabel('mean delta log p')
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(color=GRID, lw=0.6)
    save(fig, out, 'boundary_intervention')


def main():
    out = Path(sys.argv[1])
    results = json.loads((out / 'results.json').read_text())['boundaries']
    maps(out, results)
    energy(out, results)
    knobs(out, results)
    intervention(out, results)


if __name__ == '__main__':
    main()
