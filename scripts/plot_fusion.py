#!/usr/bin/env python3
"""Figures for the evidence-fusion results. Reads only saved outputs."""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fusion_data import RESULTS

OUT = RESULTS / 'kronos_fusion'
if len(sys.argv) > 1:
    from run_fusion import ROUNDS
    OUT = ROUNDS[sys.argv[1]]
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6b66', '#e6e6e2'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9.5, 'axes.titleweight': 'medium', 'legend.frameon': False})
PRIMARY = ['banksim', 's_ffsd', 'sd_stock', 'sd_prices', 'sd_synthetic']
METHODS = [('fusion', 'Evidence fusion (Kronos + naive)', BLUE, 'o'),
           ('fusion_r1', 'Evidence fusion, round 1', BLUE, '^'),
           ('fusion_no_kronos', 'Evidence fusion, no Kronos', BLUE, 's'),
           ('fusion_ensemble', 'Fusion + Isolation Forest channel', BLUE, 'D'),
           ('per_feature_pot', 'Per-feature POT (published)', ORANGE, 'o'),
           ('joint_top3', 'Joint top-3 (published)', ORANGE, 's'),
           ('isolation_forest', 'Isolation Forest', AQUA, 'o'),
           ('robust_mad', 'Robust MAD', AQUA, 's')]


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=180)
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)


def labelled():
    metrics = pd.read_csv(OUT / 'metrics.csv').set_index(['dataset', 'method'])
    intervals = pd.read_csv(OUT / 'bootstrap.csv').set_index(['dataset', 'method'])
    methods = [m for m in METHODS if (PRIMARY[0], m[0]) in metrics.index]
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(13, 3.2 + 0.25 * len(methods)), sharey=True, layout='constrained')
    for ax, name in zip(axes, PRIMARY):
        prevalence = metrics.loc[(name, 'fusion'), 'prevalence']
        ax.axvline(prevalence, color=GRID, lw=1.5, zorder=0)
        ax.text(prevalence + 0.015, -0.55, 'chance', color=MUTED, fontsize=7.5, ha='left', va='center')
        for row, (key, label, color, marker) in enumerate(methods):
            y = len(methods) - 1 - row
            value = metrics.loc[(name, key), 'ap']
            lo, hi = intervals.loc[(name, key), ['ap_lo', 'ap_hi']]
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle='round', alpha=0.9)
            ax.plot(value, y, marker=marker, color=color, ms=7, mec='white', mew=1.2)
            ax.text(min(hi + 0.02, 0.98), y, f'{value:.2f}', va='center', fontsize=7.5, color=INK)
        positives = int(metrics.loc[(name, 'fusion'), 'positives'])
        ax.set_title(f'{name}  ({positives} positives)')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.8, len(methods) - 0.4)
        ax.set_xlabel('Average precision')
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(methods)), [m[1] for m in methods][::-1])
    fig.suptitle('Test rows identical to the published Kronos-proxy experiment; bars are 95% bootstrap intervals (whole groups or 64-row blocks)', fontsize=9, color=MUTED)
    save(fig, 'fusion_labelled')


def native():
    rows = pd.read_csv(OUT / 'native_rows.csv.gz', parse_dates=['timestamp'])
    summary = json.loads((OUT / 'native_summary.json').read_text())
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True, layout='constrained',
                             gridspec_kw=dict(height_ratios=[1.2, 0.8, 1.2]))
    t = rows.timestamp
    axes[0].plot(t, rows.close * 1e5, color=INK, lw=1)
    axes[0].set_ylabel('Close, 1e-5 BTC')
    axes[1].bar(t, rows.volume, width=pd.Timedelta(minutes=5), color=MUTED, lw=0)
    axes[1].set_ylabel('Volume, BRD')
    axes[1].set_yscale('symlog', linthresh=1000)
    axes[2].plot(t, rows.score, color=BLUE, lw=1)
    axes[2].axhline(summary['threshold'], color=BLUE, lw=1, ls='--')
    axes[2].text(t.iloc[2], summary['threshold'] + 0.3, 'frozen threshold (1% on held-out calibration)', color=BLUE, fontsize=7.5)
    alarms = rows[rows.fusion_alarm]
    axes[2].plot(alarms.timestamp, alarms.score, 'o', color=BLUE, ms=5, mec='white', mew=0.8, label=f'fusion alarm ({len(alarms)})')
    original = rows[rows.original_alarm]
    axes[2].plot(original.timestamp, np.full(len(original), -0.6), '|', color=ORANGE, ms=9, mew=1.2,
                 label=f'published per-feature POT alarm ({len(original)})')
    axes[2].set_ylabel('Fused score, -log min p')
    axes[2].legend(loc='upper right', fontsize=8)
    pump = rows.timestamp[rows.weak_label == 1].iloc[0]
    for ax in axes:
        ax.axvline(pump, color=YELLOW, lw=1.2, zorder=0)
        ax.grid(axis='y', color=GRID, lw=0.6)
    axes[0].text(pump, axes[0].get_ylim()[1], ' annotated pump bar, 2018-12-30 17:00 UTC', color='#8a5d00', fontsize=8, va='top')
    fig.suptitle('BRD/BTC 5-minute bars, test period 2018-12-28 to 2018-12-31: evidence fusion over the saved Kronos-mini forecasts', fontsize=10)
    save(fig, 'fusion_native_timeline')


def injection():
    power = pd.read_csv(OUT / 'injection_power.csv')
    names = power.dataset.drop_duplicates().tolist()
    arms = [('per_feature_pot', 'Per-feature POT (published)', ORANGE, '--'),
            ('kronos_only', 'Kronos forecast only', ORANGE, '-'),
            ('fusion_no_kronos', 'Fusion, no Kronos', BLUE, '--'),
            ('fusion', 'Fusion (Kronos + naive)', BLUE, '-')]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True, layout='constrained')
    for ax, name in zip(axes.ravel(), names):
        for key, label, color, style in arms:
            sub = power[(power.dataset == name) & (power.method == key)].sort_values('delta')
            ax.plot(sub.delta, sub.power, style, color=color, lw=1.8, marker='o', ms=4, mec='white', mew=0.6, label=label)
        ax.set_title(name)
        ax.grid(color=GRID, lw=0.6)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([2, 3, 4, 6, 8, 12])
    axes.ravel()[-1].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes.ravel()[-1].legend(handles, labels, loc='center', fontsize=8.5)
    for ax in axes[-1, :3]:
        ax.set_xlabel('shock size, step-MAD units')
    for ax in axes[:, 0]:
        ax.set_ylabel('share detected at 1% alarm rate')
    fig.suptitle('Counterfactual shocks on normal test rows: a shock at t cannot change a forecast made before t', fontsize=10)
    save(fig, 'fusion_injection_power')


def coverage():
    table = pd.read_csv(OUT / 'full_coverage.csv')
    names = table.dataset.drop_duplicates().tolist()
    arms = [('fusion_no_kronos', 'Fusion pipeline, naive forecasts (all scorable rows)', BLUE),
            ('IsolationForest', 'Isolation Forest (published)', AQUA), ('RobustMAD', 'Robust MAD (published)', YELLOW)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout='constrained')
    x = np.arange(len(names))
    for ax, metric, label in zip(axes, ['ap', 'recall_1pct_fpr'], ['Average precision', 'Recall at 1% false-positive rate']):
        for k, (key, name, color) in enumerate(arms):
            sub = table[table.method == key].set_index('dataset').reindex(names)
            ax.bar(x + (k - 1) * 0.27, sub[metric], width=0.25, color=color, label=name, lw=0)
        ax.set_xticks(x, names, rotation=40, ha='right')
        ax.set_ylabel(label)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Full test coverage without Kronos: the same scoring pipeline on every row with 32 rows of history, against the published level detectors', fontsize=9.5)
    save(fig, 'fusion_full_coverage')


if __name__ == '__main__':
    labelled()
    native()
    injection()
    coverage()
