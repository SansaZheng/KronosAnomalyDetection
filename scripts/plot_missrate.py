#!/usr/bin/env python3
"""Figures for a knowledge-boundary miss-rate directory (scripts/boundary_missrate.py output)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6b66', '#e6e6e2'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5, 'legend.frameon': False})
ARMS = ['B_nll_surprise', 'L_level_magnitude', 'K_label_knob', 'K_error_knob', 'P_label_logistic',
        'B+K_label [tippett]', 'B+K_label [fisher]', 'B+K_label [or@a/2]', 'B+K_error [fisher]', 'B+P_logistic [fisher]', 'B+P_logistic [or@a/2]',
        'B+L [fisher]', 'B+L [or@a/2]', 'B+L+K_label [fisher]', 'B+L+K_label [or@a/3]']
LABELS = ['B: NLL surprise', 'L: level magnitude', 'K: label knob', 'K: error knob', 'P: logistic probe',
          'B + K label (Tippett)', 'B + K label (Fisher)', 'B or K label (budget split)', 'B + K error (Fisher)', 'B + probe (Fisher)', 'B or probe (budget split)',
          'B + L (Fisher)', 'B or L (budget split)', 'B + L + K label (Fisher)', 'B or L or K label (budget split)']


def main():
    out = Path(sys.argv[1])
    header = json.loads((out / 'results.json').read_text())
    table = pd.read_csv(out / 'missrate.csv').set_index('arm')
    rows = pd.read_csv(out / 'scores_test.csv.gz')
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.6), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
    arms = [a for a in ARMS if a in table.index]
    labels = [l for a, l in zip(ARMS, LABELS) if a in table.index]
    miss = table.loc[arms, 'miss_rate_1pct_fpr'].to_numpy()
    lo = table.loc[arms, 'ci_low'].to_numpy()
    hi = table.loc[arms, 'ci_high'].to_numpy()
    base = table.loc['B_nll_surprise', 'miss_rate_1pct_fpr']
    colors = [BLUE if a.startswith('B_') else YELLOW if a.startswith(('L_', 'K_', 'P_')) else AQUA if 'K_' in a else ORANGE for a in arms]
    y = np.arange(len(arms))[::-1]
    left.barh(y, miss, color=colors, height=0.62)
    # paired-bootstrap interval of the recall difference versus B, mapped onto the arm's miss rate:
    # miss(arm) = miss(B) - diff, so the interval is [miss(B) - hi, miss(B) - lo]
    diff = table.loc[arms, 'recall_diff_vs_B'].to_numpy()
    left.errorbar(miss, y, xerr=np.vstack([np.clip(hi - diff, 0, None), np.clip(diff - lo, 0, None)]), fmt='none', ecolor=INK, elinewidth=0.8, capsize=2)
    left.axvline(base, color=MUTED, lw=1, ls='--')
    left.set_yticks(y)
    left.set_yticklabels(labels)
    left.set_xlabel('miss rate at 1% FPR on test normals (1 - recall)')
    left.set_title(f'{header["dataset"]}: {header["positives"]} labelled {header["label_boundary"]} rows; dashed = baseline B')
    left.grid(axis='x', color=GRID, lw=0.6)
    for yi, (m, a) in enumerate(zip(miss, arms)):
        g, l = int(table.loc[a, 'pumps_gained_vs_B']), int(table.loc[a, 'pumps_lost_vs_B'])
        if a != 'B_nll_surprise':
            left.text(min(max(m, base - lo[yi]) + 0.012, 0.97), y[yi], f'+{g} / -{l}', va='center', fontsize=7.5, color=MUTED)
    normal = rows[rows.pump == 0]
    pos = rows[rows.pump == 1]
    right.scatter(normal.B_nll_surprise, normal.K_label_knob, s=4, color=MUTED, alpha=0.25, lw=0, label='normal rows')
    right.scatter(pos.B_nll_surprise, pos.K_label_knob, s=18, color=ORANGE, lw=0, label=f'{header["label_boundary"]} rows')
    missed = pos[pos.missed_by_B == 1]
    right.scatter(missed.B_nll_surprise, missed.K_label_knob, s=50, facecolors='none', edgecolors=BLUE, lw=1.0, label='missed by B at 1% FPR')
    right.axvline(header['baseline_B_threshold_1pct_fpr'], color=BLUE, lw=1, ls='--')
    right.set_xscale('symlog', linthresh=10)
    right.set_xlabel('B: NLL surprise of the realised row (symlog)')
    right.set_ylabel('K: label-boundary knob score')
    right.set_title('hidden-state knob vs likelihood surprise')
    right.legend(fontsize=7.5, loc='best')
    right.grid(color=GRID, lw=0.6)
    fig.savefig(out / 'missrate.png', dpi=180)
    fig.savefig(out / 'missrate.pdf')


if __name__ == '__main__':
    main()
