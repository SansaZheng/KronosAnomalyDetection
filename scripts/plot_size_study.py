"""Figure for the model-size study: recall at 1 % FPR of B (NLL surprise) and of "B or K" per Kronos size."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'results/2026-11-14/summary'
frame = pd.read_csv(S / 'missrate.csv')
names = list(dict.fromkeys(frame['dataset']))
fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.4), sharey=False)
for ax, name in zip(axes, names):
    part = frame[frame['dataset'] == name].set_index('model').loc[['mini', 'small', 'base']]
    x = range(3)
    ax.bar([i - 0.2 for i in x], part['recall B'], 0.38, color='#8a8f98', label='B: NLL surprise')
    ax.bar([i + 0.2 for i in x], part['recall B or K'], 0.38, color='#1f6f8b', label='B or K (knob)')
    for i, (b, k) in enumerate(zip(part['recall B'], part['recall B or K'])):
        ax.text(i - 0.2, b, f'{b:.2f}', ha='center', va='bottom', fontsize=7)
        ax.text(i + 0.2, k, f'{k:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(list(x), ['mini', 'small', 'base'])
    ax.set_title(name.split(' (')[0], fontsize=10)
    ax.set_ylim(0, max(0.05, 1.2 * max(part['recall B'].max(), part['recall B or K'].max())))
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('recall at 1 % FPR (higher = fewer misses)')
axes[2].legend(fontsize=7, frameon=False, loc='upper left')
fig.suptitle('Kronos size vs. recall of the forecast-surprise channel with and without the knob', fontsize=11)
fig.tight_layout()
fig.savefig(S / 'size_vs_recall.png', dpi=160)
print(S / 'size_vs_recall.png')
