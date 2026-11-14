#!/usr/bin/env python3
"""Model-size study: one table set for Kronos-mini / -small / -base on identical rows (results/2026-11-14/summary/).

Reads results/2026-11-14/<model>/{detector_study_<d>, knowledge_boundary_kronos_<d>, knowledge_boundary_missrate_kronos_<d>}
for every model and dataset that has them; writes CSVs and SUMMARY.md and prints the Markdown.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_kronos_rerun import DATASETS, NAMES, interval, md

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'results/2026-11-14'
MODELS = ['mini', 'small', 'base']
PARAMS = {'mini': '4.1 M', 'small': '24.7 M', 'base': '102.3 M'}


def main():
    out = S / 'summary'
    out.mkdir(exist_ok=True)
    skill, det, bound, miss = [], [], [], []
    for d in DATASETS:
        for m in MODELS:
            R = S / m
            p = R / f'detector_study_{d}'
            if (p / 'results.json').exists():
                s = pd.read_csv(p / 'forecast_skill.csv')
                j = json.loads((p / 'results.json').read_text())
                w = j['stack_weights']
                skill.append(dict(dataset=NAMES[d], model=m, features=len(s),
                                  mae_ratio_vs_last=float(np.median(s.mae_kronos / s.mae_last.replace(0, np.nan))),
                                  mae_ratio_vs_median=float(np.median(s.mae_kronos / s.mae_median.replace(0, np.nan))),
                                  beats_last=int((s.mae_kronos < s.mae_last).sum()), beats_median=int((s.mae_kronos < s.mae_median).sum()),
                                  fusion_weight=float(np.mean([x[-1] for x in w['fusion_r1_kronos']])) if 'fusion_r1_kronos' in w else np.nan))
                t = pd.read_csv(p / 'detectors.csv').set_index('detector')
                b = pd.read_csv(p / 'bootstrap.csv')
                ref = b[b.reference == 'fusion_r1_no_forecaster'].set_index('detector')
                for k in ('fusion_r1_kronos', 'fusion_r1_no_forecaster', 'published_pot_kronos', 'kronos_nll_surprise', 'robust_mad'):
                    if k not in t.index or (m != MODELS[0] and k in ('fusion_r1_no_forecaster', 'robust_mad')):
                        continue
                    r = t.loc[k]
                    row = dict(dataset=NAMES[d], model=m if 'kronos' in k else '-', detector=k, positives=int(r.positives), ap=r.ap, auroc=r.auroc,
                               recall_1pct_fpr=r.recall_1pct_fpr)
                    if k in ref.index:
                        row['ap_vs_no_forecaster'] = interval(ref.loc[k, 'ap_low'], ref.loc[k, 'ap_high'])
                    det.append(row)
            p = R / f'knowledge_boundary_kronos_{d}'
            if (p / 'results.json').exists():
                j = json.loads((p / 'results.json').read_text())
                for bd in j['boundaries']:
                    iv = pd.DataFrame(bd['intervention'])
                    big = iv[iv.tau.abs() >= 0.5]
                    bound.append(dict(dataset=NAMES[d], model=m, boundary='error' if bd['boundary'] == 'error' else 'label', logistic=bd['acc_logistic'],
                                      mlp=bd['acc_mlp'], cnn=bd['acc_cnn'], knob_k=bd['knob']['k'], knob_dims=str(bd['knob']['dims']),
                                      knob_acc=bd['knob']['distilled_accuracy'], random_k_mean=bd['knob']['random_control']['mean'],
                                      random_at_least_knob=bd['knob']['random_control']['fraction_at_least_topk'],
                                      dlogp_tau_minus1=float(iv.loc[iv.tau == -1.0, 'mean_normal'].iloc[0]), dlogp_tau_plus1=float(iv.loc[iv.tau == 1.0, 'mean_normal'].iloc[0]),
                                      intervention_rows=bd.get('intervention_rows'),
                                      passes_controls=bool((big.p_normal_vs_tangent < 0.05).all() and (big.p_normal_vs_random < 0.05).all())))
            p = R / f'knowledge_boundary_missrate_kronos_{d}'
            if (p / 'missrate.csv').exists():
                t = pd.read_csv(p / 'missrate.csv').set_index('arm')
                j = json.loads((p / 'results.json').read_text())
                row = dict(dataset=NAMES[d], model=m, positives=j['positives'])
                for arm, label in (('B_nll_surprise', 'B'), ('L_level_magnitude', 'L'), ('K_label_knob', 'K'), ('P_label_logistic', 'P logistic'),
                                   ('P_label_cnn', 'P cnn'), ('B+K_label [or@a/2]', 'B or K'), ('B+P_logistic [or@a/2]', 'B or P'),
                                   ('B+K_label [fisher]', 'B+K Fisher'), ('B+L [or@a/2]', 'B or L'), ('B+L+K_label [or@a/3]', 'B or L or K')):
                    if arm in t.index:
                        r = t.loc[arm]
                        row[f'recall {label}'] = r.recall_1pct_fpr if 'recall_1pct_fpr' in t.columns else 1 - r.miss_rate_1pct_fpr
                        if label in ('B or K', 'B or P', 'B+K Fisher'):
                            row[f'{label} vs B'] = f'{int(r.pumps_gained_vs_B):+d}/{-int(r.pumps_lost_vs_B):d} {interval(r.ci_low, r.ci_high)}'
                inc = pd.read_csv(p / 'increments.csv')
                x = inc[(inc.stat == 'or') & (inc.base == 'B_nll_surprise') & (inc.plus == 'B+K_label [or@a/2]')]
                if len(x):
                    row['subtle half: B -> B or K'] = f'{x.subtle_recall_base.iloc[0]:.3f} -> {x.subtle_recall_plus.iloc[0]:.3f} {interval(x.subtle_ci_low.iloc[0], x.subtle_ci_high.iloc[0])}'
                c = pd.read_csv(p / 'complementarity.csv').set_index('channel')
                if 'K_label_knob' in c.index:
                    row['K recovers of B-missed'] = c.loc['K_label_knob', 'recall_B_missed']
                miss.append(row)
    sections = [f'Models: ' + ', '.join(f'Kronos-{m} ({PARAMS[m]} parameters)' for m in MODELS) + '. Same rows for every model.']
    for name, title, rows in (('forecast_skill', 'Forecast skill (median over features of MAE / baseline MAE; below 1 = better than the baseline)', skill),
                              ('detectors', 'Detection (test rows, 1 % FPR)', det),
                              ('boundary', 'Knowledge boundary (balanced accuracy; label = pump or fraud)', bound),
                              ('missrate', 'Recall at 1 % FPR per channel and combination (miss rate = 1 - recall)', miss)):
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f'{name}.csv', index=False)
        sections.append(f'## {title}\n\n' + (md(frame) if len(frame) else '(not run yet)'))
    (out / 'SUMMARY.md').write_text('# Kronos model-size study\n\n' + '\n\n'.join(sections) + '\n')
    print('\n\n'.join(sections))


if __name__ == '__main__':
    main()
