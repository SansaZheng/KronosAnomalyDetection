#!/usr/bin/env python3
"""Collect the real-weight rerun into one table set (results/2026-11-10/summary/).

Reads, for every dataset that has them:
  detector_study_<d>/            forecast_skill.csv, detectors.csv, bootstrap.csv, results.json
  knowledge_boundary_kronos_<d>/ results.json (probes, knob, interventions), anomaly_detection.csv
  knowledge_boundary_missrate_kronos_<d>/ missrate.csv, increments.csv, complementarity.csv
  subtle_anomalies_kronos.csv
and the stand-in counterparts (results/2026-11-08/...) for the paired comparison.
Writes CSVs and prints Markdown tables.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R10, R08 = ROOT / 'results/2026-11-10', ROOT / 'results/2026-11-08'
DATASETS = ['xu_livshits_1h', 'banksim', 's_ffsd', 'sapienza_15s', 'ulb_creditcard']
NAMES = {'xu_livshits_1h': 'K-lines (Xu & Livshits + B0R0koko)', 'banksim': 'BankSim', 's_ffsd': 'S-FFSD', 'sapienza_15s': 'Sapienza 15 s',
         'ulb_creditcard': 'ULB credit card'}
STANDIN_MISS = {'sapienza_15s': R08 / 'knowledge_boundary_missrate', 'xu_livshits_1h': R08 / 'knowledge_boundary_missrate_xu_livshits_1h'}
STANDIN_BOUND = {'sapienza_15s': ROOT / 'results/2026-11-07/knowledge_boundary_standin'}


def standin_missrate_dir(d):
    return STANDIN_MISS.get(d, R08 / f'knowledge_boundary_missrate_{d}')


def standin_boundary_dir(d):
    return STANDIN_BOUND.get(d, R08 / f'knowledge_boundary_standin_{d}')


def md(frame, floatfmt='{:.3f}'):
    f = frame.copy()
    for c in f.columns:
        if f[c].dtype.kind == 'f':
            f[c] = f[c].map(lambda v: '' if pd.isna(v) else floatfmt.format(v))
    return '| ' + ' | '.join(f.columns) + ' |\n|' + '---|' * len(f.columns) + '\n' + '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in f.to_numpy())


def interval(lo, hi):
    return f'[{lo:+.3f}, {hi:+.3f}]'


def main():
    out = R10 / 'summary'
    out.mkdir(exist_ok=True)
    sections = []

    # ---- forecast skill and stacking weights
    rows = []
    for d in DATASETS:
        p = R10 / f'detector_study_{d}'
        if not (p / 'forecast_skill.csv').exists():
            continue
        s = pd.read_csv(p / 'forecast_skill.csv')
        j = json.loads((p / 'results.json').read_text())
        w = j['stack_weights']
        rows.append(dict(dataset=NAMES[d], features=len(s), mae_ratio_kronos_vs_last=float(np.median(s.mae_kronos / s.mae_last.replace(0, np.nan))),
                         mae_ratio_standin_vs_last=float(np.median(s.mae_standin / s.mae_last.replace(0, np.nan))),
                         features_where_kronos_beats_last=int((s.mae_kronos < s.mae_last).sum()),
                         features_where_kronos_beats_median=int((s.mae_kronos < s.mae_median).sum()),
                         kronos_weight_r1=float(np.mean([x[-1] for x in w['fusion_r1_kronos']])) if 'fusion_r1_kronos' in w else np.nan,
                         standin_weight_r1=float(np.mean([x[-1] for x in w['fusion_r1_standin']])) if 'fusion_r1_standin' in w else np.nan))
    skill = pd.DataFrame(rows)
    skill.to_csv(out / 'forecast_skill.csv', index=False)
    sections.append('## Forecast skill (test rows; MAE ratios are medians over features; weight = mean stacking weight of the model)\n\n' + md(skill))

    # ---- detectors
    keep = ['fusion_r1_kronos', 'fusion_r1_no_forecaster', 'fusion_r1_standin', 'fusion_r2_kronos', 'fusion_r2_no_forecaster', 'published_pot_kronos',
            'published_pot_last', 'kronos_nll_surprise', 'standin_nll_surprise', 'isolation_forest', 'robust_mad']
    rows = []
    for d in DATASETS:
        p = R10 / f'detector_study_{d}'
        if not (p / 'detectors.csv').exists():
            continue
        t = pd.read_csv(p / 'detectors.csv').set_index('detector')
        b = pd.read_csv(p / 'bootstrap.csv')
        ref = b[b.reference == 'fusion_r1_no_forecaster'].set_index('detector')
        for k in keep:
            if k not in t.index:
                continue
            r = t.loc[k]
            row = dict(dataset=NAMES[d], detector=k, positives=int(r.positives), ap=r.ap, auroc=r.auroc, recall_1pct_fpr=r.recall_1pct_fpr, alarm_rate=r.alarm_rate)
            if k in ref.index:
                row['ap_vs_no_forecaster'] = interval(ref.loc[k, 'ap_low'], ref.loc[k, 'ap_high'])
                row['recall_vs_no_forecaster'] = interval(ref.loc[k, 'recall_low'], ref.loc[k, 'recall_high'])
            rows.append(row)
    det = pd.DataFrame(rows)
    det.to_csv(out / 'detectors.csv', index=False)
    sections.append('## Detectors on the same test rows (paired intervals against the fusion without any forecaster)\n\n' + md(det))

    # ---- knowledge boundary
    rows = []
    for d in DATASETS:
        for model, p in (('kronos', R10 / f'knowledge_boundary_kronos_{d}'), ('standin', standin_boundary_dir(d))):
            if not (p / 'results.json').exists():
                continue
            j = json.loads((p / 'results.json').read_text())
            for bd in j['boundaries']:
                iv = pd.DataFrame(bd['intervention'])
                big = iv[iv.tau.abs() >= 0.5]
                rows.append(dict(dataset=NAMES[d], model=model, boundary=bd['boundary'], logistic=bd['acc_logistic'], mlp=bd['acc_mlp'], cnn=bd['acc_cnn'],
                                 knob_k=bd['knob']['k'], knob_dims=str(bd['knob']['dims']), knob_acc=bd['knob']['distilled_accuracy'],
                                 random_k_mean=bd['knob']['random_control']['mean'], random_at_least_knob=bd['knob']['random_control']['fraction_at_least_topk'],
                                 dlogp_tau1=float(iv.loc[iv.tau == 1.0, 'mean_normal'].iloc[0]), dlogp_tau1_tangent=float(iv.loc[iv.tau == 1.0, 'mean_tangent'].iloc[0]),
                                 dlogp_tau1_random=float(iv.loc[iv.tau == 1.0, 'mean_random'].iloc[0]),
                                 p_vs_tangent_min=float(big.p_normal_vs_tangent.min()), p_vs_random_min=float(big.p_normal_vs_random.min()),
                                 passes_controls=bool((big.p_normal_vs_tangent < 0.05).all() and (big.p_normal_vs_random < 0.05).all())))
    bound = pd.DataFrame(rows)
    bound.to_csv(out / 'boundary.csv', index=False)
    sections.append('## Knowledge boundary (balanced accuracy of probes and knob; interventions at |tau| >= 0.5: change of log-likelihood along the knob, '
                    'tangent and random directions, and whether the knob beats both controls at every such tau)\n\n' + md(bound))

    # ---- miss rate
    arms = {'B_nll_surprise': 'B', 'L_level_magnitude': 'L', 'K_label_knob': 'K', 'P_label_logistic': 'probe', 'B+K_label [tippett]': 'B+K Tippett',
            'B+K_label [fisher]': 'B+K Fisher', 'B+K_label [or@a/2]': 'B or K', 'B+P_logistic [or@a/2]': 'B or probe', 'B+L [or@a/2]': 'B or L',
            'B+L+K_label [or@a/3]': 'B or L or K'}
    rows = []
    for d in DATASETS:
        for model, p in (('kronos', R10 / f'knowledge_boundary_missrate_kronos_{d}'), ('standin', standin_missrate_dir(d))):
            if not (p / 'missrate.csv').exists():
                continue
            t = pd.read_csv(p / 'missrate.csv').set_index('arm')
            j = json.loads((p / 'results.json').read_text())
            row = dict(dataset=NAMES[d], model=model, positives=j['positives'], units=j['bootstrap_units'])
            for arm, label in arms.items():
                if arm in t.index:
                    r = t.loc[arm]
                    row[f'miss {label}'] = r.miss_rate_1pct_fpr
                    if label in ('B or K', 'B+K Fisher', 'B or L or K', 'B or L'):
                        row[f'{label} vs B'] = f'{int(r.pumps_gained_vs_B):+d}/{-int(r.pumps_lost_vs_B):d} {interval(r.ci_low, r.ci_high)}'
            inc = pd.read_csv(p / 'increments.csv')
            x = inc[(inc.stat == 'or') & (inc.base == 'B+L [or@a/2]')]
            if len(x):
                row['B or L -> B or L or K'] = f'{x.recall_base.iloc[0]:.3f} -> {x.recall_plus.iloc[0]:.3f} {interval(x.ci_low.iloc[0], x.ci_high.iloc[0])}'
            x = inc[(inc.stat == 'or') & (inc.base == 'B_nll_surprise') & (inc.plus == 'B+K_label [or@a/2]')]
            if len(x):
                row['subtle half: B -> B or K'] = f'{x.subtle_recall_base.iloc[0]:.3f} -> {x.subtle_recall_plus.iloc[0]:.3f} {interval(x.subtle_ci_low.iloc[0], x.subtle_ci_high.iloc[0])}'
            c = pd.read_csv(p / 'complementarity.csv').set_index('channel')
            if 'K_label_knob' in c.index:
                row['K recovers of B-missed'] = c.loc['K_label_knob', 'recall_B_missed']
                row['spearman(K,B)'] = c.loc['K_label_knob', 'spearman_with_B_on_pumps']
            rows.append(row)
    miss = pd.DataFrame(rows)
    miss.to_csv(out / 'missrate.csv', index=False)
    sections.append('## Miss rate at 1% FPR (recall differences vs B: anomalies gained/lost and the paired-bootstrap 95% interval)\n\n' + md(miss))

    # ---- who is recovered
    p = R10 / 'subtle_anomalies_kronos.csv'
    if p.exists():
        s = pd.read_csv(p)
        cols = [c for c in ['dataset', 'grp', 'n', 'share_with_anomaly_in_context', 'share_first_in_run', 'B_nll_surprise', 'L_level_magnitude', 'K_label_knob',
                            'prev_frac', 'run_pos', 'ctx_zero_volume_share', 'ctx_volume_rise_last8h'] if c in s.columns]
        sections.append('## Who the recovered anomalies are (group medians; real Kronos)\n\n' + md(s[cols]))
    (out / 'SUMMARY.md').write_text('# Real-weight rerun summary\n\n' + '\n\n'.join(sections) + '\n')
    print('\n\n'.join(sections))


if __name__ == '__main__':
    main()
