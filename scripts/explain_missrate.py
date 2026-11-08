#!/usr/bin/env python3
"""What are the anomalies the knob recovers, and why can the hidden state see them?

Splits the labelled anomalies of each miss-rate run into three groups at the 1% FPR
operating point -- caught by the forecaster's NLL surprise B, missed by B but above the
label knob's own 1% FPR threshold (recovered by K), missed by both -- and profiles each
group by (i) the row itself (level magnitude, per-feature |z| against the reference) and
(ii) its context, the 32 rows the hidden state is computed from: how many of them are
labelled anomalies, whether the row is the first of a run, and the context's mean |z|.

Writes results/2026-11-08/subtle_anomalies.csv and prints the tables.
"""
import numpy as np
import pandas as pd

from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.fusion import robust_reference

LOOK = 32
RUNS = {'sapienza_15s': 'results/2026-11-08/knowledge_boundary_missrate',
        'banksim': 'results/2026-11-08/knowledge_boundary_missrate_banksim',
        's_ffsd': 'results/2026-11-08/knowledge_boundary_missrate_s_ffsd',
        'ulb_creditcard': 'results/2026-11-08/knowledge_boundary_missrate_ulb_creditcard'}


def context_stats(labels, groups):
    """Share of labelled anomalies in the LOOK rows before each row (same group) and the row's position in its run."""
    lab = (labels == 1).astype(int)
    prev_frac, run_pos = np.full(len(labels), np.nan), np.zeros(len(labels), int)
    for idx in pd.Series(np.arange(len(labels))).groupby(groups).indices.values():
        idx = np.asarray(idx)
        cs = np.concatenate([[0], np.cumsum(lab[idx])])
        run = 0
        for j, r in enumerate(idx):
            lo = max(0, j - LOOK)
            prev_frac[r] = (cs[j] - cs[lo]) / max(1, j - lo)
            run = run + 1 if lab[r] else 0
            run_pos[r] = run
    return prev_frac, run_pos


def main():
    out = []
    for name, d in RUNS.items():
        data = load_rows(name)
        values, labels, groups, parts = data['values'], data['labels'], data['groups'], data['parts']
        keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
        feats = [f for f, k in zip(data['features'], keep) if k]
        center, scale = robust_reference(values[parts == 'reference'][:, keep])
        z = np.abs((values[:, keep] - center) / scale)
        s = pd.read_csv(f'{d}/scores_test.csv.gz')
        prev_frac, run_pos = context_stats(labels, groups)
        s['prev_frac'], s['run_pos'] = prev_frac[s.row], run_pos[s.row]
        normal = s.pump == 0
        thr_b, thr_k = np.quantile(s.B_nll_surprise[normal], 0.99), np.quantile(s.K_label_knob[normal], 0.99)
        s['grp'] = np.where(s.pump == 0, 'normal', np.where(s.B_nll_surprise > thr_b, 'caught_by_B',
                            np.where(s.K_label_knob > thr_k, 'recovered_by_K', 'missed_by_both')))
        pos = pd.Series(np.arange(len(labels))).groupby(groups).cumcount().to_numpy()
        ctx = np.full((len(s), len(feats)), np.nan)
        for i, r in enumerate(s.row.to_numpy()):
            rows = np.arange(max(r - pos[r], r - LOOK), r)
            if len(rows):
                ctx[i] = z[rows].mean(axis=0)
        ctx = pd.DataFrame(ctx, columns=[f'ctx_{f}' for f in feats])
        rowz = pd.DataFrame(z[s.row.to_numpy()], columns=[f'row_{f}' for f in feats])
        frame = pd.concat([s[['grp', 'B_nll_surprise', 'L_level_magnitude', 'K_label_knob', 'prev_frac', 'run_pos']], rowz, ctx], axis=1)
        if name == 'ulb_creditcard':   # 29 PCA dims: keep the row/context summary only
            frame = frame[[c for c in frame.columns if not c.startswith(('row_V', 'ctx_V'))]]
        summary = frame.groupby('grp').median(numeric_only=True)
        summary.insert(0, 'n', frame.groupby('grp').size())
        summary.insert(1, 'share_with_anomaly_in_context', frame.groupby('grp').prev_frac.apply(lambda v: float((v > 0).mean())))
        summary.insert(2, 'share_first_in_run', frame.groupby('grp').run_pos.apply(lambda v: float((v == 1).mean())))
        summary.insert(0, 'dataset', name)
        print(f'\n=== {name}  (B threshold {thr_b:.2f}, K threshold {thr_k:.2f})')
        print(summary.round(3).to_string())
        out.append(summary.reset_index())
    pd.concat(out, ignore_index=True).to_csv('results/2026-11-08/subtle_anomalies.csv', index=False)


if __name__ == '__main__':
    main()
