#!/usr/bin/env python3
"""What do the knob's numbers mean, and which way should they be turned for detection?

For the frozen label knob of a real-Kronos knowledge-boundary run (results/2026-11-10) this script

  A. explains the values: which hidden dimensions the knob reads, how the five layers are weighted
     (alpha), the closed form Phi(z), how the knob coordinate z is distributed on normal and on
     anomalous test rows, how well each coordinate separates them on its own, what Phi does as z moves,
     and what z correlates with (forecast surprise B, level magnitude L, anomalies in the 32-row
     context, per-feature deviations of the row and of its context);
  B. steers the knob on the test rows and re-scores them: the same push tau along the normal
     direction of Phi (the knob's own "more anomalous" direction for tau > 0) is applied to every
     evaluated row, labels unseen, the teacher-forced NLL surprise B is recomputed, and AP, AUROC and
     recall at 1 % FPR of B(tau), and of the equal-budget OR of B(tau) with the unsteered knob score K,
     are compared with tau = 0.  A random direction of the same size is the control.

Nothing is fitted here: the knob predictor is loaded from the frozen run's checkpoint.
Outputs: results/2026-11-16/knob_values/<dataset>/.
"""
import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from boundary_standin import label_boundary
from explain_missrate import context_stats
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.boundary import hidden_patch
from kronos_ad.fusion import robust_reference
from kronos_states import KronosStates, load_kronos, proxy_sequences
from run_fusion import recall_at_fpr

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'results/2026-11-10'
OUT = ROOT / 'results/2026-11-16/knob_values'
TAUS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
CONTROL_TAUS = (-2.0, 2.0)
NORMAL_CAP = 6000
SEED = 20261116


def metrics(y, score):
    return dict(ap=float(average_precision_score(y, score)), auroc=float(roc_auc_score(y, score)),
                recall_1pct_fpr=recall_at_fpr(y, score))


def or_rule(y, b, k):
    """Equal-budget OR: alarm when B or K exceeds its own 99.5th percentile on normal rows."""
    alarm = (b > np.quantile(b[y == 0], 0.995)) | (k > np.quantile(k[y == 0], 0.995))
    return dict(or_recall=float(alarm[y == 1].mean()), or_fpr=float(alarm[y == 0].mean()))


def main(dataset, steer=True):
    warnings.filterwarnings('ignore')
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    name = label_boundary(dataset)
    with open(FROZEN / f'knowledge_boundary_kronos_{dataset}/{name}_state.pkl', 'rb') as fh:
        result, _, predictor, _ = pickle.load(fh)
    meta = json.loads((FROZEN / f'kronos_rows/{dataset}/meta.json').read_text())
    cache = np.load(FROZEN / f'kronos_rows/{dataset}/test.npz')
    t_test, h_test, nll, H = cache['targets'], cache['histories'], cache['nll'], cache['H'].astype(np.float32)
    data = load_rows(dataset)
    values, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    features = [f for f, k in zip(data['features'], keep) if k]
    F = len(features)
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    z_raw = ((values[:, keep] - center) / scale).astype(np.float64)
    positions = pd.Series(np.arange(len(groups))).groupby(groups).cumcount().to_numpy()
    y_all = labels[t_test].astype(int)
    known = np.isin(y_all, [0, 1])
    y = y_all[known]

    # ------------------------------------------------------------------ A. the values
    dims, alpha = predictor.dims_.tolist(), predictor.alpha()
    zk = predictor.pooled(H)                      # (n, k) standardized, layer-pooled knob coordinates
    K = predictor.score(H)
    grad = predictor.gradient(zk)
    log(f'{dataset}: label knob "{name}" reads hidden dimensions {dims}; layer weights alpha {[round(float(a), 3) for a in alpha]}')
    log(f'  closed form: {predictor.closed_form()}')
    info = dict(dataset=dataset, boundary=name, dims=dims, alpha=alpha.tolist(), closed_form=predictor.closed_form(),
                layer_mean=predictor.scaler_.mean_.tolist(), layer_std=predictor.scaler_.std_.tolist(),
                test_rows=int(known.sum()), anomalies=int(y.sum()))
    stats = []
    for j, dim in enumerate(dims):
        zj = zk[known, j]
        auc = roc_auc_score(y, zj)
        q = {f'{g}_p{p}': float(np.percentile(zj[y == c], p)) for g, c in (('normal', 0), ('anomaly', 1)) for p in (5, 25, 50, 75, 95)}
        stats.append(dict(coordinate=f'z{j + 1}', hidden_dim=dim, auroc_alone=float(auc),
                          anomaly_side='higher' if auc > 0.5 else 'lower',
                          dphi_dz_at_normal_median=float(predictor.gradient(np.median(zk[known][y == 0], axis=0)[None])[0, j]),
                          share_rows_phi_rises_with_z=float((grad[known, j] > 0).mean()), **q))
        log(f'  z{j + 1} (dim {dim}): normal median {q["normal_p50"]:+.2f} [{q["normal_p5"]:+.2f}, {q["normal_p95"]:+.2f}], '
            f'anomaly median {q["anomaly_p50"]:+.2f} [{q["anomaly_p5"]:+.2f}, {q["anomaly_p95"]:+.2f}]; AUROC alone {auc:.3f}; '
            f'Phi rises with z on {stats[-1]["share_rows_phi_rises_with_z"]:.0%} of rows')
    kq = {f'{g}_p{p}': float(np.percentile(K[known][y == c], p)) for g, c in (('normal', 0), ('anomaly', 1)) for p in (5, 25, 50, 75, 95)}
    stats.append(dict(coordinate='K=Phi(z)', hidden_dim=-1, auroc_alone=float(roc_auc_score(y, K[known])), anomaly_side='higher', **kq))
    log(f'  K = Phi(z): normal median {kq["normal_p50"]:+.2f}, anomaly median {kq["anomaly_p50"]:+.2f}; AUROC {stats[-1]["auroc_alone"]:.3f}')
    pd.DataFrame(stats).to_csv(out / 'value_stats.csv', index=False)

    base = np.median(zk[known][y == 0], axis=0)
    curve = []
    for j in range(len(dims)):
        for v in np.linspace(-3, 3, 25):
            point = base.copy()
            point[j] = v
            import torch
            with torch.no_grad():
                phi = float(predictor._phi(torch.tensor(point[None], dtype=torch.float32))[0])
            curve.append(dict(coordinate=f'z{j + 1}', z=float(v), phi=phi))
    pd.DataFrame(curve).to_csv(out / 'phi_curve.csv', index=False)

    prev_frac, run_pos = context_stats(labels, groups)
    absz = np.abs(z_raw)
    rows_k, hist_k = t_test[known], h_test[known]
    covariates = {'B_nll_surprise': nll[known], 'L_level_magnitude': absz[rows_k].max(axis=1),
                  'anomaly_share_in_context': prev_frac[rows_k], 'position_in_anomaly_run': run_pos[rows_k].astype(float),
                  'context_mean_abs_z': absz[hist_k].mean(axis=(1, 2))}
    for f, feat in enumerate(features):
        covariates[f'row_{feat}'] = z_raw[rows_k, f]
        covariates[f'context_{feat}'] = z_raw[hist_k][:, :, f].mean(axis=1)
    corr = []
    for cname, c in covariates.items():
        row = dict(covariate=cname)
        for j in range(len(dims)):
            row[f'z{j + 1}'] = float(spearmanr(zk[known, j], c, nan_policy='omit').statistic) if np.nanstd(c) > 0 else np.nan
        row['K'] = float(spearmanr(K[known], c, nan_policy='omit').statistic) if np.nanstd(c) > 0 else np.nan
        corr.append(row)
    corr = pd.DataFrame(corr)
    corr.to_csv(out / 'correlations.csv', index=False)
    log('  Spearman correlation of the knob with what the row and its context look like:')
    log(corr.round(3).to_string(index=False))

    # ------------------------------------------------------------------ B. steering for detection
    if steer:
        rng = np.random.default_rng(SEED)
        idx_known = np.flatnonzero(known)
        normal_idx = idx_known[y == 0]
        if len(normal_idx) > NORMAL_CAP:
            normal_idx = np.sort(rng.choice(normal_idx, NORMAL_CAP, replace=False))
        E = np.sort(np.r_[idx_known[y == 1], normal_idx])
        yE, HE, KE, rows, hist = y_all[E], H[E], K[E], t_test[E], h_test[E]
        model, tokenizer, official = load_kronos()
        states = KronosStates(model, tokenizer, official, seed=42)
        chunk = max(8, 1024 // F)

        def surprise(patch):
            ll = []
            for start in range(0, len(rows), chunk):
                sl = slice(start, start + chunk)
                x, stamp, _ = proxy_sequences(z_raw[hist[sl]], z_raw[rows[sl]], positions[rows[sl]], meta['prescale'])
                p = None if patch is None else np.repeat(patch[sl], F, axis=0)
                ll.append(states.loglik(x, stamp, patch=p).reshape(-1, F).sum(axis=1))
            return -np.concatenate(ll)

        t = time.time()
        b0 = surprise(None)
        log(f'  steering on {len(E)} rows ({int(yE.sum())} anomalies, {int((yE == 0).sum())} normal); unsteered B recomputed in '
            f'{time.time() - t:.0f}s, max |difference to cached NLL| {np.abs(b0 - nll[E]).max():.4f}')
        table = [dict(direction='none', tau=0.0, **metrics(yE, b0), **or_rule(yE, b0, KE),
                      median_dB_normal=0.0, median_dB_anomaly=0.0, mean_dB_normal=0.0, mean_dB_anomaly=0.0)]
        log(f'  tau  0.0 (unsteered): AP {table[0]["ap"]:.3f} AUROC {table[0]["auroc"]:.3f} recall@1% {table[0]["recall_1pct_fpr"]:.3f} '
            f'| B or K recall {table[0]["or_recall"]:.3f} at FPR {table[0]["or_fpr"]:.4f}')
        for direction, taus in (('normal', TAUS), ('random', CONTROL_TAUS)):
            for tau in taus:
                t = time.time()
                b = surprise(hidden_patch(predictor, HE, tau, direction, seed=0))
                d = b - b0
                row = dict(direction=direction, tau=tau, **metrics(yE, b), **or_rule(yE, b, KE),
                           median_dB_normal=float(np.median(d[yE == 0])), median_dB_anomaly=float(np.median(d[yE == 1])),
                           mean_dB_normal=float(d[yE == 0].mean()), mean_dB_anomaly=float(d[yE == 1].mean()))
                table.append(row)
                log(f'  {direction:6s} tau {tau:+.1f}: AP {row["ap"]:.3f} AUROC {row["auroc"]:.3f} recall@1% {row["recall_1pct_fpr"]:.3f} '
                    f'| B or K recall {row["or_recall"]:.3f} | mean change of B: normal {row["mean_dB_normal"]:+.3f}, '
                    f'anomaly {row["mean_dB_anomaly"]:+.3f} ({time.time() - t:.0f}s)')
                pd.DataFrame(table).to_csv(out / 'steering.csv', index=False)
        info['steering_rows'] = dict(total=int(len(E)), anomalies=int(yE.sum()), normal=int((yE == 0).sum()))
    (out / 'values.json').write_text(json.dumps(info, indent=1) + '\n')
    (out / 'run.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--no-steer', action='store_true')
    args = parser.parse_args()
    main(args.dataset, steer=not args.no_steer)
