#!/usr/bin/env python3
"""S-FFSD: how far does fusing ordinary detectors go, and what does the Kronos knob add on top?

Step 1 pushes the fusion of the twelve Financial_unsuperviesdAD detectors, the level magnitude L and the
Kronos NLL surprise B as far as it goes: label-free Fisher fusion of everything, a Fisher subset chosen
greedily on labelled calibration rows, and stacked models (logistic regression, gradient boosting) trained
on those calibration labels.  Step 2 repeats each of them with the Kronos knob score K added.  Both steps
use exactly the same labels (labelled calibration rows), so the difference is the knob.

Detectors are fitted on unlabelled reference rows (seeds 42-44).  p-values use one random half of the
normal calibration rows; selection and stacking use the other half plus all calibration fraud rows.
Everything is evaluated once on the labelled test rows.  Outputs: results/2026-11-25/sffsd_fusion/.
"""
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from boundary_missrate import conformal_logp
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.fusion import robust_reference
from run_fusion import recall_at_fpr

ROOT = Path(__file__).resolve().parents[1]
FINAD = Path(os.environ.get('FINAD_REPO', '/home/claude/sansazheng/financial_unsuperviesdad'))
FROZEN, OUT = ROOT / 'results/2026-11-10', ROOT / 'results/2026-11-25/sffsd_fusion'
DATASET, SEEDS, TRAIN_CAP = 's_ffsd', (42, 43, 44), 8192


def metrics(y, score):
    out = dict(ap=float(average_precision_score(y, score)), auroc=float(roc_auc_score(y, score)))
    for fpr in (0.01, 0.05, 0.10):
        thr = np.quantile(score[y == 0], 1 - fpr)
        out[f'recall_{int(100 * fpr)}pct_fpr'] = float((score[y == 1] > thr).mean())
    return out


def greedy(Z, y, names, start=()):
    """Forward selection of channels whose summed -log p maximises AP on the selection rows."""
    chosen, best = list(start), (average_precision_score(y, Z[:, [names.index(c) for c in start]].sum(axis=1)) if start else -1)
    while True:
        trial = [(average_precision_score(y, Z[:, [names.index(c) for c in chosen + [n]]].sum(axis=1)), n) for n in names if n not in chosen]
        if not trial or max(trial)[0] <= best + 1e-4:
            return chosen
        best, pick = max(trial)
        chosen.append(pick)


def main():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(FINAD))
    from finad.detectors import METHODS, make_detector
    from finad.preprocessing import Preprocessor
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(FROZEN / f'knowledge_boundary_kronos_{DATASET}/fraud_state.pkl', 'rb') as fh:
        _, _, predictor, _ = pickle.load(fh)
    caches = {s: np.load(FROZEN / f'kronos_rows/{DATASET}/{s}.npz') for s in ('cal', 'test')}
    data = load_rows(DATASET)
    values, parts, labels = data['values'], data['parts'], data['labels']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    X = values[:, keep].astype(np.float64)
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    level = np.abs((X - center) / scale).max(axis=1)
    reference = np.flatnonzero(parts == 'reference')
    t = {s: caches[s]['targets'] for s in caches}
    y = {s: labels[t[s]].astype(int) for s in caches}
    lab = {s: np.isin(y[s], [0, 1]) for s in caches}
    kron = {s: dict(B=caches[s]['nll'].astype(float), K=predictor.score(caches[s]['H'].astype(np.float32)), L=level[t[s]]) for s in caches}
    rows, singles, chosen_log = [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        normal_cal = rng.permutation(np.flatnonzero(y['cal'] == 0))
        null, sel = np.sort(normal_cal[: len(normal_cal) // 2]), np.sort(np.r_[normal_cal[len(normal_cal) // 2:], np.flatnonzero(y['cal'] == 1)])
        train_idx = np.sort(rng.choice(reference, min(TRAIN_CAP, len(reference)), replace=False))
        prep = Preprocessor().fit(X[train_idx])
        raw = {s: dict(kron[s]) for s in caches}
        for method in METHODS:
            model = make_detector(method, seed, 'cpu', 400, 200).fit(prep.transform(X[train_idx]))
            if method == 'rttad2026':
                pool = np.sort(np.random.default_rng(seed + 2000).choice(t['test'], min(8192, len(t['test'])), replace=False))
                model.adapt(prep.transform(X[pool]))
            for s in caches:
                raw[s][method] = np.concatenate([model.score(prep.transform(X[t[s][i:i + 8192]])) for i in range(0, len(t[s]), 8192)])
        names = list(METHODS) + ['L', 'B', 'K']
        Z = {s: np.column_stack([-conformal_logp(raw[s][n], raw['cal'][n][null]) for n in names]) for s in caches}
        # 32-row context mean of every channel (previous rows only, labels unseen): the same window Kronos reads
        ctx = {s: pd.DataFrame(Z[s]).shift(1).rolling(32, min_periods=1).mean().bfill().to_numpy() for s in caches}
        Z = {s: np.column_stack([Z[s], ctx[s]]) for s in caches}
        base_names = list(names)
        names = base_names + [f'ctx_{n}' for n in base_names]
        Zs, ys, Zt, yt = Z['cal'][sel], y['cal'][sel], Z['test'][lab['test']], y['test'][lab['test']]
        col = lambda keys: [names.index(k) for k in keys]
        ordinary, no_knob = list(METHODS) + ['L'], list(METHODS) + ['L', 'B']
        for n in base_names:
            singles.append(dict(seed=seed, channel=n, **metrics(yt, Zt[:, names.index(n)])))

        def add(step, arm, score, **extra):
            rows.append(dict(seed=seed, step=step, arm=arm, **metrics(yt, score), **extra))

        def stack(keys, kind):
            if kind == 'cv':      # L1 logistic regression, strength chosen by 5-fold cross-validation on the selection rows
                est = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='saga', scoring='average_precision',
                                                                           max_iter=5000, random_state=seed))
                return est.fit(Zs[:, col(keys)], ys).predict_proba(Zt[:, col(keys)])[:, 1]
            est = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)) if kind == 'lr' else \
                HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=seed)
            return est.fit(Zs[:, col(keys)], ys).predict_proba(Zt[:, col(keys)])[:, 1]

        add('1 ordinary methods', 'Fisher of all 12 detectors + L (label-free)', Zt[:, col(ordinary)].sum(axis=1))
        g1 = greedy(Zs[:, col(no_knob)], ys, no_knob)
        add('1 ordinary methods', 'Fisher of a subset chosen on calibration labels', Zt[:, col(g1)].sum(axis=1), channels='+'.join(g1))
        add('1 ordinary methods', 'stacked logistic regression (12 + L + B)', stack(no_knob, 'lr'))
        add('1 ordinary methods', 'stacked gradient boosting (12 + L + B)', stack(no_knob, 'gb'))
        add('2 with the knob', 'knob K alone', Zt[:, names.index('K')])
        add('2 with the knob', 'Fisher of the chosen subset + K', Zt[:, col(g1 + ['K'])].sum(axis=1), channels='+'.join(g1 + ['K']))
        g2 = greedy(Zs[:, col(no_knob + ['K'])], ys, no_knob + ['K'])
        add('2 with the knob', 'Fisher of a subset chosen with K available', Zt[:, col(g2)].sum(axis=1), channels='+'.join(g2))
        add('2 with the knob', 'stacked logistic regression (12 + L + B + K)', stack(no_knob + ['K'], 'lr'))
        add('2 with the knob', 'stacked gradient boosting (12 + L + B + K)', stack(no_knob + ['K'], 'gb'))
        with_ctx = no_knob + [f'ctx_{n}' for n in no_knob]
        add('1 ordinary methods', 'stacked logistic regression (12 + L + B, with 32-row context means)', stack(with_ctx, 'lr'))
        add('2 with the knob', 'stacked logistic regression (12 + L + B + K, with 32-row context means)', stack(with_ctx + ['K', 'ctx_K'], 'lr'))
        add('2 with the knob', 'stacked logistic regression (12 + L + B with context means, + K only)', stack(with_ctx + ['K'], 'lr'))
        add('2 with the knob', 'stacked logistic regression (K + its context mean)', stack(['K', 'ctx_K'], 'lr'))
        add('1 ordinary methods', 'cross-validated L1 logistic regression (12 + L + B)', stack(no_knob, 'cv'))
        add('1 ordinary methods', 'cross-validated L1 logistic regression (12 + L + B, with context means)', stack(with_ctx, 'cv'))
        add('2 with the knob', 'cross-validated L1 logistic regression (12 + L + B + K)', stack(no_knob + ['K'], 'cv'))
        add('2 with the knob', 'cross-validated L1 logistic regression (12 + L + B + K, with context means)', stack(with_ctx + ['K', 'ctx_K'], 'cv'))
        chosen_log.append(dict(seed=seed, without_knob=g1, with_knob=g2, null_rows=int(len(null)), selection_rows=int(len(sel)),
                               selection_fraud=int(ys.sum())))
        print(pd.DataFrame([r for r in rows if r['seed'] == seed]).drop(columns=['seed']).round(3).to_string(index=False), flush=True)
    frame, single = pd.DataFrame(rows), pd.DataFrame(singles)
    frame.to_csv(OUT / 'per_seed.csv', index=False)
    num = [c for c in frame.columns if c not in ('seed', 'step', 'arm', 'channels')]
    summary = frame.groupby(['step', 'arm'], sort=False)[num].agg(['mean', 'std'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    summary.reset_index().to_csv(OUT / 'summary.csv', index=False)
    s2 = single.groupby('channel', sort=False)[num].mean().reset_index()
    s2.to_csv(OUT / 'single_channels.csv', index=False)
    (OUT / 'protocol.json').write_text(json.dumps(dict(
        dataset=DATASET, test_rows=int(lab['test'].sum()), test_fraud=int((y['test'] == 1).sum()), seeds=list(SEEDS),
        selections=chosen_log, finad_commit=os.popen(f'git -C {FINAD} rev-parse HEAD').read().strip()), indent=1) + '\n')
    print(summary.round(3).to_string())
    print(s2.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
