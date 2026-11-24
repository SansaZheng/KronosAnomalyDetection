#!/usr/bin/env python3
"""Kronos against the unsupervised detectors of SansaZheng/Financial_unsuperviesdAD, on identical rows.

The twelve detectors of that repository (ten classical: Isolation Forest, LOF, kNN, One-Class SVM, PCA-SPE,
HBOS, GMM, Autoencoder, VAE, Deep SVDD; two 2026 preprint adaptations: RGLD-ID, RTTAD-TTA) are fitted with
that repository's own code, preprocessing and default budgets (at most 8,192 unlabelled training rows,
400 / 200 neural steps, seeds 42-44) on the *reference period* of this project's split, and scored on the
same test rows as Kronos.  Features are the same columns Kronos sees.  Labels are used only for evaluation.

Per detector: AP, AUROC, recall at 1 % false alarms on the labelled test rows, and what the Kronos knob K
adds when it is Fisher-combined with the detector's score (p-values against normal calibration rows).
Outputs: results/2026-11-24/recent_methods/<dataset>/.
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from boundary_missrate import conformal_logp, fisher
from boundary_standin import label_boundary
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from run_fusion import recall_at_fpr

ROOT = Path(__file__).resolve().parents[1]
FINAD = Path(os.environ.get('FINAD_REPO', '/home/claude/sansazheng/financial_unsuperviesdad'))
FROZEN = ROOT / 'results/2026-11-10'
OUT = ROOT / 'results/2026-11-24/recent_methods'
SEEDS, TRAIN_CAP, ADAPT_CAP = (42, 43, 44), 8192, 8192
NAMES = {'iforest': 'Isolation Forest', 'lof': 'LOF', 'knn': 'kNN', 'ocsvm': 'One-Class SVM', 'pca': 'PCA-SPE', 'hbos': 'HBOS',
         'gmm': 'GMM', 'ae': 'Autoencoder', 'vae': 'VAE', 'deep_svdd': 'Deep SVDD', 'rgld2026': 'RGLD-ID (2026)',
         'rttad2026': 'RTTAD-TTA (2026)'}


def metrics(y, score):
    return dict(ap=float(average_precision_score(y, score)), auroc=float(roc_auc_score(y, score)),
                recall_1pct_fpr=recall_at_fpr(y, score))


def main(dataset):
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(FINAD))
    from finad.detectors import METHODS, make_detector
    from finad.preprocessing import Preprocessor
    torch.set_num_threads(2)
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    name = label_boundary(dataset)
    with open(FROZEN / f'knowledge_boundary_kronos_{dataset}/{name}_state.pkl', 'rb') as fh:
        _, _, predictor, _ = pickle.load(fh)
    caches = {s: np.load(FROZEN / f'kronos_rows/{dataset}/{s}.npz') for s in ('cal', 'test')}
    data = load_rows(dataset)
    values, parts, labels = data['values'], data['parts'], data['labels']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    X = values[:, keep].astype(np.float64)
    reference = np.flatnonzero(parts == 'reference')
    t_cal, t_test = caches['cal']['targets'], caches['test']['targets']
    y_all = labels[t_test].astype(int)
    known = np.isin(y_all, [0, 1])
    y = y_all[known]
    null = labels[t_cal].astype(int) == 0
    kronos = {'B': (caches['test']['nll'].astype(float), caches['cal']['nll'].astype(float)[null]),
              'K': (predictor.score(caches['test']['H'].astype(np.float32)), predictor.score(caches['cal']['H'].astype(np.float32))[null])}
    logp = {k: conformal_logp(*v) for k, v in kronos.items()}
    print(f'{dataset}: train on up to {TRAIN_CAP} of {len(reference)} unlabelled reference rows, {X.shape[1]} features; '
          f'test {int(known.sum())} labelled rows, {int(y.sum())} anomalies', flush=True)
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            t0 = time.time()
            train_idx = np.sort(np.random.default_rng(seed).choice(reference, min(TRAIN_CAP, len(reference)), replace=False))
            prep = Preprocessor().fit(X[train_idx])
            model = make_detector(method, seed, 'cpu', 400, 200).fit(prep.transform(X[train_idx]))
            if method == 'rttad2026':          # bounded transductive adaptation on unlabelled test features, as in that repository
                pool = np.sort(np.random.default_rng(seed + 2000).choice(t_test, min(ADAPT_CAP, len(t_test)), replace=False))
                model.adapt(prep.transform(X[pool]))
            score = lambda idx: np.concatenate([model.score(prep.transform(X[idx[i:i + 8192]])) for i in range(0, len(idx), 8192)])
            s_test, s_null = score(t_test), score(t_cal[null])
            lp = conformal_logp(s_test, s_null)
            row = dict(method=method, name=NAMES[method], seed=seed, **metrics(y, s_test[known]), seconds=time.time() - t0)
            for tag, keys in (('plus_K', 'K'), ('plus_B_K', 'BK')):
                fused = fisher(np.stack([lp] + [logp[k] for k in keys]))
                row.update({f'{tag}_{m}': v for m, v in metrics(y, fused[known]).items()})
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / 'per_seed.csv', index=False)
            print(f'  {NAMES[method]:18s} seed {seed}: AP {row["ap"]:.3f} AUROC {row["auroc"]:.3f} recall@1% {row["recall_1pct_fpr"]:.3f} '
                  f'| + K: AP {row["plus_K_ap"]:.3f} recall {row["plus_K_recall_1pct_fpr"]:.3f} '
                  f'| + B + K: AP {row["plus_B_K_ap"]:.3f} ({row["seconds"]:.0f}s)', flush=True)
    frame = pd.DataFrame(rows)
    cols = [c for c in frame.columns if c not in ('method', 'name', 'seed')]
    summary = frame.groupby(['method', 'name'], sort=False)[cols].agg(['mean', 'std'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    summary.reset_index().to_csv(out / 'summary.csv', index=False)
    (out / 'protocol.json').write_text(json.dumps(dict(
        dataset=dataset, finad_repo=str(FINAD), finad_commit=os.popen(f'git -C {FINAD} rev-parse HEAD').read().strip(),
        train_rows=int(min(TRAIN_CAP, len(reference))), reference_rows=int(len(reference)), features=int(X.shape[1]),
        test_rows=int(known.sum()), anomalies=int(y.sum()), null_rows=int(null.sum()), seeds=list(SEEDS),
        neural_steps=400, pretrain_steps=200), indent=1) + '\n')
    print(summary.round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    main(parser.parse_args().dataset)
