#!/usr/bin/env python3
"""S-FFSD: search for the best Kronos-based detector, with an honest, forward-in-time selection rule.

Every candidate is a set of per-row features and a combiner trained on labelled rows.  Candidates are
ranked on the *calibration* period after training on the *reference* period only (forward validation, the
same direction of drift as deployment); the test period is scored once per candidate after refitting on
reference + calibration and is never used to choose.  The table reports both, so nothing is hidden.

Feature blocks
  ORD     twelve Financial_unsuperviesdAD detectors (fitted on unlabelled reference rows) + level magnitude L
  RAW     the five standardised input features
  B       Kronos NLL surprise, total and per feature
  K       the knob: hidden dimension 220, one value per layer (the model combines the layers)
  K20     the twenty top-ranked knob dimensions of the frozen run, last layer and layer mean
  PCA     sixteen principal components of the last-layer hidden state (fitted on reference rows)
  *_ctx   rolling means (previous 8 / 32 / 128 rows) and an exponential average of a block, labels unseen
  @128    the same Kronos block computed with a 128-row context instead of 32
  F32/F128  NLL surprise with the context frozen at the start of each 32- / 128-row "day"

Outputs: results/2026-11-26/sffsd_best/{candidates.csv, selected.json, run.log}.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.fusion import robust_reference

ROOT = Path(__file__).resolve().parents[1]
FINAD = Path(os.environ.get('FINAD_REPO', '/home/claude/sansazheng/financial_unsuperviesdad'))
FROZEN, OUT = ROOT / 'results/2026-11-10', ROOT / 'results/2026-11-26/sffsd_best'
SEED, TRAIN_CAP, START = 42, 8192, 128                 # rows before START lack a 128-row context and are dropped everywhere


def metrics(y, score, prefix=''):
    out = {f'{prefix}ap': float(average_precision_score(y, score)), f'{prefix}auroc': float(roc_auc_score(y, score))}
    for fpr in (0.01, 0.05, 0.10):
        out[f'{prefix}recall_{int(100 * fpr)}pct'] = float((score[y == 1] > np.quantile(score[y == 0], 1 - fpr)).mean())
    return out


def context(block, names):
    frame = pd.DataFrame(block).shift(1)
    parts, labels = [], []
    for w in (8, 32, 128):
        parts.append(frame.rolling(w, min_periods=1).mean().bfill().to_numpy())
        labels += [f'{n}_mean{w}' for n in names]
    parts.append(frame.ewm(halflife=16).mean().bfill().to_numpy())
    labels += [f'{n}_ewm' for n in names]
    return np.column_stack(parts), labels


def build_blocks(log):
    sys.path.insert(0, str(FINAD))
    from finad.detectors import METHODS, make_detector
    from finad.preprocessing import Preprocessor
    data = load_rows('s_ffsd')
    values, parts, labels = data['values'], data['parts'], data['labels'].astype(int)
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    X = values[:, keep].astype(np.float64)
    center, scale = robust_reference(X[parts == 'reference'])
    z = (X - center) / scale
    reference = np.flatnonzero(parts == 'reference')
    blocks = {'RAW': (np.clip(z, -10, 10), [f'raw{i}' for i in range(X.shape[1])])}
    cache = OUT / 'ordinary_scores.npz'
    if cache.exists():
        ordinary = np.load(cache)['scores']
    else:
        train_idx = np.sort(np.random.default_rng(SEED).choice(reference, TRAIN_CAP, replace=False))
        prep = Preprocessor().fit(X[train_idx])
        cols = []
        for method in METHODS:
            model = make_detector(method, SEED, 'cpu', 400, 200).fit(prep.transform(X[train_idx]))
            if method == 'rttad2026':
                pool = np.sort(np.random.default_rng(SEED + 2000).choice(np.flatnonzero(parts == 'test'), 8192, replace=False))
                model.adapt(prep.transform(X[pool]))
            cols.append(np.concatenate([model.score(prep.transform(X[i:i + 8192])) for i in range(0, len(X), 8192)]))
            log(f'  fitted {method}')
        ordinary = np.column_stack(cols + [np.abs(z).max(axis=1)])
        np.savez(cache, scores=ordinary)
    # rank-normalise each detector against the reference period so that heavy tails do not dominate
    ref_sorted = np.sort(ordinary[reference], axis=0)
    ranks = np.column_stack([np.searchsorted(ref_sorted[:, j], ordinary[:, j]) / len(reference) for j in range(ordinary.shape[1])])
    blocks['ORD'] = (-np.log1p(-np.clip(ranks, 0, 1 - 1e-4)), list(METHODS) + ['L'])
    ranking = json.loads((FROZEN / 'knowledge_boundary_kronos_s_ffsd/results.json').read_text())
    top20 = next(b for b in ranking['boundaries'] if b['boundary'] == 'fraud')['knob']['ranking_top20']
    for lookback, tag in ((32, ''), (128, '@128')):
        path = OUT / f'states_L{lookback}.npz'
        if not path.exists():
            log(f'  no states for lookback {lookback}; Kronos blocks{tag} skipped')
            continue
        st = np.load(path)
        full = lambda a: np.concatenate([np.full((st['targets'][0],) + a.shape[1:], np.nan, np.float32), a.astype(np.float32)])
        nllf, H = full(st['nll_feature']), full(st['H'])
        mean, std = np.nanmean(H[reference], axis=0), np.nanstd(H[reference], axis=0) + 1e-6
        Hs = (H - mean) / std
        blocks['B' + tag] = (np.column_stack([nllf.sum(axis=1), nllf]), ['B'] + [f'B_f{i}' for i in range(nllf.shape[1])])
        blocks['K' + tag] = (Hs[:, :, 220], [f'k220_l{i}' for i in range(Hs.shape[1])])
        blocks['K20' + tag] = (np.column_stack([Hs[:, -1, top20], Hs[:, :, top20].mean(axis=1)]),
                               [f'd{d}_last' for d in top20] + [f'd{d}_mean' for d in top20])
        ok = reference[reference >= lookback]
        pca = PCA(16, random_state=SEED).fit(Hs[ok, -1])
        blocks['PCA' + tag] = (np.column_stack([pca.transform(np.nan_to_num(Hs[:, -1]))]), [f'pc{i}' for i in range(16)])
    for D in (32, 128):                       # "day"-frozen reconstruction: scripts/sffsd_block_generation.py --mode frozen
        path = OUT / f'frozen_D{D}.npz'
        if path.exists():
            fz = np.load(path)
            nllf = np.concatenate([np.full((fz['targets'][0], fz['nll_feature'].shape[1]), np.nan, np.float32), fz['nll_feature']])
            blocks[f'F{D}'] = (np.column_stack([nllf.sum(axis=1), nllf]), [f'F{D}'] + [f'F{D}_f{i}' for i in range(nllf.shape[1])])
    for name in list(blocks):
        if name.startswith(('ORD', 'RAW', 'B', 'K')) and not name.startswith('K20'):
            blocks[name + '_ctx'] = context(*blocks[name])
    return blocks, labels, parts


CANDIDATES = [
    ('without Kronos', 'ordinary detectors', ['ORD']),
    ('without Kronos', 'ordinary detectors + raw features', ['ORD', 'RAW']),
    ('without Kronos', 'ordinary detectors + raw features, with context', ['ORD', 'RAW', 'ORD_ctx', 'RAW_ctx']),
    ('Kronos only', 'surprise B (total and per feature)', ['B']),
    ('Kronos only', 'knob K (dimension 220, per layer)', ['K']),
    ('Kronos only', 'knob K with context', ['K', 'K_ctx']),
    ('Kronos only', 'B + K', ['B', 'K']),
    ('Kronos only', 'B + K, with context', ['B', 'K', 'B_ctx', 'K_ctx']),
    ('Kronos only', 'B + twenty knob dimensions', ['B', 'K20']),
    ('Kronos only', 'B + hidden-state PCA', ['B', 'PCA']),
    ('Kronos only', 'B + K + twenty dimensions + PCA, with context', ['B', 'K', 'K20', 'PCA', 'B_ctx', 'K_ctx']),
    ('Kronos only', 'knob K, 128-row context', ['K@128']),
    ('Kronos only', 'knob K with context, 128-row context', ['K@128', 'K@128_ctx']),
    ('Kronos only', 'B + K, 32- and 128-row contexts', ['B', 'K', 'B@128', 'K@128']),
    ('Kronos only', 'B + K with context, 32- and 128-row contexts', ['B', 'K', 'B_ctx', 'K_ctx', 'B@128', 'K@128', 'B@128_ctx', 'K@128_ctx']),
    ('Kronos only', 'day-frozen surprise, 32-row days', ['F32']),
    ('Kronos only', 'day-frozen surprise, 128-row days', ['F128']),
    ('Kronos only', 'B + day-frozen surprise (32 and 128)', ['B', 'F32', 'F128']),
    ('Kronos only', 'B + K + day-frozen surprise', ['B', 'K', 'F32', 'F128']),
    ('Kronos only', 'B + K with context + day-frozen surprise', ['B', 'K', 'B_ctx', 'K_ctx', 'F32', 'F128']),
    ('Kronos + ordinary', 'ordinary + raw + B + K + day-frozen surprise', ['ORD', 'RAW', 'B', 'K', 'F32', 'F128']),
    ('Kronos + ordinary', 'ordinary + raw + B + K with Kronos context + day-frozen', ['ORD', 'RAW', 'B', 'K', 'B_ctx', 'K_ctx', 'F32', 'F128']),
    ('Kronos + ordinary', 'ordinary detectors + B + K', ['ORD', 'B', 'K']),
    ('Kronos + ordinary', 'ordinary detectors + raw + B + K', ['ORD', 'RAW', 'B', 'K']),
    ('Kronos + ordinary', 'ordinary detectors + raw + B + K, Kronos context', ['ORD', 'RAW', 'B', 'K', 'B_ctx', 'K_ctx']),
    ('Kronos + ordinary', 'ordinary detectors + raw + B + K, all context', ['ORD', 'RAW', 'B', 'K', 'ORD_ctx', 'RAW_ctx', 'B_ctx', 'K_ctx']),
    ('Kronos + ordinary', 'ordinary + raw + B + K, 32- and 128-row contexts', ['ORD', 'RAW', 'B', 'K', 'B@128', 'K@128']),
    ('Kronos + ordinary', 'ordinary + raw + B + K with Kronos context, 32 and 128', ['ORD', 'RAW', 'B', 'K', 'B_ctx', 'K_ctx', 'B@128', 'K@128', 'B@128_ctx', 'K@128_ctx']),
    ('Kronos + ordinary', 'everything', ['ORD', 'RAW', 'B', 'K', 'K20', 'PCA', 'ORD_ctx', 'RAW_ctx', 'B_ctx', 'K_ctx']),
]
MODELS = {
    'logistic C=1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
    'logistic C=0.05': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
    'boosting': lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0,
                                                       min_samples_leaf=40, random_state=SEED),
}


def main():
    warnings.filterwarnings('ignore')
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    blocks, labels, parts = build_blocks(log)
    rows = np.arange(len(labels))
    usable = (rows >= START) & np.isin(labels, [0, 1])
    split = {p: np.flatnonzero(usable & (parts == p)) for p in ('reference', 'calibration', 'test')}
    log('labelled rows: ' + ', '.join(f'{p} {len(i)} ({int(labels[i].sum())} fraud)' for p, i in split.items()))
    fit_all = np.r_[split['reference'], split['calibration']]
    half = len(split['calibration']) // 2               # second rule: reference + first half of calibration -> second half
    fit_half, val_half = np.r_[split['reference'], split['calibration'][:half]], split['calibration'][half:]
    log(f'second validation rule: validate on the last {len(val_half)} calibration rows ({int(labels[val_half].sum())} fraud)')
    # The test period holds one long fraud burst: 1,000-row bins where more than half of the labelled rows are fraud.
    # Rows outside it are scored separately, because detecting "we are inside the burst" is a different skill.
    t0 = np.flatnonzero(parts == 'test')[0]
    bins = (split['test'] - t0) // 1000
    share = pd.Series(labels[split['test']]).groupby(bins).mean()
    outside = ~np.isin(bins, share.index[share > 0.5])
    log(f'test burst bins {share.index[share > 0.5].tolist()}: {int(labels[split["test"]][~outside].sum())} of {int(labels[split["test"]].sum())} '
        f'fraud rows; outside the burst {int(outside.sum())} rows, {int(labels[split["test"]][outside].sum())} fraud')
    table = []
    for family, name, keys in CANDIDATES:
        if any(k not in blocks for k in keys):
            continue
        Xc = np.nan_to_num(np.column_stack([blocks[k][0] for k in keys]))
        for model_name, make in MODELS.items():
            val = make().fit(Xc[split['reference']], labels[split['reference']]).predict_proba(Xc[split['calibration']])[:, 1]
            test = make().fit(Xc[fit_all], labels[fit_all]).predict_proba(Xc[split['test']])[:, 1]
            val2 = make().fit(Xc[fit_half], labels[fit_half]).predict_proba(Xc[val_half])[:, 1]
            row = dict(family=family, features=name, model=model_name, n_features=Xc.shape[1],
                       **metrics(labels[split['calibration']], val, 'val_'), **metrics(labels[val_half], val2, 'val2_'),
                       **metrics(labels[split['test']], test, 'test_'), **metrics(labels[split['test']][outside], test[outside], 'outside_'))
            table.append(row)
            pd.DataFrame(table).to_csv(OUT / 'candidates.csv', index=False)
            log(f'  {family:18s} {name:58s} {model_name:16s} val AP {row["val_ap"]:.3f} R@1% {row["val_recall_1pct"]:.3f} '
                f'| test AP {row["test_ap"]:.3f} AUROC {row["test_auroc"]:.3f} R@1% {row["test_recall_1pct"]:.3f} R@5% {row["test_recall_5pct"]:.3f}')
    frame = pd.DataFrame(table)
    chosen = {}
    for rule, column in (('reference -> calibration', 'val_ap'), ('reference + first half of calibration -> second half', 'val2_ap')):
        for family, part in frame.groupby('family', sort=False):
            best = part.sort_values(column, ascending=False).iloc[0]
            chosen[f'{rule} | {family}'] = best.to_dict()
            log(f'selected by [{rule}], {family}: {best.features} / {best.model} ({column} {best[column]:.3f}) -> test AP {best.test_ap:.3f} '
                f'AUROC {best.test_auroc:.3f} R@1% {best.test_recall_1pct:.3f} R@5% {best.test_recall_5pct:.3f}')
    (OUT / 'selected.json').write_text(json.dumps(chosen, indent=1) + '\n')
    (OUT / 'run.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
