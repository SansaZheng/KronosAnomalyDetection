#!/usr/bin/env python3
"""Does the knob lower the miss rate on subtle anomalies?  (stand-in transformer)

The knowledge-boundary run (results/2026-11-07/knowledge_boundary_standin) treated the
knob as an explanatory instrument.  This script tests the detection claim directly:
adding the knob as a scoring channel next to the detector's own score should recover
labelled anomalies that the detector alone misses at the same false-positive budget.

Pre-specified before the run (nothing below was tuned on test rows):

  Baselines      B  teacher-forced NLL surprise of the stand-in (its strongest detector so far)
                 L  level magnitude max_j |z_j| of the target row against the reference robust
                    centre/scale (the Robust-MAD baseline of the repository, one feature set)
  Knob channels  K_label  closed-form knob score of the labelled (pump / fraud) boundary (dims from results.json)
                 K_error  closed-form knob score of the error boundary
  Comparators    logistic and CNN probes of the labelled boundary (is a knob better than any probe?)
  Combination    every channel -> conformal p-value on calibration normal rows that no probe was
                 trained on (null half A); combined statistic = Tippett -log min p (primary) and
                 Fisher -2 sum log p (secondary)
  Operating pts  (i) 1% FPR on the test normal rows, the metric used throughout the repository;
                 (ii) deployable: alpha = 0.01 conformal threshold set on null half B
  Miss rate      1 - recall over labelled anomaly rows (64 pump rows in the sapienza_15s test sample)
  Subtle pumps   (a) relative: pumps that B misses at operating point (i);
                 (b) intrinsic: pumps whose level magnitude L is below the pump median
  Primary test   miss rate(B + K_label, Tippett) < miss rate(B) at operating point (i).
                 Paired bootstrap over pump groups (2000 resamples, seed 20261108) of the recall
                 difference with the threshold held fixed, and exact paired counts (detected by
                 one arm only) for a McNemar view.

Outputs go to results/2026-11-08/knowledge_boundary_missrate/ (sapienza_15s) or
results/2026-11-08/knowledge_boundary_missrate_<dataset>/ for the other transaction datasets,
whose stand-ins come from `boundary_standin.py --dataset <name>`.  On the single-stream
fraud datasets the labelled boundary is `fraud` and the bootstrap units are contiguous
blocks of positive rows (about 50 blocks) instead of pump groups.

Revision 1 (after the first sapienza_15s run, before any other dataset): (a) the level
magnitude L is computed from the unclipped z-scores -- the clip at |z| = 5 tied half of the
pumps and most extreme normals at exactly 5.0; (b) p-values use kronos_ad.fusion.TailPValue
(conformal below the 0.9 quantile, GPD tail above) instead of the plain conformal rank --
the test NLL distribution is shifted up relative to calibration, so ~12% of test rows sat
beyond every calibration null score and tied at the minimum p, which made the Tippett
statistic degenerate; (c) an assumption-free "OR at alpha/2 per channel" arm is added,
which splits the false-positive budget equally between the two channels in rank space and
does not depend on any calibration distribution.  The first run is kept as run_A.log.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from boundary_pipeline import batched, thin
from boundary_standin import (CLIP, DATASET, LOOKBACK, ROW_CAP, SEED, StandinForecaster, StandinStates, default_out,
                              group_windows, label_boundary)
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.boundary import CNNProbe, KnobPredictor, LogisticProbe, balanced_accuracy, split_balanced
from kronos_ad.fusion import TailPValue, robust_reference

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.01
RESAMPLES = 2000
BOOT_SEED = 20261108


def conformal_logp(scores, null):
    """log p-value against the null scores: conformal rank below the 0.9 quantile, GPD tail above."""
    return TailPValue(null)(np.asarray(scores, dtype=float))


def tippett(logp):
    return -np.min(logp, axis=0)


def fisher(logp):
    return -2.0 * np.sum(logp, axis=0)


def threshold_at_fpr(score, normal, fpr=ALPHA):
    return float(np.quantile(score[normal], 1 - fpr))


def bootstrap_units(groups, rows):
    """Pump groups when there are enough of them, otherwise ~50 contiguous blocks of positive rows."""
    groups = np.asarray(groups)
    if len(np.unique(groups)) >= 10:
        return groups
    order = np.argsort(rows)
    block = max(1, int(np.ceil(len(rows) / 50)))
    units = np.empty(len(rows), dtype=object)
    units[order] = [f'block{i // block}' for i in range(len(rows))]
    return units


def paired_bootstrap(detected_a, detected_b, groups, resamples=RESAMPLES, seed=BOOT_SEED):
    """95% interval of recall(b) - recall(a) over positive rows, resampling whole units."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    diffs = np.empty(resamples)
    for i in range(resamples):
        pick = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([members[j] for j in pick])
        diffs[i] = detected_b[rows].mean() - detected_a[rows].mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), int(len(unique))


def main(dataset=DATASET, standin=None, out=None):
    warnings.filterwarnings('ignore')
    STANDIN = Path(standin) if standin else default_out(dataset)
    OUT = Path(out) if out else ROOT / ('results/2026-11-08/knowledge_boundary_missrate' + ('' if dataset == DATASET else f'_{dataset}'))
    OUT.mkdir(parents=True, exist_ok=True)
    positive_name = label_boundary(dataset)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    recorded = json.loads((STANDIN / 'results.json').read_text())
    knob_dims = {b['boundary']: b['knob']['dims'] for b in recorded['boundaries']}
    log(f'knob dimensions from the frozen run: {knob_dims}')

    # ---- data, identical to boundary_standin.py (same seeds, same thinning)
    torch.manual_seed(SEED)
    data = load_rows(dataset)
    values, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    z_raw = ((values[:, keep] - center) / scale).astype(np.float32)
    z = np.clip(z_raw, -CLIP, CLIP)
    windows = group_windows(values, groups, parts, LOOKBACK)

    def tensors(part, pick):
        targets, histories = windows[part]
        pick = np.union1d(pick, np.flatnonzero(labels[targets] == 1))
        return torch.tensor(z[histories[pick]]), torch.tensor(z[targets[pick]]), targets[pick]

    def sample(part, cap, seed):
        return thin(np.arange(len(windows[part][0])), cap, seed)

    x_cal, y_cal, t_cal = tensors('calibration', sample('calibration', ROW_CAP // 2, 2))
    x_test, y_test, t_test = tensors('test', sample('test', ROW_CAP, 3))
    x_pool, y_pool, t_pool = tensors('reference', sample('reference', ROW_CAP // 2, 4))
    x_pool, y_pool, t_pool = torch.cat([x_pool, x_cal]), torch.cat([y_pool, y_cal]), np.r_[t_pool, t_cal]
    model = StandinForecaster(x_test.shape[2])
    model.load_state_dict(torch.load(STANDIN / 'standin_forecaster.pt', map_location='cpu'))
    states = StandinStates(model)
    H_pool, H_test = batched(states.hidden_states, x_pool), batched(states.hidden_states, x_test)
    nll_pool, nll_test = -batched(states.loglik, x_pool, y_pool), -batched(states.loglik, x_test, y_test)
    calibration_rows = np.arange(len(t_pool) - len(t_cal), len(t_pool))
    threshold = float(np.quantile(nll_pool[calibration_rows][labels[t_cal] != 1], 0.9))
    assert abs(threshold - recorded['error_threshold']) < 1e-6, (threshold, recorded['error_threshold'])
    label_test = labels[t_test].astype(int)
    known = np.isin(label_test, [0, 1])          # S-FFSD label 2 = unknown, excluded from every metric
    normal = label_test == 0
    pumps = label_test == 1
    pump_test = np.where(pumps, 1, 0)
    pump_groups = bootstrap_units(groups[t_test][pumps], t_test[pumps])
    log(f'test rows {len(t_test)} ({int(known.sum())} with known label): {int(pumps.sum())} {positive_name} rows in '
        f'{len(np.unique(pump_groups))} bootstrap units; error threshold {threshold:.3f} reproduced')

    # ---- probes and knobs, refit with the frozen run's seeds and dimensions
    boundaries = {'error': ((nll_pool > threshold).astype(int), (nll_test > threshold).astype(int)),
                  positive_name: (labels[t_pool].astype(int), label_test)}
    channels = {'B_nll_surprise': nll_test, 'L_level_magnitude': np.abs(z_raw[t_test]).max(axis=1)}
    trained = np.zeros(len(t_pool), bool)
    for name, (y_pool_b, y_test_b) in boundaries.items():
        train_idx, _ = split_balanced(y_pool_b, test_fraction=0.0, seed=SEED)
        test_idx, _ = split_balanced(y_test_b, test_fraction=0.0, seed=SEED + 1)
        trained[train_idx] = True
        t = time.time()
        knob = KnobPredictor(epochs=400, seed=SEED).fit(H_pool[train_idx], y_pool_b[train_idx], np.asarray(knob_dims[name]))
        acc = balanced_accuracy(knob, H_test[test_idx], y_test_b[test_idx])
        log(f'  {name}: knob k={len(knob_dims[name])} dims={knob_dims[name]} balanced accuracy {acc:.4f} '
            f'(frozen run {next(b for b in recorded["boundaries"] if b["boundary"] == name)["knob"]["distilled_accuracy"]:.4f}; {time.time() - t:.0f}s)')
        channels['K_error_knob' if name == 'error' else 'K_label_knob'] = knob.score(H_test)
        if name == positive_name:
            for key, probe in (('logistic', LogisticProbe()), ('cnn', CNNProbe(epochs=80))):
                t = time.time()
                probe.fit(H_pool[train_idx], y_pool_b[train_idx])
                log(f'  {name}: {key} probe balanced accuracy {balanced_accuracy(probe, H_test[test_idx], y_test_b[test_idx]):.4f} ({time.time() - t:.0f}s)')
                channels[f'P_label_{key}'] = probe.score(H_test)

    # ---- null rows: calibration normal rows that trained nothing, split into two halves
    null_rows = calibration_rows[(labels[t_cal] != 1) & ~trained[calibration_rows]]
    rng = np.random.default_rng(BOOT_SEED)
    rng.shuffle(null_rows)
    null_a, null_b = np.sort(null_rows[: len(null_rows) // 2]), np.sort(null_rows[len(null_rows) // 2:])
    log(f'null rows for calibration: {len(null_a)} (p-values) + {len(null_b)} (combined thresholds)')
    pool_scores = {'B_nll_surprise': nll_pool, 'L_level_magnitude': np.abs(z_raw[t_pool]).max(axis=1)}
    # hidden-state channels on the pool rows
    for name in boundaries:
        train_idx, _ = split_balanced(boundaries[name][0], test_fraction=0.0, seed=SEED)
        knob = KnobPredictor(epochs=400, seed=SEED).fit(H_pool[train_idx], boundaries[name][0][train_idx], np.asarray(knob_dims[name]))
        pool_scores['K_error_knob' if name == 'error' else 'K_label_knob'] = knob.score(H_pool)
        if name == positive_name:
            for key, probe in (('logistic', LogisticProbe()), ('cnn', CNNProbe(epochs=80))):
                probe.fit(H_pool[train_idx], boundaries[name][0][train_idx])
                pool_scores[f'P_label_{key}'] = probe.score(H_pool)
    logp_test = {k: conformal_logp(v, pool_scores[k][null_a]) for k, v in channels.items()}
    logp_null_b = {k: conformal_logp(pool_scores[k][null_b], pool_scores[k][null_a]) for k in channels}

    # ---- arms
    arms = {k: (v, pool_scores[k][null_b]) for k, v in channels.items()}
    combos = {'B+K_label': ('B_nll_surprise', 'K_label_knob'), 'B+K_error': ('B_nll_surprise', 'K_error_knob'),
              'B+P_logistic': ('B_nll_surprise', 'P_label_logistic'), 'B+P_cnn': ('B_nll_surprise', 'P_label_cnn'),
              'L+K_label': ('L_level_magnitude', 'K_label_knob'), 'B+L': ('B_nll_surprise', 'L_level_magnitude'),
              'B+L+K_label': ('B_nll_surprise', 'L_level_magnitude', 'K_label_knob')}
    for combo, keys in combos.items():
        for stat, fn in (('tippett', tippett), ('fisher', fisher)):
            arms[f'{combo} [{stat}]'] = (fn(np.stack([logp_test[k] for k in keys])), fn(np.stack([logp_null_b[k] for k in keys])))
    # assumption-free OR: each channel gets alpha / m of the false-positive budget, thresholds in rank space
    or_arms = {}
    for combo, keys in combos.items():
        m = len(keys)
        flags = np.zeros(len(t_test), bool)
        for k in keys:
            flags |= channels[k] > threshold_at_fpr(channels[k], normal, ALPHA / m)
        or_arms[f'{combo} [or@a/{m}]'] = flags

    base_threshold = threshold_at_fpr(channels['B_nll_surprise'], normal)
    detected_base = channels['B_nll_surprise'] > base_threshold
    subtle_relative = pumps & ~detected_base
    level = channels['L_level_magnitude']
    subtle_intrinsic = pumps & (level < np.median(level[pumps]))
    log(f'baseline B at 1% FPR: recall {detected_base[pumps].mean():.3f}; subtle pumps: relative {int(subtle_relative.sum())}, '
        f'intrinsic (below pump-median level magnitude {np.median(level[pumps]):.2f}) {int(subtle_intrinsic.sum())}')

    rows, pairs = [], []
    evaluations = [(arm, score, null_score, None) for arm, (score, null_score) in arms.items()]
    evaluations += [(arm, None, None, flags) for arm, flags in or_arms.items()]
    for arm, score, null_score, flags in evaluations:
        if flags is None:
            thr = threshold_at_fpr(score, normal)
            det = score > thr
            deploy_thr = float(np.quantile(null_score, 1 - ALPHA))
            det_deploy = score > deploy_thr
            ap, auroc = float(average_precision_score(pump_test[known], score[known])), float(roc_auc_score(pump_test[known], score[known]))
        else:
            det, det_deploy, ap, auroc = flags, flags, np.nan, np.nan
        row = dict(arm=arm, ap=ap, auroc=auroc, fpr=float(det[normal].mean()),
                   recall_1pct_fpr=float(det[pumps].mean()), miss_rate_1pct_fpr=float(1 - det[pumps].mean()),
                   recall_subtle_relative=float(det[subtle_relative].mean()) if subtle_relative.any() else np.nan,
                   recall_subtle_intrinsic=float(det[subtle_intrinsic].mean()),
                   recall_obvious_intrinsic=float(det[pumps & ~subtle_intrinsic].mean()),
                   deploy_alarm_rate=float(det_deploy[normal].mean()), deploy_recall=float(det_deploy[pumps].mean()),
                   deploy_recall_subtle_intrinsic=float(det_deploy[subtle_intrinsic].mean()))
        lo, hi, units = paired_bootstrap(detected_base[pumps], det[pumps], pump_groups)
        only_arm, only_base = int((det & ~detected_base)[pumps].sum()), int((detected_base & ~det)[pumps].sum())
        row.update(recall_diff_vs_B=float(det[pumps].mean() - detected_base[pumps].mean()), ci_low=lo, ci_high=hi, units=units,
                   pumps_gained_vs_B=only_arm, pumps_lost_vs_B=only_base,
                   p_mcnemar=float(binomtest(only_arm, only_arm + only_base).pvalue) if only_arm + only_base else 1.0)
        rows.append(row)
        log(f'  {arm:34s} AP {row["ap"]:.3f}  R@1% {row["recall_1pct_fpr"]:.3f}  miss {row["miss_rate_1pct_fpr"]:.3f}  '
            f'subtle rel {row["recall_subtle_relative"]:.3f} intr {row["recall_subtle_intrinsic"]:.3f}  '
            f'vs B {row["recall_diff_vs_B"]:+.3f} [{lo:+.3f}, {hi:+.3f}] gained {only_arm} lost {only_base}  '
            f'deploy alarm {row["deploy_alarm_rate"]:.4f} recall {row["deploy_recall"]:.3f}')
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'missrate.csv', index=False)

    # ---- complementarity: do hidden-state channels rank the pumps that B misses?
    comp = []
    for key, score in channels.items():
        thr = threshold_at_fpr(score, normal)
        comp.append(dict(channel=key, recall_all=float((score > thr)[pumps].mean()),
                         recall_B_missed=float((score > thr)[subtle_relative].mean()) if subtle_relative.any() else np.nan,
                         recall_B_detected=float((score > thr)[pumps & detected_base].mean()),
                         auroc_subtle_intrinsic_vs_normal=float(roc_auc_score(np.r_[np.zeros(normal.sum()), np.ones(subtle_intrinsic.sum())],
                                                                              np.r_[score[normal], score[subtle_intrinsic]])),
                         spearman_with_B_on_pumps=float(pd.Series(score[pumps]).corr(pd.Series(channels['B_nll_surprise'][pumps]), method='spearman'))))
    pd.DataFrame(comp).to_csv(OUT / 'complementarity.csv', index=False)
    per_row = pd.DataFrame({'row': t_test, 'group': groups[t_test], 'pump': pump_test, 'subtle_intrinsic': subtle_intrinsic.astype(int),
                            'missed_by_B': subtle_relative.astype(int), **{k: v for k, v in channels.items()}})
    per_row.to_csv(OUT / 'scores_test.csv.gz', index=False)
    header = dict(dataset=dataset, label_boundary=positive_name, test_rows=int(len(t_test)), known_rows=int(known.sum()), positives=int(pumps.sum()),
                  bootstrap_units=int(len(np.unique(pump_groups))),
                  error_threshold=threshold, knob_dims=knob_dims, alpha=ALPHA, resamples=RESAMPLES, boot_seed=BOOT_SEED,
                  null_rows=dict(pvalues=int(len(null_a)), thresholds=int(len(null_b))),
                  baseline_B_threshold_1pct_fpr=base_threshold, subtle_relative=int(subtle_relative.sum()),
                  subtle_intrinsic=int(subtle_intrinsic.sum()), pump_level_median=float(np.median(level[pumps])))
    (OUT / 'results.json').write_text(json.dumps(header, indent=1) + '\n')
    (OUT / 'run.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=DATASET)
    parser.add_argument('--standin', help='directory of the stand-in run (default: that dataset\'s boundary_standin output)')
    parser.add_argument('--out')
    args = parser.parse_args()
    main(args.dataset, args.standin, args.out)
