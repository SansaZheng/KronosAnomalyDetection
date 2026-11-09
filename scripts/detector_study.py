#!/usr/bin/env python3
"""Detector comparison with the real Kronos-mini in the forecaster slot (any dataset).

Generalizes `ohlcva_pump_study.py` (whose Kronos slot held the stand-in) to the forecasts
cached by `run_kronos_rows.py`: the real model, loaded offline, scores the same rows.  The
stand-in transformer of the same dataset is scored on the same rows too, so the two
forecasters are paired.

    PYTHONPATH=.:scripts python scripts/detector_study.py --dataset xu_livshits_1h
    PYTHONPATH=.:scripts python scripts/detector_study.py --dataset banksim

Rows: the K-line study rows (calibration 24,000 + test 200,000 + pump hours) for
xu_livshits_1h, the boundary rows (calibration 12,000 + test 24,000 + positives) for the
tabular datasets; calibration = fit / validation halves as before.

Arms (all on the same test rows):
  fusion_r1 / fusion_r2          evidence fusion with {kronos, standin, none} in the forecaster slot
  published_pot_{kronos,standin,last}   the published per-feature POT rule with each forecast
  kronos_nll_surprise, standin_nll_surprise   each model's teacher-forced surprise of the realised row
  isolation_forest, robust_mad   level baselines fitted on reference rows
Metrics: AP, AUROC, R@1% FPR, alarm rate; paired bootstrap (2,000 resamples) over groups or
64-row blocks.  Also forecast skill (MAE per feature) and stacking weights.
Outputs: results/2026-11-10/detector_study_<dataset>/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

import run_fusion
from boundary_pipeline import thin
from boundary_standin import CLIP, StandinForecaster, default_out as standin_out
from fusion_data import NOT_DETECTION_TARGETS, load_rows, long_context, split_calibration
from kronos_ad.fusion import robust_reference
from kronos_ad.pot import fit_pot
from ohlcva_pump_study import published_pot, standin_forecasts
from run_fusion import build, evaluate, recall_at_fpr, resamples, subset
from run_kronos_rows import OUT as ROWS

ROOT = Path(__file__).resolve().parents[1]
REF_CAP, REF_SEED = 60000, 3
SEED = 20261110


def bars_to_features(bar, prev_close, unit_v, unit_a):
    """Kronos's forecast bar -> the four bar features; high/low made consistent as in native_bundle."""
    b = np.array(bar, dtype=float)
    b[:, 1] = b[:, [0, 1, 3]].max(axis=1)
    b[:, 2] = b[:, [0, 2, 3]].min(axis=1)
    tiny = 1e-12
    return np.column_stack([np.log(np.maximum(b[:, 3], tiny) / np.maximum(prev_close, tiny)),
                            np.log(np.maximum(b[:, 1], tiny) / np.maximum(b[:, 2], tiny)),
                            np.log1p(np.maximum(b[:, 4], 0) / unit_v), np.log1p(np.maximum(b[:, 5], 0) / unit_a)])


def bar_spread_to_features(mean, spread, unit_v, unit_a):
    """Delta-method spread of the four features from the spread of the forecast bar."""
    tiny = 1e-12
    return np.column_stack([spread[:, 3] / np.maximum(mean[:, 3], tiny),
                            np.sqrt((spread[:, 1] / np.maximum(mean[:, 1], tiny)) ** 2 + (spread[:, 2] / np.maximum(mean[:, 2], tiny)) ** 2),
                            spread[:, 4] / (unit_v + np.maximum(mean[:, 4], 0)), spread[:, 5] / (unit_a + np.maximum(mean[:, 5], 0))])


def main(dataset, out=None):
    out = Path(out) if out else ROOT / f'results/2026-11-10/detector_study_{dataset}'
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    t0 = time.time()
    data = load_rows(dataset)
    values_all, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
    native = 'bars' in data
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    features = [f for f, k in zip(data['features'], keep) if k]
    values = values_all[:, keep]
    center, scale = robust_reference(values[parts == 'reference'])
    z = np.clip(((values - center) / scale).astype(np.float32), -CLIP, CLIP)
    meta = json.loads((ROWS / dataset / 'meta.json').read_text())
    names = ('study_cal', 'study_test') if native else ('cal', 'test')
    cal, test_c = (np.load(ROWS / dataset / f'{n}.npz') for n in names)
    t_cal, h_cal, t_test, h_test = cal['targets'], cal['histories'], test_c['targets'], test_c['histories']
    ref_all = np.flatnonzero(parts == 'reference')
    t_ref = thin(ref_all, REF_CAP, REF_SEED)
    log(f'{dataset}: rows {len(values)}, features {features}, groups {len(np.unique(groups))}, positives {int((labels == 1).sum())} ({time.time() - t0:.0f}s)')
    log(f'calibration rows {len(t_cal)} (positives {int((labels[t_cal] == 1).sum())}), test rows {len(t_test)} (positives {int((labels[t_test] == 1).sum())}), '
        f'reference rows {len(t_ref)}; Kronos draws {meta["draws"]}, prescale {meta["prescale"]}')

    # ---- forecasts in feature units: real Kronos (cached) and the stand-in (same rows)
    index = np.r_[t_cal, t_test]
    hist = np.r_[h_cal, h_test]
    x, win = values[index], values[hist]
    if native:
        bars, units = data['bars'], data['units']
        prev_close = bars[hist[:, -1], 3]
        fc_bar = np.concatenate([cal['forecast_mean'], test_c['forecast_mean']])
        sp_bar = np.concatenate([cal['forecast_spread'], test_c['forecast_spread']])
        fc_kronos = bars_to_features(fc_bar, prev_close, units['volume'][index], units['amount'][index])
        sp_kronos = bar_spread_to_features(fc_bar, sp_bar, units['volume'][index], units['amount'][index])
    else:
        fc_kronos = np.concatenate([cal['forecast_mean'], test_c['forecast_mean']])
        sp_kronos = np.concatenate([cal['forecast_spread'], test_c['forecast_spread']])
    fc_kronos, sp_kronos = np.nan_to_num(fc_kronos), np.nan_to_num(sp_kronos)
    nll_kronos = np.concatenate([cal['nll'], test_c['nll']])
    standin = StandinForecaster(values.shape[1])
    standin.load_state_dict(torch.load(standin_out(dataset) / 'standin_forecaster.pt', map_location='cpu'))
    standin.eval()
    fc_standin, mean_z, log_var = standin_forecasts(standin, z, hist, center, scale)
    sp_standin = np.exp(0.5 * log_var) * scale
    nll_standin = 0.5 * (log_var + (z[index] - mean_z) ** 2 / np.exp(log_var) + np.log(2 * np.pi)).sum(axis=1)
    log(f'forecasts ready ({time.time() - t0:.0f}s)')

    test = np.arange(len(t_cal), len(index))
    last, median = win[:, -1, :], np.median(win, axis=1)
    skill = []
    for j, f in enumerate(features):
        row = dict(feature=f)
        for name, fc in (('kronos', fc_kronos), ('standin', fc_standin), ('last', last), ('median', median)):
            row[f'mae_{name}'] = float(np.mean(np.abs(x[test, j] - fc[test, j])))
        skill.append(row)
    skill = pd.DataFrame(skill)
    skill.to_csv(out / 'forecast_skill.csv', index=False)
    log('forecast MAE on test rows (kronos / standin / last / median):\n' + skill.round(5).to_string(index=False))

    # ---- bundle
    lc, ls = long_context(values, groups)
    fit_mask, val_mask = split_calibration(t_cal, groups[t_cal])
    fit, validation = np.flatnonzero(fit_mask), np.flatnonzero(val_mask)
    base_forecasts = {'long:center': lc[index], 'long:scale': ls[index]}
    slots = {'kronos': {'kronos': fc_kronos, 'kronos:spread': sp_kronos}, 'standin': {'standin': fc_standin, 'standin:spread': sp_standin}, 'none': {}}
    bundle = dict(name=dataset, x=x, windows=win, fit=fit, validation=validation, test=test, reference=values[t_ref], y=labels[index],
                  groups=groups[index], features=features, global_mask=np.ones(values.shape[1], bool),
                  sequential_mask=np.ones(values.shape[1], bool), block=64)
    log(f'fit rows {len(fit)} (positives {int(labels[t_cal][fit_mask].sum())}), validation rows {len(validation)} (positives {int(labels[t_cal][val_mask].sum())})')
    y_test, g_test = labels[t_test], groups[t_test]
    scores, flags, weights = {}, {}, {}
    for round_name in ('r1', 'r2'):
        run_fusion.use_round(round_name)
        for slot, fc in slots.items():
            name = f'fusion_{round_name}_{slot}' if slot != 'none' else f'fusion_{round_name}_no_forecaster'
            t = time.time()
            b = dict(bundle, forecasts={**base_forecasts, **fc})
            fused, fcs = build(b, kronos=True)
            s = fused.score(x[test], win[test], subset(fcs, test))
            scores[name], flags[name] = s, s > fused.threshold_
            w = np.array(fused.state()['weights'])
            weights[name] = w.tolist()
            log(f'  {name}: threshold {fused.threshold_:.3f}, stack weights {np.round(w, 2).tolist()} ({time.time() - t:.0f}s)')
    eps = np.maximum(values[parts == 'reference'].std(axis=0) * 1e-6, 1e-8)
    cal_rows = np.arange(len(t_cal))
    for name, fc in (('kronos', fc_kronos), ('standin', fc_standin), ('last', last)):
        scores[f'published_pot_{name}'], flags[f'published_pot_{name}'] = published_pot(x, fc, win, cal_rows, test, eps, log)
    for name, nll in (('kronos', nll_kronos), ('standin', nll_standin)):
        s = nll[test]
        scores[f'{name}_nll_surprise'] = s
        flags[f'{name}_nll_surprise'] = s > np.quantile(nll[cal_rows][labels[t_cal] != 1], 0.99)   # calibration 99th percentile of normal rows
    ref = values[t_ref]
    forest = IsolationForest(n_estimators=100, max_samples=256, contamination='auto', random_state=42, n_jobs=2).fit(ref)
    med = np.median(ref, axis=0)
    spread = np.median(np.abs(ref - med), axis=0) * 1.4826
    spread = np.where(spread > 1e-10, spread, ref.std(axis=0))
    spread = np.where(spread > 1e-10, spread, 1.0)
    for name, fn in (('isolation_forest', lambda v: -forest.score_samples(v)), ('robust_mad', lambda v: np.abs((v - med) / spread).mean(axis=1))):
        try:
            threshold = fit_pot(fn(values[t_cal]), quantile=.9, risk=.01, min_exceedances=30)['threshold']
        except ValueError:
            threshold = float(np.quantile(fn(values[t_cal]), 0.99))
        s = fn(x[test])
        scores[name], flags[name] = s, s > threshold

    rows = []
    for name in scores:
        row = dict(detector=name, **evaluate(y_test, scores[name], flags[name]))
        rows.append(row)
        log(f'  {name:28s} AP {row["ap"]:.3f}  AUROC {row["auroc"]:.3f}  R@1% {row["recall_1pct_fpr"]:.3f}  alarm {row["alarm_rate"]:.4f}  '
            f'recall@thr {row["recall"]:.3f}  precision@thr {row["precision"]:.3f}')
    pd.DataFrame(rows).to_csv(out / 'detectors.csv', index=False)

    # ---- paired bootstrap
    rng = np.random.default_rng(SEED)
    known = np.isin(y_test, [0, 1])
    stats = {k: [] for k in scores}
    for idx in resamples(g_test, rng, 64, 2000):
        idx = idx[known[idx]]
        yy = y_test[idx]
        if not 0 < yy.sum() < len(yy):
            continue
        for k, v in scores.items():
            stats[k].append((average_precision_score(yy, v[idx]), recall_at_fpr(yy, v[idx])))
    stats = {k: np.array(v) for k, v in stats.items()}
    boot = []
    units = len(np.unique(g_test)) if len(np.unique(g_test)) > 1 else int(np.ceil(len(g_test) / 64))
    for ref_name in ('fusion_r1_kronos', 'fusion_r1_no_forecaster', 'fusion_r1_standin', 'fusion_r2_kronos', 'robust_mad', 'published_pot_last'):
        for k in scores:
            if k == ref_name:
                continue
            d = stats[k] - stats[ref_name]
            boot.append(dict(reference=ref_name, detector=k, units=units,
                             ap_diff=float(np.nanmean(d[:, 0])), ap_low=float(np.nanquantile(d[:, 0], .025)), ap_high=float(np.nanquantile(d[:, 0], .975)),
                             recall_diff=float(np.nanmean(d[:, 1])), recall_low=float(np.nanquantile(d[:, 1], .025)), recall_high=float(np.nanquantile(d[:, 1], .975))))
    boot = pd.DataFrame(boot)
    boot.to_csv(out / 'bootstrap.csv', index=False)
    for _, r in boot[boot.reference.isin(['fusion_r1_kronos', 'fusion_r1_no_forecaster'])].iterrows():
        log(f'  {r.detector:28s} - {r.reference:24s}: AP [{r.ap_low:+.3f}, {r.ap_high:+.3f}]  R@1% [{r.recall_low:+.3f}, {r.recall_high:+.3f}]')
    per_row = pd.DataFrame({'row': t_test, 'group': g_test, 'label': y_test, **{k: v for k, v in scores.items()}})
    per_row.to_csv(out / 'scores_test.csv.gz', index=False)
    header = dict(dataset=dataset, rows=int(len(values)), representation=meta['representation'], draws=meta['draws'], prescale=meta['prescale'],
                  model_sha256=meta['model_sha256'], tokenizer_sha256=meta['tokenizer_sha256'], calibration_rows=int(len(t_cal)),
                  test_rows=int(len(t_test)), test_positives=int((y_test == 1).sum()), bootstrap_units=units, reference_rows=int(len(t_ref)),
                  standin=str(standin_out(dataset)), seed=SEED, stack_weights=weights, features=features)
    (out / 'results.json').write_text(json.dumps(header, indent=1, default=str) + '\n')
    (out / 'run.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out')
    a = p.parse_args()
    main(a.dataset, a.out)
