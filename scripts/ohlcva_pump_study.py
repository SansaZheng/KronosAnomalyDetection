#!/usr/bin/env python3
"""KronosAnomalyDetection on hourly OHLCVA K-lines with pump labels (Xu & Livshits 2019 + B0R0koko).

This is the first dataset of the study in Kronos's own domain (OHLCVA bars).  Kronos weights
are still unreachable from the evaluation container, so the Kronos slot of every forecast-based
arm is filled by the stand-in transformer trained on the reference candles by
`boundary_standin.py --dataset xu_livshits_1h`; nothing below is a statement about Kronos.

Rows: `fusion_data.load_rows('xu_livshits_1h')` -- 921 coin:exchange series, bar features
(log return, log range, log1p volume / amount in series units), time split reference /
calibration / test, labels = pump hours.  Calibration rows are thinned to 24,000 + all pump
hours (fit / validation = earlier / later series halves, as in the proxy protocol); test rows
to 200,000 + all pump hours (uniform thinning changes prevalence, not the ranking comparison).

Arms (all scored on the same test rows):
  published_pot_standin   the published scoring structure: squared per-feature error of the
                          stand-in forecast / window std, POT threshold per feature, OR
  published_pot_last      the same with the last value as the forecast (no model at all)
  fusion_r1 / fusion_r1_no_forecaster     evidence fusion, round-1 config, with / without the stand-in
  fusion_r2 / fusion_r2_no_forecaster     round-2 config (long channel + multi-statistic)
  standin_nll_surprise    the stand-in's own likelihood surprise of the realised bar
  isolation_forest, robust_mad            level baselines fitted on reference rows
Metrics: AP, AUROC, R@1% FPR, alarm rate; paired bootstrap over series (2,000 resamples).
Outputs: results/2026-11-10/ohlcva_xu_livshits/.
"""
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
from boundary_standin import CLIP, LOOKBACK, StandinForecaster, default_out, group_windows
from fusion_data import load_rows, long_context, split_calibration
from kronos_ad.fusion import robust_reference
from kronos_ad.pot import fit_pot
from run_fusion import build, evaluate, recall_at_fpr, resamples, subset

DATASET = 'xu_livshits_1h'
OUT = Path(__file__).resolve().parents[1] / 'results/2026-11-10/ohlcva_xu_livshits'
CAL_CAP, TEST_CAP, REF_CAP = 24000, 200000, 60000
SEED = 20261110


def standin_forecasts(model, z, histories, center, scale, batch=1024):
    """Mean forecast of the next row in feature space, and the Gaussian NLL of the realised row."""
    means, log_vars = [], []
    with torch.no_grad():
        for start in range(0, len(histories), batch):
            x = torch.tensor(z[histories[start:start + batch]])
            m, lv = model(x)                       # per-position heads; the last position forecasts the next row
            means.append(m[:, -1].numpy()), log_vars.append(lv[:, -1].numpy())
    mean, log_var = np.concatenate(means), np.concatenate(log_vars)
    return mean * scale + center, mean, log_var


def published_pot(x, forecast, windows, calibration, test, eps, log=print):
    """The published per-feature POT rule.  On these candles most hours have zero volume, so a feature's
    error score ties at zero on more than 99% of calibration rows and the GPD tail cannot be fitted at
    risk 1%; such a feature falls back to the empirical 99% quantile of its calibration scores."""
    scores = np.square((x - forecast) / (windows.std(axis=1) + eps))
    thresholds = []
    for j in range(x.shape[1]):
        try:
            thresholds.append(fit_pot(scores[calibration, j], .9, .01, 30)['threshold'])
        except ValueError as error:
            thresholds.append(float(np.quantile(scores[calibration, j], 0.99)))
            log(f'    POT feature {j}: {error}; empirical 99% quantile used')
    thresholds = np.maximum(np.array(thresholds), 1e-12)
    ratio = (scores[test] / thresholds).max(axis=1)
    return ratio, ratio >= 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    t0 = time.time()
    data = load_rows(DATASET)
    values, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
    center, scale = robust_reference(values[parts == 'reference'])
    z = np.clip(((values - center) / scale).astype(np.float32), -CLIP, CLIP)
    windows = group_windows(values, groups, parts, LOOKBACK)
    log(f'rows {len(values)}, series {len(np.unique(groups))}, pump hours {int(labels.sum())} ({time.time() - t0:.0f}s)')

    def sample(part, cap, seed):
        targets, histories = windows[part]
        pick = np.union1d(thin(np.arange(len(targets)), cap, seed), np.flatnonzero(labels[targets] == 1))
        return targets[pick], histories[pick]
    t_cal, h_cal = sample('calibration', CAL_CAP, 1)
    t_test, h_test = sample('test', TEST_CAP, 2)
    t_ref, _ = sample('reference', REF_CAP, 3)
    log(f'calibration rows {len(t_cal)} (pumps {int(labels[t_cal].sum())}), test rows {len(t_test)} (pumps {int(labels[t_test].sum())}, '
        f'series with pumps {len(np.unique(groups[t_test][labels[t_test] == 1]))}), reference rows {len(t_ref)}')

    model = StandinForecaster(values.shape[1])
    model.load_state_dict(torch.load(default_out(DATASET) / 'standin_forecaster.pt', map_location='cpu'))
    model.eval()
    fc_cal, _, _ = standin_forecasts(model, z, h_cal, center, scale)
    fc_test, mean_test, log_var_test = standin_forecasts(model, z, h_test, center, scale)
    nll_test = 0.5 * (log_var_test + (z[t_test] - mean_test) ** 2 / np.exp(log_var_test) + np.log(2 * np.pi)).sum(axis=1)
    log(f'stand-in forecasts computed ({time.time() - t0:.0f}s)')

    # ---- bundle in the run_fusion format: calibration rows first (fit / validation), then test rows
    index = np.r_[t_cal, t_test]
    hist = np.r_[h_cal, h_test]
    x, win = values[index], values[hist]
    forecasts = {'kronos': np.r_[fc_cal, fc_test]}
    lc, ls = long_context(values, groups)
    forecasts['long:center'], forecasts['long:scale'] = lc[index], ls[index]
    fit_mask, val_mask = split_calibration(t_cal, groups[t_cal])
    fit, validation = np.flatnonzero(fit_mask), np.flatnonzero(val_mask)
    test = np.arange(len(t_cal), len(index))
    bundle = dict(name=DATASET, x=x, windows=win, forecasts=forecasts, fit=fit, validation=validation, test=test,
                  reference=values[t_ref], y=labels[index], groups=groups[index], features=data['features'],
                  global_mask=np.ones(values.shape[1], bool), sequential_mask=np.ones(values.shape[1], bool), block=64)
    log(f'fit rows {len(fit)} (pumps {int(labels[t_cal][fit_mask].sum())}), validation rows {len(validation)} '
        f'(pumps {int(labels[t_cal][val_mask].sum())}); long-context rows without history: {int(np.isnan(lc[index][:, 0]).sum())}')

    y_test, g_test = labels[t_test], groups[t_test]
    scores, flags = {}, {}
    for round_name in ('r1', 'r2'):
        run_fusion.use_round(round_name)
        for kronos in (True, False):
            name = f'fusion_{round_name}' + ('' if kronos else '_no_forecaster')
            t = time.time()
            fused, fc = build(bundle, kronos=kronos)
            s = fused.score(x[test], win[test], subset(fc, test))
            scores[name], flags[name] = s, s > fused.threshold_
            state = fused.state()
            log(f'  {name}: threshold {fused.threshold_:.3f}, stack weights {np.round(np.array(state["weights"]), 2).tolist()} ({time.time() - t:.0f}s)')
    eps = np.maximum(values[parts == 'reference'].std(axis=0) * 1e-6, 1e-8)
    cal_rows = np.arange(len(t_cal))
    last = win[:, -1, :]
    scores['published_pot_standin'], flags['published_pot_standin'] = published_pot(x, forecasts['kronos'], win, cal_rows, test, eps, log)
    scores['published_pot_last'], flags['published_pot_last'] = published_pot(x, last, win, cal_rows, test, eps, log)
    scores['standin_nll_surprise'] = nll_test
    flags['standin_nll_surprise'] = nll_test > np.quantile(nll_test, 0.99)   # no calibration threshold: alarm rate fixed at 1% by construction
    ref = values[parts == 'reference']
    forest = IsolationForest(n_estimators=100, max_samples=256, contamination='auto', random_state=42, n_jobs=2).fit(ref)
    median = np.median(ref, axis=0)
    spread = np.median(np.abs(ref - median), axis=0) * 1.4826
    spread = np.where(spread > 1e-10, ref.std(axis=0), spread)
    spread = np.where(spread > 1e-10, spread, 1.0)
    for name, fn in (('isolation_forest', lambda v: -forest.score_samples(v)), ('robust_mad', lambda v: np.abs((v - median) / spread).mean(axis=1))):
        threshold = fit_pot(fn(values[t_cal]), quantile=.9, risk=.01, min_exceedances=30)['threshold']
        s = fn(x[test])
        scores[name], flags[name] = s, s > threshold

    rows = []
    for name in scores:
        row = dict(detector=name, **evaluate(y_test, scores[name], flags[name]))
        rows.append(row)
        log(f'  {name:28s} AP {row["ap"]:.3f}  AUROC {row["auroc"]:.3f}  R@1% {row["recall_1pct_fpr"]:.3f}  alarm {row["alarm_rate"]:.4f}  '
            f'recall@thr {row["recall"]:.3f}  precision@thr {row["precision"]:.3f}')
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'detectors.csv', index=False)

    # ---- paired bootstrap over series: AP and R@1% differences against fusion_r1 and against robust_mad
    rng = np.random.default_rng(SEED)
    stats = {k: [] for k in scores}
    for idx in resamples(g_test, rng, 64, 2000):
        yy = y_test[idx]
        if not 0 < yy.sum() < len(yy):
            continue
        for k, v in scores.items():
            stats[k].append((average_precision_score(yy, v[idx]), recall_at_fpr(yy, v[idx])))
    stats = {k: np.array(v) for k, v in stats.items()}
    boot = []
    for ref_name in ('fusion_r1', 'fusion_r1_no_forecaster', 'robust_mad', 'published_pot_last'):
        for k in scores:
            if k == ref_name:
                continue
            d = stats[k] - stats[ref_name]
            boot.append(dict(reference=ref_name, detector=k, units=len(np.unique(g_test)),
                             ap_diff=float(np.nanmean(d[:, 0])), ap_low=float(np.nanquantile(d[:, 0], .025)), ap_high=float(np.nanquantile(d[:, 0], .975)),
                             recall_diff=float(np.nanmean(d[:, 1])), recall_low=float(np.nanquantile(d[:, 1], .025)), recall_high=float(np.nanquantile(d[:, 1], .975))))
    boot = pd.DataFrame(boot)
    boot.to_csv(OUT / 'bootstrap.csv', index=False)
    for _, r in boot[boot.reference.isin(['fusion_r1', 'robust_mad'])].iterrows():
        log(f'  {r.detector:28s} - {r.reference:14s}: AP [{r.ap_low:+.3f}, {r.ap_high:+.3f}]  R@1% [{r.recall_low:+.3f}, {r.recall_high:+.3f}]')
    per_row = pd.DataFrame({'row': t_test, 'series': g_test, 'exchange': data['exchange'][t_test], 'timestamp': data['timestamps'][t_test],
                            'pump': y_test, **{k: v for k, v in scores.items()}})
    per_row.to_csv(OUT / 'scores_test.csv.gz', index=False)
    header = dict(dataset=DATASET, rows=int(len(values)), series=int(len(np.unique(groups))), pump_hours=int(labels.sum()),
                  b0r0koko_hours_added=int(data['b0r0koko_hours_added']), splits=data['protocol'], calibration_rows=int(len(t_cal)),
                  test_rows=int(len(t_test)), test_pumps=int(y_test.sum()), test_series_with_pumps=int(len(np.unique(g_test[y_test == 1]))),
                  reference_rows=int(len(t_ref)), standin=str(default_out(DATASET)), seed=SEED, kronos_weights='unreachable (Hugging Face blocked); stand-in used')
    (OUT / 'results.json').write_text(json.dumps(header, indent=1, default=str) + '\n')
    (OUT / 'run.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
