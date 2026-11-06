#!/usr/bin/env python3
"""Evaluate the frozen evidence-fusion method on cached real Kronos forecasts.

No new Kronos inference runs here. Sections:
  skill      forecast error of Kronos against naive forecasters
  labelled   same-position metrics with paired bootstrap intervals
  injection  counterfactual power: altering x_t cannot change a forecast made before t
  coverage   naive pipeline on every scorable row, against the published baselines
  native     alarm audit for BRD/BTC
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from fusion_data import (NATIVE, PROXY, PROXY_NAMES, RESULTS, SD, bar_features, load_rows,
                         native_bundle, proxy_bundle, split_calibration, NOT_DETECTION_TARGETS)
from kronos_ad.fusion import EvidenceFusion, FusionConfig, mad, robust_reference, track_threshold
from kronos_ad.joint import JointResidualCalibrator
from kronos_ad.pot import fit_pot
from kronos_ad.proxy import select_targets

OUT = RESULTS / 'kronos_fusion'
FROZEN = json.loads((OUT / 'frozen_config.json').read_text())
ALPHA = FROZEN['alpha']
ONLINE = FROZEN['online_threshold']
PRIMARY_MIN = FROZEN['endpoints']['primary_datasets_min_test_positives']
SEED = 20261106
DELTAS = (2, 3, 4, 6, 8, 12)
INJECTION_SEEDS = (1, 2, 3)


def subset(forecasts, index):
    return {k: v[index] for k, v in forecasts.items()}


def build(bundle, kronos=True, channels=None, naive=None):
    config = FusionConfig(channels=tuple(channels or FROZEN['channels']),
                          naive=tuple(FROZEN['naive_forecasts'] if naive is None else naive),
                          tail_quantile=FROZEN['tail_quantile'], scale_quantile=FROZEN['scale_quantile'],
                          aggregate=FROZEN['aggregate'], alpha=ALPHA)
    forecasts = bundle['forecasts'] if kronos else {}
    fit, validation = bundle['fit'], bundle['validation']
    model = EvidenceFusion(config).fit(bundle['x'][fit], bundle['windows'][fit], subset(forecasts, fit),
                                       bundle['reference'], bundle['global_mask'])
    model.calibrate(bundle['x'][validation], bundle['windows'][validation], subset(forecasts, validation))
    return model, forecasts


def recall_at_fpr(labels, score, fpr=0.01):
    negative, positive = score[labels == 0], score[labels == 1]
    if not len(positive) or not len(negative):
        return np.nan
    return float((positive > np.quantile(negative, 1 - fpr)).mean())


def evaluate(labels, score, flags):
    known = np.isin(labels, [0, 1])
    y, s, f = labels[known].astype(int), score[known], flags[known]
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
    has_both = 0 < y.sum() < len(y)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else np.nan
    return dict(known=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
                ap=float(average_precision_score(y, s)) if has_both else np.nan,
                auroc=float(roc_auc_score(y, s)) if has_both else np.nan,
                recall_1pct_fpr=recall_at_fpr(y, s), alarm_rate=float(flags.mean()),
                fpr=fp / (fp + tn) if fp + tn else np.nan, precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if has_both and precision + recall else (0.0 if has_both else np.nan),
                tp=tp, fp=fp, fn=fn, tn=tn)


def resampling_units(groups, block):
    labels = np.unique(groups)
    return len(labels) if len(labels) > 1 else int(np.ceil(len(groups) / block))


def resamples(groups, rng, block, count):
    """Index arrays: whole groups when grouped, else moving blocks of a single stream."""
    labels, inverse = np.unique(groups, return_inverse=True)
    n = len(groups)
    if len(labels) > 1:
        members = [np.flatnonzero(inverse == i) for i in range(len(labels))]
        for _ in range(count):
            yield np.concatenate([members[i] for i in rng.integers(0, len(labels), len(labels))])
    else:
        for _ in range(count):
            starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
            yield (starts[:, None] + np.arange(block)).ravel()[:n]


def bootstrap(labels, scores, groups, block, reference='fusion', count=2000):
    known = np.isin(labels, [0, 1])
    y, g = labels[known].astype(int), groups[known]
    s = {k: v[known] for k, v in scores.items()}
    rng = np.random.default_rng(SEED)
    samples = {k: [] for k in s}
    for index in resamples(g, rng, block, count):
        yy = y[index]
        if not 0 < yy.sum() < len(yy):
            continue
        for k, v in s.items():
            vv = v[index]
            samples[k].append((average_precision_score(yy, vv), recall_at_fpr(yy, vv)))
    rows = []
    base = np.array(samples[reference])
    for k, v in samples.items():
        v = np.array(v)
        d = base - v
        rows.append(dict(method=k, resamples=len(v), units=resampling_units(g, block), ap_lo=np.quantile(v[:, 0], .025), ap_hi=np.quantile(v[:, 0], .975),
                         r1_lo=np.nanquantile(v[:, 1], .025), r1_hi=np.nanquantile(v[:, 1], .975),
                         ap_diff_lo=np.quantile(d[:, 0], .025), ap_diff_hi=np.quantile(d[:, 0], .975),
                         ap_diff_share_positive=float((d[:, 0] > 0).mean()),
                         r1_diff_lo=np.nanquantile(d[:, 1], .025), r1_diff_hi=np.nanquantile(d[:, 1], .975)))
    return rows


def published(name, test_rows):
    old = pd.read_csv(PROXY / (name + '__detections.csv.gz'))
    old = old[old.split == 'test']
    np.testing.assert_array_equal(old.source_row.to_numpy(), test_rows)
    new = pd.read_csv(RESULTS / f'kronos_tuning/{name}__joint_unsupervised__detections.csv.gz')
    new = new[new.split == 'test']
    np.testing.assert_array_equal(new.source_row.to_numpy(), test_rows)
    return dict(per_feature_pot=(old.max_score_ratio.to_numpy(), old.anomaly.to_numpy(bool)),
                joint_top3=(new.score.to_numpy(), new.anomaly.to_numpy(bool)))


def pot_same_features(b, index):
    """Published per-feature POT recomputed on the detection features only (no calendar/symbol)."""
    raw = b['raw']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in raw['features']])
    if keep.all():
        return None
    ref_std = raw['values'][raw['parts'] == 'reference'].std(axis=0)
    ref_std = np.where(ref_std > 1e-10, ref_std, 1.0)
    eps = np.maximum(ref_std * 1e-6, 1e-8)
    scores = np.square((raw['x'] - raw['forecast']) / (raw['windows'].std(axis=1) + eps))[:, keep]
    thresholds = np.array([fit_pot(scores[raw['calibration'], j], .9, .01, 30)['threshold'] for j in range(scores.shape[1])])
    ratio = (scores[index] / thresholds).max(axis=1)
    return ratio, ratio >= 1


def level_baselines(name, targets_test, calibration_rows=None):
    rows = load_rows(name)
    values, parts = rows['values'], rows['parts']
    train, cal = values[parts == 'reference'], values[parts == 'calibration']
    forest = IsolationForest(n_estimators=100, max_samples=256, contamination='auto', random_state=42, n_jobs=2).fit(train)
    median = np.median(train, axis=0)
    spread = np.median(np.abs(train - median), axis=0) * 1.4826
    spread = np.where(spread > 1e-10, spread, train.std(axis=0))
    spread = np.where(spread > 1e-10, spread, 1.0)
    out = {}
    for label, function in [('isolation_forest', lambda v: -forest.score_samples(v)),
                            ('robust_mad', lambda v: np.abs((v - median) / spread).mean(axis=1))]:
        threshold = fit_pot(function(cal), quantile=.9, risk=.01, min_exceedances=30)['threshold']
        score = function(values[targets_test])
        out[label] = (score, score >= threshold)
    return out


def section_skill(bundles):
    rows = []
    for b in bundles:
        for part, index in [('calibration', np.r_[b['fit'], b['validation']]), ('test', b['test'])]:
            keep = index[b['y'][index] != 1]
            x, w, k = b['x'][keep], b['windows'][keep], b['forecasts']['kronos'][keep]
            ek = np.abs(x - k).mean(axis=0)
            last = np.abs(x - w[:, -1]).mean(axis=0)
            median = np.abs(x - np.median(w, axis=1)).mean(axis=0)
            best = np.minimum(last, median)
            usable = best > 0
            rows.append(dict(dataset=b['name'], part=part, rows=len(keep), dims=x.shape[1],
                             kronos_vs_last=float(np.median((ek / np.where(last > 0, last, np.nan))[last > 0])),
                             kronos_vs_median=float(np.median((ek / np.where(median > 0, median, np.nan))[median > 0])),
                             kronos_vs_best_naive=float(np.median(ek[usable] / best[usable])),
                             share_dims_kronos_beats_best=float((ek[usable] < best[usable]).mean())))
    return pd.DataFrame(rows)


def section_labelled(bundles):
    metrics, intervals, weights = [], [], []
    for b in bundles:
        name, test = b['name'], b['test']
        y, g = b['y'][test], b['groups'][test]
        model, forecasts = build(b)
        plain, _ = build(b, kronos=False)
        score = model.score(b['x'][test], b['windows'][test], subset(forecasts, test))
        score_plain = plain.score(b['x'][test], b['windows'][test], {})
        methods = {'fusion': (score, score > model.threshold_),
                   'fusion_no_kronos': (score_plain, score_plain > plain.threshold_)}
        online, _ = track_threshold(score, model.calibration_scores_, ALPHA, ONLINE['gamma'], ONLINE['window'])
        online_plain, _ = track_threshold(score_plain, plain.calibration_scores_, ALPHA, ONLINE['gamma'], ONLINE['window'])
        methods['fusion_online'] = (score, online)
        methods['fusion_no_kronos_online'] = (score_plain, online_plain)
        if name == NATIVE:
            old = pd.read_csv(RESULTS / 'kronos/detections.csv.gz')
            old = old.iloc[b['offset']:].reset_index(drop=True).iloc[test]
            methods['per_feature_pot'] = (old.max_score_ratio.to_numpy(), old.anomaly.to_numpy(bool))
        else:
            methods.update(published(name, b['raw']['targets'][test]))
            same = pot_same_features(b, test)
            if same is not None:
                methods['per_feature_pot_same_features'] = same
            methods.update(level_baselines(name, b['raw']['targets'][test]))
        for method, (s, f) in methods.items():
            metrics.append(dict(dataset=name, method=method, **evaluate(y, s, f)))
        save_rows(b, model, plain, forecasts, methods)
        weights.append(dict(dataset=name, names=model.forecast_names_, mean_weight=model.weights_.mean(axis=0).tolist(),
                            kronos_weight_by_feature=dict(zip(b['features'], model.weights_[:, -1].tolist())),
                            threshold=model.threshold_))
        positives = int((y == 1).sum())
        if positives >= PRIMARY_MIN:
            ranked = {k: v[0] for k, v in methods.items() if not k.endswith('_online')}
            for row in bootstrap(y, ranked, g, b['block']):
                intervals.append(dict(dataset=name, **row))
        print('labelled', name, 'positives', positives, flush=True)
    return pd.DataFrame(metrics), pd.DataFrame(intervals), weights


def save_rows(b, model, plain, forecasts, methods):
    """Row-level scores for every partition, so thresholds and metrics can be recomputed."""
    parts = []
    for split, index in [('fit', b['fit']), ('validation', b['validation']), ('test', b['test'])]:
        source = b['raw']['targets'][index] if 'raw' in b else index + b['offset']
        frame = pd.DataFrame(dict(position=index, source_row=source, split=split, group=b['groups'][index], label=b['y'][index],
                                  score=model.score(b['x'][index], b['windows'][index], subset(forecasts, index)),
                                  score_no_kronos=plain.score(b['x'][index], b['windows'][index], {})))
        if split == 'test':
            for method, (s, f) in methods.items():
                if method not in ('fusion', 'fusion_no_kronos'):
                    frame[method + '_score'], frame[method + '_alarm'] = s, f
        parts.append(frame)
    rows = pd.concat(parts, ignore_index=True)
    rows['threshold'], rows['threshold_no_kronos'] = model.threshold_, plain.threshold_
    rows.to_csv(OUT / f"{b['name']}__rows.csv.gz", index=False, compression={'method': 'gzip', 'mtime': 0})


def step_scale(b):
    d = b['x'].shape[1]
    steps = np.diff(b['windows'][b['fit']], axis=1).reshape(-1, d)
    scale = mad(steps, 0)
    scale = np.where(scale > 1e-12, scale, steps.std(axis=0))
    return np.where(scale > 1e-12, scale, 1.0)


def proxy_shift(b, index, delta, seed):
    rng = np.random.default_rng(seed)
    d = b['x'].shape[1]
    k = max(1, round(.2 * d))
    scale = step_scale(b)
    shift = np.zeros((len(index), d))
    for row in range(len(index)):
        chosen = rng.choice(d, k, replace=False)
        shift[row, chosen] = rng.choice([-1, 1], k) * delta * scale[chosen]
    return shift


def pump_bar(b, index, delta, seed):
    """Pump-like native bar: close, high, volume and amount all rise together."""
    bars = b['bars'].iloc[index]
    scale = step_scale(b)
    x = b['x'][index].copy()
    x[:, 0] += delta * scale[0]
    x[:, 1] += delta * scale[1]
    x[:, 2] += delta * scale[2]
    x[:, 3] += delta * scale[3]
    return x


def section_injection(bundles):
    rows = []
    forecasters = {'kronos_only': dict(naive=(), kronos=True), 'last_only': dict(naive=('last',), kronos=False),
                   'median_only': dict(naive=('median',), kronos=False),
                   'stacked_with_kronos': dict(naive=('last', 'median'), kronos=True),
                   'stacked_no_kronos': dict(naive=('last', 'median'), kronos=False)}
    for b in bundles:
        test = b['test'][b['y'][b['test']] == 0]
        if b['name'] == NATIVE:
            test = b['test'][b['y'][b['test']] != 1]
        arms = {}
        for label, spec in forecasters.items():
            arms[label] = build(b, kronos=spec['kronos'], channels=('forecast',), naive=spec['naive'])
        arms['fusion'] = build(b)
        arms['fusion_no_kronos'] = build(b, kronos=False)
        for label, (model, forecasts) in arms.items():
            base = model.score(b['x'][test], b['windows'][test], subset(forecasts, test))
            cut = np.quantile(base, .99)
            for delta in DELTAS:
                hits = []
                for seed in INJECTION_SEEDS:
                    x = pump_bar(b, test, delta, seed) if b['name'] == NATIVE else b['x'][test] + proxy_shift(b, test, delta, seed)
                    hits.append(float((model.score(x, b['windows'][test], subset(forecasts, test)) > cut).mean()))
                    if b['name'] == NATIVE:
                        break
                rows.append(dict(dataset=b['name'], method=label, delta=delta, power=float(np.mean(hits)), rows=len(test)))
        rows.extend(original_power(b, test))
        rows.extend(original_power(b, test, same_features=True))
        print('injection', b['name'], flush=True)
    return pd.DataFrame(rows)


def original_power(b, test, same_features=False):
    """The published per-feature POT score under the same counterfactuals."""
    out = []
    if same_features and (b['name'] == NATIVE or all(f not in NOT_DETECTION_TARGETS for f in b['raw']['features'])):
        return out
    label = 'per_feature_pot_same_features' if same_features else 'per_feature_pot'
    if b['name'] == NATIVE:
        names = ['open', 'high', 'low', 'close', 'volume', 'amount']
        r = pd.read_csv(RESULTS / 'kronos/detections.csv.gz')
        thresholds = r[['threshold_' + n for n in names]].iloc[0].to_numpy()
        frame = r.iloc[b['offset']:].reset_index(drop=True)
        bars = frame[names].to_numpy()[test]
        pred = frame[['prediction_' + n for n in names]].to_numpy()[test]
        score = frame[['score_' + n for n in names]].to_numpy()[test]
        sigma = np.abs(bars - pred) / np.sqrt(np.where(score > 0, score, np.nan))
        sigma = np.where(np.isfinite(sigma), sigma, np.nanmedian(sigma, axis=0))
        base = (score / thresholds).max(axis=1)
        cut = np.quantile(base, .99)
        scale = step_scale(b)
        v_unit, a_unit = (lambda e: (np.median(e[e[:, 4] > 0, 4]), np.median(e[e[:, 5] > 0, 5])))(pd.read_csv(RESULTS / 'kronos/detections.csv.gz')[names].to_numpy()[:864])
        for delta in DELTAS:
            shocked = bars.copy()
            close = bars[:, 3] * np.exp(delta * scale[0])
            spread = np.exp(delta * scale[1])
            shocked[:, 3] = close
            shocked[:, 1] = np.maximum(bars[:, 1], close) * spread
            shocked[:, 4] = v_unit * (np.exp(np.log1p(bars[:, 4] / v_unit) + delta * scale[2]) - 1)
            shocked[:, 5] = a_unit * (np.exp(np.log1p(bars[:, 5] / a_unit) + delta * scale[3]) - 1)
            s = np.square((shocked - pred) / (sigma + 1e-8))
            out.append(dict(dataset=b['name'], method='per_feature_pot', delta=delta, power=float(((s / thresholds).max(axis=1) > cut).mean()), rows=len(test)))
        return out
    raw = b['raw']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in raw['features']])
    x, w, f = raw['x'], raw['windows'], raw['forecast']
    ref_std = raw['values'][raw['parts'] == 'reference'].std(axis=0)
    ref_std = np.where(ref_std > 1e-10, ref_std, 1.0)
    eps = np.maximum(ref_std * 1e-6, 1e-8)
    std = w.std(axis=1) + eps
    calibration = raw['calibration']
    scores = np.square((x - f) / std)
    valid, thresholds = [], []
    for j in range(x.shape[1]):
        if same_features and not keep[j]:
            continue
        try:
            thresholds.append(fit_pot(scores[calibration, j], .9, .01, 30)['threshold'])
            valid.append(j)
        except Exception:
            continue
    valid, thresholds = np.array(valid), np.array(thresholds)
    position = np.flatnonzero(keep)
    base = (scores[test][:, valid] / thresholds).max(axis=1)
    cut = np.quantile(base, .99)
    for delta in DELTAS:
        hits = []
        for seed in INJECTION_SEEDS:
            shifted = x[test].copy()
            shifted[:, position] += proxy_shift(b, test, delta, seed)
            s = np.square((shifted - f[test]) / std[test])
            hits.append(float(((s[:, valid] / thresholds).max(axis=1) > cut).mean()))
        out.append(dict(dataset=b['name'], method=label, delta=delta, power=float(np.mean(hits)), rows=len(test)))
    return out


def thin(index, cap=4000):
    return index if len(index) <= cap else index[np.linspace(0, len(index) - 1, cap, dtype=int)]


def section_coverage():
    rows = []
    for name in PROXY_NAMES:
        data = load_rows(name)
        values, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
        keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
        targets, histories, _ = select_targets(parts, groups, 32, 0, 0)
        split = parts[targets]
        cal = np.flatnonzero(split == 'calibration')
        fit, validation = split_calibration(targets[cal], groups[targets[cal]])
        fit, validation = thin(cal[fit]), thin(cal[validation])
        test = np.flatnonzero(split == 'test')
        x = values[targets][:, keep]
        view = lambda index: (x[index], values[histories[index]][:, :, keep])
        config = FusionConfig(channels=tuple(FROZEN['channels']), naive=tuple(FROZEN['naive_forecasts']),
                              tail_quantile=FROZEN['tail_quantile'], scale_quantile=FROZEN['scale_quantile'],
                              aggregate=FROZEN['aggregate'], alpha=ALPHA)
        model = EvidenceFusion(config).fit(*view(fit), {}, values[parts == 'reference'][:, keep],
                                           np.full(int(keep.sum()), name not in SD))
        model.calibrate(*view(validation), {})
        score = np.concatenate([model.score(*view(chunk), {}) for chunk in np.array_split(test, max(1, len(test) // 20000))])
        flags = score > model.threshold_
        online, _ = track_threshold(score, model.calibration_scores_, ALPHA, ONLINE['gamma'], ONLINE['window'])
        y = labels[targets[test]]
        all_test = np.flatnonzero(parts == 'test')
        position = np.searchsorted(all_test, targets[test])
        methods = {'fusion_no_kronos': (score, flags), 'fusion_no_kronos_online': (score, online)}
        for baseline in ['IsolationForest', 'RobustMAD']:
            published_rows = pd.read_csv(RESULTS / f'baselines/{name}__{baseline}__detections.csv.gz')
            np.testing.assert_array_equal(published_rows.label.to_numpy(), labels[all_test])
            methods[baseline] = (published_rows.score.to_numpy()[position], published_rows.anomaly.to_numpy(bool)[position])
        for method, (s, f) in methods.items():
            rows.append(dict(dataset=name, method=method, test_rows_published=len(all_test), test_rows_scorable=len(test),
                             positives_lost_to_warmup=int((labels[all_test] == 1).sum() - (y == 1).sum()), **evaluate(y, s, f)))
        print('coverage', name, len(test), flush=True)
    return pd.DataFrame(rows)


def section_native(b):
    test = b['test']
    model, forecasts = build(b)
    plain, _ = build(b, kronos=False)
    x, w = b['x'][test], b['windows'][test]
    f = subset(forecasts, test)
    score = model.score(x, w, f)
    flags = score > model.threshold_
    online, _ = track_threshold(score, model.calibration_scores_, ALPHA, ONLINE['gamma'], ONLINE['window'])
    feature, channel = model.explain(x, w, f)
    old = pd.read_csv(RESULTS / 'kronos/detections.csv.gz').iloc[b['offset']:].reset_index(drop=True).iloc[test].reset_index(drop=True)
    names = ['open', 'high', 'low', 'close', 'volume', 'amount']
    price = old[['anomaly_' + n for n in names[:4]]].any(axis=1).to_numpy()
    flow = old[['anomaly_' + n for n in names[4:]]].any(axis=1).to_numpy()
    orig = old.anomaly.to_numpy(bool)
    pump = np.flatnonzero(b['y'][test] == 1)[0]
    frame = pd.DataFrame(dict(timestamp=old.timestamps, close=old.close, volume=old.volume, score=score,
                              fusion_alarm=flags, fusion_online_alarm=online, original_alarm=orig,
                              original_price_alarm=price, original_flow_alarm=flow,
                              top_feature=np.array(b['features'])[feature],
                              top_channel=np.array(model.config.channels)[channel], weak_label=b['y'][test]))
    summary = dict(test_bars=len(test), rows_dropped_for_warmup=int(1152 - len(test)),
                   original_alarms=int(orig.sum()), original_price_only=int((price & ~flow).sum()),
                   original_with_flow=int(flow.sum()), fusion_alarms=int(flags.sum()), fusion_online_alarms=int(online.sum()),
                   overlap_original_fusion=int((orig & flags).sum()),
                   zero_volume_share_test=float((old.volume == 0).mean()),
                   original_alarm_on_zero_volume=int((orig & (old.volume == 0).to_numpy()).sum()),
                   pump_bar=str(old.timestamps.iloc[pump]), pump_fusion_alarm=bool(flags[pump]), pump_online_alarm=bool(online[pump]),
                   pump_original_alarm=bool(orig[pump]), pump_rank_fusion=int((score > score[pump]).sum() + 1),
                   pump_rank_original=int((old.max_score_ratio.to_numpy() > old.max_score_ratio.iloc[pump]).sum() + 1),
                   pump_top_feature=str(frame.top_feature.iloc[pump]), pump_top_channel=str(frame.top_channel.iloc[pump]),
                   threshold=model.threshold_, weights=dict(zip(b['features'], np.round(model.weights_, 3).tolist())), forecast_names=model.forecast_names_)
    return frame, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sections', nargs='+', default=['skill', 'labelled', 'injection', 'coverage', 'native'])
    a = p.parse_args()
    warnings.filterwarnings('ignore')
    bundles = None
    if set(a.sections) & {'skill', 'labelled', 'injection'}:
        bundles = [native_bundle()] + [proxy_bundle(n) for n in PROXY_NAMES]
    if 'skill' in a.sections:
        section_skill(bundles).to_csv(OUT / 'forecast_skill.csv', index=False)
    if 'labelled' in a.sections:
        metrics, intervals, weights = section_labelled(bundles)
        metrics.to_csv(OUT / 'metrics.csv', index=False)
        intervals.to_csv(OUT / 'bootstrap.csv', index=False)
        (OUT / 'stack_weights.json').write_text(json.dumps(weights, indent=1) + '\n')
    if 'injection' in a.sections:
        section_injection(bundles).to_csv(OUT / 'injection_power.csv', index=False)
    if 'coverage' in a.sections:
        section_coverage().to_csv(OUT / 'full_coverage.csv', index=False)
    if 'native' in a.sections:
        frame, summary = section_native(native_bundle())
        frame.to_csv(OUT / 'native_rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        (OUT / 'native_summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
