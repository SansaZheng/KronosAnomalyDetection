#!/usr/bin/env python3
"""Real Kronos-mini inference on the experiment rows, cached for every later analysis.

For one dataset this computes, on exactly the thinned row sets of the stand-in protocol
(`boundary_standin.py`: reference pool 12,000 + calibration 12,000 + test 24,000 rows, same
seeds, every labelled positive kept), the model's hidden states at the last context bar,
its teacher-forced NLL of the realised next bar, and S sampled one-step forecasts
(mean and spread, in the dataset's own feature units) -- see `kronos_states.py` for the
two row representations (native K-lines, published proxy for tabular rows).

    PYTHONPATH=.:scripts python scripts/run_kronos_rows.py --dataset banksim
    PYTHONPATH=.:scripts python scripts/run_kronos_rows.py --dataset xu_livshits_1h --study

`--study` instead computes forecasts and NLL on the K-line detector study's own rows
(`ohlcva_pump_study.py`: calibration 24,000 and test 200,000 + pump hours) with more draws.

Outputs: results/2026-11-10/kronos_rows/<dataset>/{ref,cal,test}.npz (or study_{cal,test}.npz)
and meta.json.  Sets already on disk are skipped, so an interrupted run resumes.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from boundary_pipeline import thin
from boundary_standin import CAL_SEED, LOOKBACK, REF_SEED, ROW_CAP, TEST_SEED, group_windows
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.fusion import robust_reference
from kronos_states import (COLUMNS, MODEL_DIR, TOKENIZER_DIR, KronosStates, denormalize, load_kronos, native_sequences,
                           proxy_pool, proxy_sequences, self_test)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/2026-11-10/kronos_rows'
DRAWS, STUDY_DRAWS = 5, 10
STUDY_CAL_CAP, STUDY_TEST_CAP = 24000, 200000
SEQUENCES_PER_BATCH = 1024


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_sets(windows, labels, study=False):
    """(name, targets, histories) of each row set; positives are always kept."""
    def pick(part, cap, seed):
        targets, histories = windows[part]
        idx = np.union1d(thin(np.arange(len(targets)), cap, seed), np.flatnonzero(labels[targets] == 1))
        return targets[idx], histories[idx]
    if study:
        return [('study_cal',) + pick('calibration', STUDY_CAL_CAP, 1), ('study_test',) + pick('test', STUDY_TEST_CAP, 2)]
    return [('ref',) + pick('reference', ROW_CAP // 2, REF_SEED), ('cal',) + pick('calibration', ROW_CAP // 2, CAL_SEED),
            ('test',) + pick('test', ROW_CAP, TEST_SEED)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--study', action='store_true')
    p.add_argument('--draws', type=int)
    p.add_argument('--no-prescale', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--limit', type=int, help='debug: rows per set')
    a = p.parse_args()
    out = OUT / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    prescale = not a.no_prescale
    draws = a.draws if a.draws is not None else (STUDY_DRAWS if a.study else DRAWS)
    t0 = time.time()
    data = load_rows(a.dataset)
    values, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
    native = 'bars' in data
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    features = [f for f, k in zip(data['features'], keep) if k]
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    z_raw = ((values[:, keep] - center) / scale).astype(np.float64)
    positions = pd.Series(np.arange(len(groups))).groupby(groups).cumcount().to_numpy()
    windows = group_windows(values, groups, parts, LOOKBACK)
    model, tokenizer, official = load_kronos()
    states = KronosStates(model, tokenizer, official, seed=42)
    self_test(states)
    F = len(features)
    rows_per_batch = 256 if native else max(8, SEQUENCES_PER_BATCH // F)
    meta = dict(dataset=a.dataset, representation='native' if native else 'proxy', features=features, lookback=LOOKBACK,
                prescale=prescale, draws=draws, study=a.study, layers=states.layer_count, d_model=int(model.d_model),
                model_sha256=sha256(MODEL_DIR / 'model.safetensors'), tokenizer_sha256=sha256(TOKENIZER_DIR / 'model.safetensors'),
                rows_per_batch=rows_per_batch, sets={})
    print(f'{a.dataset}: {len(values)} rows, {F} features, {"native bars" if native else "proxy"}; {rows_per_batch} rows per batch '
          f'({time.time() - t0:.0f}s to load)', flush=True)
    for name, targets, histories in row_sets(windows, labels, a.study):
        path = out / f'{name}.npz'
        if path.exists() and not a.force:
            print(f'  {name}: {path} exists, skipped', flush=True)
            continue
        if a.limit:
            targets, histories = targets[:a.limit], histories[:a.limit]
        want_draws = draws if (a.study or name != 'ref') else 0
        want_H = not a.study
        n = len(targets)
        H = np.empty((n, states.layer_count, model.d_model), np.float16) if want_H else None
        nll = np.empty(n)
        nll_feature = np.empty((n, F)) if not native else None
        fc_mean = np.empty((n, 6 if native else F)) if want_draws else None
        fc_spread = np.empty((n, 6 if native else F)) if want_draws else None
        partial = out / f'{name}.partial.npz'
        first = 0
        if partial.exists() and not a.force:
            with np.load(partial) as ck:
                if ck['targets'].shape == targets.shape and np.array_equal(ck['targets'], targets):
                    first = int(ck['done'])
                    for key, array in (('H', H), ('nll', nll), ('nll_feature', nll_feature), ('fc_mean', fc_mean), ('fc_spread', fc_spread)):
                        if array is not None:
                            array[:first] = ck[key][:first]
                    print(f'  {name}: resuming from row {first} ({partial})', flush=True)
        t1 = time.time()
        for start in range(first, n, rows_per_batch):
            sl = slice(start, min(n, start + rows_per_batch))
            b = sl.stop - sl.start
            if native:
                x, stamp, terms = native_sequences(data['bars'], histories[sl], targets[sl], data['timestamps'], prescale)
                res = states.run(x, stamp, capture=want_H, draws=want_draws)
                nll[sl] = -res['loglik']
                if want_draws:
                    raw = denormalize(res['draws'], terms)                       # (b, S, 6) bars in raw units
                    fc_mean[sl], fc_spread[sl] = raw.mean(axis=1), raw.std(axis=1)
            else:
                x, stamp, terms = proxy_sequences(z_raw[histories[sl]], z_raw[targets[sl]], positions[targets[sl]], prescale)
                res = states.run(x, stamp, capture=want_H, draws=want_draws)
                nll_feature[sl] = -res['loglik'].reshape(b, F)
                pooled = proxy_pool(res, b, F, want_draws)
                nll[sl] = -pooled['loglik']
                if want_H:
                    res_H = pooled['H']
                if want_draws:
                    mean, std, factor = (t[:, 3].reshape(b, F) for t in terms)
                    z = (pooled['draws'] * (std[..., None] + 1e-5) + mean[..., None]) * factor[..., None]   # standardized units
                    rawf = z * scale[None, :, None] + center[None, :, None]                                 # feature units
                    fc_mean[sl], fc_spread[sl] = rawf.mean(axis=2), rawf.std(axis=2)
            if want_H:
                H[sl] = (res['H'] if native else res_H).astype(np.float16)
            if (start // rows_per_batch) % 20 == 0:
                done = sl.stop
                rate = (done - first) / max(time.time() - t1, 1e-9)
                print(f'  {name}: {done}/{n} rows, {rate:.1f} rows/s, eta {(n - done) / max(rate, 1e-9) / 60:.1f} min', flush=True)
            if (start // rows_per_batch) % 40 == 39 and sl.stop < n:
                np.savez(partial, done=sl.stop, targets=targets, **{k: v for k, v in (('H', H), ('nll', nll), ('nll_feature', nll_feature),
                                                                                     ('fc_mean', fc_mean), ('fc_spread', fc_spread)) if v is not None})
        payload = dict(targets=targets, histories=histories, nll=nll, labels=labels[targets])
        if want_H:
            payload['H'] = H
        if nll_feature is not None:
            payload['nll_feature'] = nll_feature
        if want_draws:
            payload['forecast_mean'], payload['forecast_spread'] = fc_mean, fc_spread
        np.savez_compressed(path, **payload)
        if partial.exists():
            partial.unlink()
        meta['sets'][name] = dict(rows=int(n), positives=int((labels[targets] == 1).sum()), seconds=round(time.time() - t1, 1),
                                  draws=want_draws, hidden_states=want_H)
        print(f'  {name}: {n} rows done in {(time.time() - t1) / 60:.1f} min -> {path}', flush=True)
        (out / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'total {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
