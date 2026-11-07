"""Shared knowledge-boundary pipeline (steps 1-5) for any model that exposes
hidden states at the last context position and a teacher-forced log-likelihood
of the realised next bar with an optional hidden-state patch."""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from kronos_ad.boundary import (CNNProbe, KnobPredictor, LogisticProbe, MLPProbe, Standardizer, balanced_accuracy,
                                difference_map, dimension_energy, hidden_patch, knee, nested_knob_search,
                                rank_dimensions, summary_metrics)

TAUS = (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0)
DELTA = 0.01


def batched(function, *arrays, size=4096):
    return np.concatenate([function(*[a[i:i + size] for a in arrays]) for i in range(0, len(arrays[0]), size)])


def thin(index, cap, seed=0):
    if len(index) <= cap:
        return index
    return np.sort(np.random.default_rng(seed).choice(index, cap, replace=False))


def analyse_boundary(name, H_train, y_train, H_test, y_test, loglik_test, out, log, k_max=10, random_trials=50,
                     seed=0, delta=DELTA, taus=TAUS, cnn_epochs=80):
    """Probe, localise, search knobs, distil, intervene. `loglik_test(patch, parts)` scores
    the balanced test rows; with parts=True it also returns (log-variance, error) terms
    or None when the model has no such decomposition."""
    result = dict(boundary=name, train_rows=int(len(H_train)), test_rows=int(len(H_test)),
                  train_positives=int(np.sum(y_train)), test_positives=int(np.sum(y_test)))
    probes = {'logistic': LogisticProbe(), 'mlp': MLPProbe(), 'cnn': CNNProbe(epochs=cnn_epochs)}
    for key, probe in probes.items():
        t = time.time()
        probe.fit(H_train, y_train)
        result[f'acc_{key}'] = balanced_accuracy(probe, H_test, y_test)
        log(f'  {name}: {key} probe balanced accuracy {result[f"acc_{key}"]:.4f} ({time.time() - t:.0f}s)')
    cnn = probes['cnn']
    maps = cnn.activation_maps(H_train, branch=0)
    delta_cnn = difference_map(maps, y_train)
    delta_raw = difference_map(Standardizer().fit(H_train)(H_train), y_train)
    for label, delta_map in (('cnn', delta_cnn), ('raw', delta_raw)):
        result[f'structure_{label}'] = summary_metrics(delta_map)
        energy = np.sort(dimension_energy(delta_map))[::-1]
        result[f'knee_{label}'] = int(knee(energy / energy[0]))
        np.save(out / f'{name}_delta_{label}.npy', delta_map)
    ranking = rank_dimensions(delta_cnn)
    search = nested_knob_search(H_train, y_train, H_test, y_test, ranking, result['acc_cnn'], delta=delta, k_max=k_max,
                                make_predictor=lambda: KnobPredictor(epochs=300), random_trials=random_trials, seed=seed)
    result['knob'] = dict(k=search.knob_k, dims=search.knob_dims.tolist(),
                          accuracies={int(k): float(v) for k, v in search.accuracies.items()},
                          full_accuracy=search.full_accuracy, random_control=search.random_control,
                          near_optimal=bool(search.accuracies[search.knob_k] >= search.full_accuracy - delta),
                          ranking_top20=ranking[:20].tolist())
    log(f'  {name}: knob set k={search.knob_k} dims={search.knob_dims.tolist()} acc={search.accuracies[search.knob_k]:.4f} '
        f'(full {search.full_accuracy:.4f}; random-{search.knob_k} mean {search.random_control["mean"]:.4f}, '
        f'fraction >= top-k {search.random_control["fraction_at_least_topk"]:.2f})')
    predictor = KnobPredictor(epochs=400, seed=seed).fit(H_train, y_train, search.knob_dims)
    result['knob']['closed_form'] = predictor.closed_form()
    result['knob']['alpha'] = predictor.alpha().tolist()
    result['knob']['distilled_accuracy'] = balanced_accuracy(predictor, H_test, y_test)
    np.savez(out / f'{name}_knob_space.npz', z=predictor.pooled(H_test), y=y_test, score=predictor.score(H_test))
    base = loglik_test(None, True)
    base_ll, base_parts = base[0], base[1]
    rows = []
    for tau in taus:
        effects, readout = {}, {}
        for direction in ('normal', 'tangent', 'random'):
            patch = hidden_patch(predictor, H_test, tau, direction, seed=seed)
            ll, parts = loglik_test(patch, True)
            effects[direction] = ll - base_ll
            if parts is not None:
                readout[direction] = (parts[0] - base_parts[0], parts[1] - base_parts[1])
        normal = effects['normal']
        row = dict(boundary=name, tau=tau, mean_normal=float(normal.mean()), median_normal=float(np.median(normal)),
                   improved_normal=float((normal > 0).mean()), mean_tangent=float(effects['tangent'].mean()),
                   mean_random=float(effects['random'].mean()),
                   mean_normal_y0=float(normal[np.asarray(y_test) == 0].mean()),
                   mean_normal_y1=float(normal[np.asarray(y_test) == 1].mean()),
                   p_normal_vs_zero=float(wilcoxon(normal, alternative='two-sided').pvalue),
                   p_normal_vs_tangent=float(wilcoxon(normal, effects['tangent'], alternative='two-sided').pvalue),
                   p_normal_vs_random=float(wilcoxon(normal, effects['random'], alternative='two-sided').pvalue))
        if readout:
            row.update(dlogvar_normal=float(readout['normal'][0].mean()), derror_normal=float(readout['normal'][1].mean()),
                       dlogvar_tangent=float(readout['tangent'][0].mean()), dlogvar_random=float(readout['random'][0].mean()))
        rows.append(row)
        log(f'  {name}: tau {tau:+.2f} dlogp normal {normal.mean():+.4f} (y0 {row["mean_normal_y0"]:+.4f}, y1 {row["mean_normal_y1"]:+.4f}) '
            f'tangent {effects["tangent"].mean():+.4f} random {effects["random"].mean():+.4f}')
    result['intervention'] = rows
    return result, probes, predictor


def save_results(out, header, results, detection):
    (out / 'results.json').write_text(json.dumps(dict(header, boundaries=results), indent=1) + '\n')
    pd.DataFrame(detection).to_csv(out / 'anomaly_detection.csv', index=False)
    pd.DataFrame([r for res in results for r in res['intervention']]).to_csv(out / 'intervention.csv', index=False)
