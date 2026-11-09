#!/usr/bin/env python3
"""Knowledge-boundary analysis on a stand-in transformer forecaster.

No Kronos weights are reachable from this environment, so the paper's pipeline is
exercised end to end on a small causal transformer trained here on the repo's own
15-second trade-chunk feature rows (Sapienza pump data, sapienza_15s). Everything below the model
is the same code path a real Kronos run uses (kronos_ad.boundary); only the
hidden-state adapter differs (StandinStates here, kronos_ad.hidden for Kronos).

Partitions follow the repo: the forecaster is trained on reference rows only; probes
are trained on labelled reference+calibration rows and evaluated on test rows.

Two boundaries:
  error  y = 1 when the model's Gaussian NLL of the realised next bar is above the
         calibration 90th percentile ("the model does not know this bar");
  pump   y = 1 on labelled pump bars.

Outputs go to results/2026-11-07/knowledge_boundary_standin/ for the original sapienza_15s
run; `--dataset <name>` trains a stand-in on another transaction dataset of the repository
(banksim, s_ffsd, ulb_creditcard) with the same protocol and writes to
results/2026-11-08/knowledge_boundary_standin_<name>/. The labelled boundary is called
`pump` on the Sapienza data and `fraud` on the transaction-fraud datasets.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from boundary_pipeline import analyse_boundary, batched, save_results, thin
from fusion_data import NOT_DETECTION_TARGETS, load_rows
from kronos_ad.boundary import split_balanced
from kronos_ad.fusion import robust_reference
from run_fusion import recall_at_fpr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/2026-11-07/knowledge_boundary_standin'
DATASET = 'sapienza_15s'


def default_out(dataset):
    return OUT if dataset == DATASET else ROOT / f'results/2026-11-08/knowledge_boundary_standin_{dataset}'


def label_boundary(dataset):
    return 'pump' if dataset.startswith(('sapienza', 'xu_')) else 'fraud'


LOOKBACK = 32
SEED = 20261107
D_MODEL, N_LAYERS, N_HEADS = 64, 4, 4
EPOCHS, BATCH, TRAIN_CAP = 3, 256, 240000
LOG_VAR_RANGE = (-4.0, 4.0)
CLIP = 5.0
ROW_CAP = 24000
K_MAX = 10
RANDOM_TRIALS = 50


# ---------------------------------------------------------------------------- data


def group_windows(values, groups, parts, lookback):
    """Target rows with `lookback` earlier rows of the same group, per partition."""
    groups = np.asarray(groups)
    local = np.empty(len(groups), dtype=np.int64)
    members = {}
    for group in pd.unique(groups):
        positions = np.flatnonzero(groups == group)
        local[positions] = np.arange(len(positions))
        members[group] = positions
    out = {}
    for part in ('reference', 'calibration', 'test'):
        targets = np.flatnonzero((np.asarray(parts) == part) & (local >= lookback))
        histories = np.array([members[groups[i]][local[i] - lookback:local[i]] for i in targets], dtype=np.int64)
        out[part] = (targets, histories)
    return out


# --------------------------------------------------------------------------- model


class StandinForecaster(torch.nn.Module):
    """Causal transformer with a Gaussian next-bar head; hidden states are exposed by hooks."""

    def __init__(self, d_in, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, lookback=LOOKBACK):
        super().__init__()
        self.embed = torch.nn.Linear(d_in, d_model)
        self.position = torch.nn.Embedding(lookback, d_model)
        self.blocks = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(d_model)
        self.mean = torch.nn.Linear(d_model, d_in)
        self.log_var = torch.nn.Linear(d_model, d_in)
        self.register_buffer('mask', torch.triu(torch.full((lookback, lookback), float('-inf')), diagonal=1))

    def forward(self, x):
        T = x.shape[1]
        h = self.embed(x) + self.position(torch.arange(T, device=x.device))[None]
        for block in self.blocks:
            h = block(h, src_mask=self.mask[:T, :T], is_causal=True)
        h = self.norm(h)
        return self.mean(h), self.log_var(h).clamp(*LOG_VAR_RANGE)

    def nll(self, x, target):
        """Gaussian NLL of the next row at every position: targets are x shifted by one plus `target`."""
        mean, log_var = self(x)
        future = torch.cat([x[:, 1:], target[:, None]], dim=1)
        return 0.5 * (log_var + (future - mean) ** 2 / log_var.exp() + np.log(2 * np.pi)).sum(dim=-1)


class StandinStates:
    """Same contract as kronos_ad.hidden.KronosHiddenStates for the stand-in model."""

    def __init__(self, model):
        self.model = model.eval()

    @property
    def layer_count(self):
        return len(self.model.blocks) + 1

    def _hooks(self, patch=None, capture=None):
        handles = []

        def pre_hook(module, args, kwargs):
            x = args[0]
            if capture is not None:
                capture.append(x[:, -1].detach().clone())
            if patch is not None:
                x = x.clone()
                x[:, -1] = x[:, -1] + torch.as_tensor(patch[:, 0], dtype=x.dtype)
            return (x,) + tuple(args[1:]), kwargs

        handles.append(self.model.blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
        for index, block in enumerate(self.model.blocks):
            def hook(module, args, output, layer=index + 1):
                out = output
                if capture is not None:
                    capture.append(out[:, -1].detach().clone())
                if patch is not None:
                    out = out.clone()
                    out[:, -1] = out[:, -1] + torch.as_tensor(patch[:, layer], dtype=out.dtype)
                return out
            handles.append(block.register_forward_hook(hook))
        return handles

    def hidden_states(self, x):
        captured = []
        handles = self._hooks(capture=captured)
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            for h in handles:
                h.remove()
        return torch.stack(captured, dim=1).numpy()

    def loglik(self, x, target, patch=None, parts=False):
        """log p(next bar | context) under the Gaussian head at the last position.

        With `parts`, also return the summed predictive log-variance and the squared
        standardized error, the two terms whose change explains an intervention.
        """
        handles = self._hooks(patch=patch)
        try:
            with torch.no_grad():
                mean, log_var = self.model(x)
                m, v = mean[:, -1], log_var[:, -1]
                error = ((target - m) ** 2 / v.exp()).sum(dim=-1)
                ll = -0.5 * (v.sum(dim=-1) + error + x.shape[2] * np.log(2 * np.pi))
        finally:
            for h in handles:
                h.remove()
        if parts:
            return ll.numpy(), v.sum(dim=-1).numpy(), error.numpy()
        return ll.numpy()


# ------------------------------------------------------------------------ pipeline


def train_forecaster(x_train, y_train, x_val, y_val, log):
    torch.manual_seed(SEED)
    model = StandinForecaster(x_train.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    n = len(x_train)
    history = []
    best = (np.inf, None)
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            optimizer.zero_grad()
            loss = model.nll(x_train[idx], y_train[idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            val = float(np.mean(batched(lambda a, b: model.nll(a, b)[:, -1].numpy(), x_val, y_val)))
        history.append(dict(epoch=epoch + 1, train_nll=total / n, val_last_step_nll=val))
        log(f'epoch {epoch + 1}: train nll {total / n:.3f}  calibration last-step nll {val:.3f}')
        if val < best[0]:
            best = (val, {k: v.clone() for k, v in model.state_dict().items()})
    # keep the epoch with the best calibration likelihood (the stand-in's own validation)
    model.load_state_dict(best[1])
    model.eval()
    history.append(dict(selected_epoch=int(np.argmin([h['val_last_step_nll'] for h in history]) + 1)))
    return model, history


def naive_nll(x_val, y_val):
    """Persistence forecaster with calibration-fitted variance, as a skill reference."""
    residual = (y_val - x_val[:, -1]).numpy()
    var = residual.var(axis=0)
    return float(np.mean(0.5 * (np.log(var) + residual ** 2 / var + np.log(2 * np.pi)).sum(axis=1)))


def main(dataset=DATASET, out=None):
    warnings.filterwarnings('ignore')
    out = Path(out) if out else default_out(dataset)
    out.mkdir(parents=True, exist_ok=True)
    positive_name = label_boundary(dataset)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    torch.manual_seed(SEED)
    data = load_rows(dataset)
    values, parts, groups, labels = data['values'], data['parts'], data['groups'], data['labels']
    keep = np.array([f not in NOT_DETECTION_TARGETS for f in data['features']])
    features = [f for f, k in zip(data['features'], keep) if k]
    center, scale = robust_reference(values[parts == 'reference'][:, keep])
    z = ((values[:, keep] - center) / scale).astype(np.float32)
    z = np.clip(z, -CLIP, CLIP)
    windows = group_windows(values, groups, parts, LOOKBACK)
    log(f'{dataset}: features {features}; rows with history: ' + ', '.join(f'{k} {len(v[0])}' for k, v in windows.items()))

    def tensors(part, pick):
        """Windows for the given positions within a partition's eligible rows; every labelled positive is kept."""
        targets, histories = windows[part]
        positives = np.flatnonzero(labels[targets] == 1)
        pick = np.union1d(pick, positives)
        return torch.tensor(z[histories[pick]]), torch.tensor(z[targets[pick]]), targets[pick]

    def sample(part, cap, seed):
        return thin(np.arange(len(windows[part][0])), cap, seed)

    x_ref, y_ref, _ = tensors('reference', sample('reference', TRAIN_CAP, 1))
    x_cal, y_cal, t_cal = tensors('calibration', sample('calibration', ROW_CAP // 2, 2))
    x_test, y_test, t_test = tensors('test', sample('test', ROW_CAP, 3))
    log(f'forecaster training windows {len(x_ref)}, calibration rows {len(x_cal)}, test rows {len(x_test)}')
    t0 = time.time()
    model, history = train_forecaster(x_ref, y_ref, x_cal, y_cal, log)
    persistence = naive_nll(x_cal, y_cal)
    log(f'training took {time.time() - t0:.0f}s; persistence+calibration-variance nll on calibration {persistence:.3f}')
    torch.save(model.state_dict(), out / 'standin_forecaster.pt')
    states = StandinStates(model)

    # labelled pools: probes train on reference+calibration rows, evaluate on test rows
    x_pool, y_pool, t_pool = tensors('reference', sample('reference', ROW_CAP // 2, 4))
    x_pool = torch.cat([x_pool, x_cal])
    y_pool = torch.cat([y_pool, y_cal])
    t_pool = np.r_[t_pool, t_cal]
    H_pool = batched(states.hidden_states, x_pool)
    H_test = batched(states.hidden_states, x_test)
    nll_pool = -batched(states.loglik, x_pool, y_pool)
    nll_test = -batched(states.loglik, x_test, y_test)
    calibration_rows = np.arange(len(t_pool) - len(t_cal), len(t_pool))
    threshold = float(np.quantile(nll_pool[calibration_rows][labels[t_cal] != 1], 0.9))  # normal calibration rows only
    boundaries = {
        'error': ((nll_pool > threshold).astype(int), (nll_test > threshold).astype(int)),
        positive_name: (labels[t_pool].astype(int), labels[t_test].astype(int)),
    }
    log(f'hidden states {H_pool.shape[1:]}; error boundary threshold (calibration 90th pct NLL) {threshold:.3f}; '
        f'{positive_name} positives pool {int((boundaries[positive_name][0] == 1).sum())}, test {int((boundaries[positive_name][1] == 1).sum())}')
    results, detection = [], []
    for name, (y_pool_b, y_test_b) in boundaries.items():
        train_idx, _ = split_balanced(y_pool_b, test_fraction=0.0, seed=SEED)
        test_idx, _ = split_balanced(y_test_b, test_fraction=0.0, seed=SEED + 1)
        if (y_test_b == 1).sum() < 10 or (y_pool_b == 1).sum() < 20:
            log(f'  {name}: too few positives for a balanced probe; skipped')
            continue
        x_rows, target_rows = x_test[test_idx], y_test[test_idx]

        def loglik_test(patch, parts):
            ll, log_var, error = states.loglik(x_rows, target_rows, patch=patch, parts=True)
            return ll, (log_var, error)

        result, probes, predictor = analyse_boundary(name, H_pool[train_idx], y_pool_b[train_idx], H_test[test_idx], y_test_b[test_idx],
                                                     loglik_test, out, log, k_max=K_MAX, random_trials=RANDOM_TRIALS, seed=SEED)
        results.append(result)
        # anomaly relevance on all sampled test rows: pump labels, scores from internal signals and the black-box NLL
        pump = labels[t_test].astype(int)
        known = np.isin(pump, [0, 1])
        scores = {'nll_surprise': nll_test, 'cnn_probe': probes['cnn'].score(H_test), 'logistic_probe': probes['logistic'].score(H_test),
                  'knob_closed_form': predictor.score(H_test)}
        for key, s in scores.items():
            detection.append(dict(boundary=name, score=key, test_rows=int(known.sum()), positives=int(pump[known].sum()),
                                  ap=float(average_precision_score(pump[known], s[known])), auroc=float(roc_auc_score(pump[known], s[known])),
                                  recall_1pct_fpr=recall_at_fpr(pump[known], s[known])))
    header = dict(dataset=dataset, label_boundary=positive_name, features=features, model=dict(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, lookback=LOOKBACK, epochs=EPOCHS, training_windows=int(len(x_ref))),
                  training=history, persistence_nll=persistence, error_threshold=threshold)
    save_results(out, header, results, detection)
    (out / 'run.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=DATASET)
    parser.add_argument('--out')
    args = parser.parse_args()
    main(args.dataset, args.out)
