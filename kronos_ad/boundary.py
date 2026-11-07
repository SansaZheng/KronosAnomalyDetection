"""Knowledge-boundary analysis of a transformer forecaster.

Transfer of the coarse-to-fine framework of Huang, Mao, Jiang and Cambria,
"Uncovering the Knowledge Boundaries and Mechanistic 'Knob' of LLMs", from
last-token hidden states of an LLM to the last-context-bar hidden states of a
time-series transformer such as Kronos.

Inputs are H with shape (n, L, d): for each row, the hidden state of the last
context position after every layer (layer 0 = embedding output), and a binary
label y. Two labels are meaningful for anomaly detection:

* the model's own boundary: y = 1 when its next-bar forecast was poor, which needs
  no anomaly labels and answers "does the model know that it does not know";
* the anomaly boundary: y = 1 on labelled anomalous bars.

Steps, each a function or class below:

1. prediction   probes g(H): logistic regression, MLP, multi-view CNN;
2. structure    class-conditional difference map, V_corr, H_corr, SNR_dB, LateRatio,
                ranked dimension energy and its knee;
3. knobs        nested top-k dimension subsets ranked by s_j = sum_l |dA_lj|; the
                knob set is the smallest k whose accuracy is within delta of the
                full probe, checked against random subsets of the same size;
4. distillation layer-attention pooling z = sum_l alpha_l H[l, J] and a closed-form
                score Phi(z) = c + sum_i sum_m w_im log(a_im + b_im z_i), the paper's
                log-primitive form, fitted directly by cross-entropy;
5. intervention a step tau along the boundary normal in knob space, spread over
                layers proportionally to alpha_l (the minimum-norm patch), with
                tangent and random directions as controls; the effect is measured
                by a model-specific teacher-forced log-likelihood of the realised
                next bar (see kronos_ad.hidden).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier


# ----------------------------------------------------------------------------- data


def balanced_indices(y, rng):
    """Downsample the majority class to a 1:1 set, as the paper does for its splits."""
    y = np.asarray(y).astype(int)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    size = min(len(positive), len(negative))
    chosen = np.r_[rng.choice(positive, size, replace=False), rng.choice(negative, size, replace=False)]
    return np.sort(chosen)


def split_balanced(y, test_fraction=0.2, seed=0):
    rng = np.random.default_rng(seed)
    chosen = balanced_indices(y, rng)
    rng.shuffle(chosen)
    cut = int(round(len(chosen) * (1 - test_fraction)))
    return np.sort(chosen[:cut]), np.sort(chosen[cut:])


class Standardizer:
    """Per-(layer, dim) standardization fitted on training rows."""

    def fit(self, H):
        H = np.asarray(H, dtype=float)
        self.mean_ = H.mean(axis=0)
        self.std_ = H.std(axis=0)
        self.std_ = np.where(self.std_ > 1e-12, self.std_, 1.0)
        return self

    def __call__(self, H):
        return (np.asarray(H, dtype=float) - self.mean_) / self.std_


# ---------------------------------------------------------------------------- probes


class LogisticProbe:
    """g(H) = linear classifier on the flattened, standardized hidden states."""

    def __init__(self, C=1.0, max_iter=2000):
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, H, y):
        self.scaler_ = Standardizer().fit(H)
        self.model.fit(self.scaler_(H).reshape(len(H), -1), np.asarray(y).astype(int))
        return self

    def score(self, H):
        return self.model.decision_function(self.scaler_(H).reshape(len(H), -1))

    def predict(self, H):
        return (self.score(H) > 0).astype(int)


class MLPProbe(LogisticProbe):
    def __init__(self, hidden=(128,), max_iter=400, seed=0):
        self.model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=seed,
                                   early_stopping=True)

    def score(self, H):
        p = self.model.predict_proba(self.scaler_(H).reshape(len(H), -1))[:, 1]
        return np.log(np.clip(p, 1e-9, 1 - 1e-9)) - np.log(np.clip(1 - p, 1e-9, 1 - 1e-9))


def _torch():
    import torch
    return torch


class CNNProbe:
    """Multi-view CNN probe: LayerNorm over d, then vertical (k_v x 1), horizontal
    (1 x k_h) and local (k x k) convolutions with ReLU and global max pooling, concatenated
    into a linear classifier. Kernel sizes shrink to fit shallow or narrow models."""

    def __init__(self, channels=8, kernels=((9, 1), (1, 33), (5, 5)), epochs=80, lr=3e-3, seed=0, weight_decay=1e-4):
        self.channels, self.kernels, self.epochs, self.lr, self.seed, self.weight_decay = channels, kernels, epochs, lr, seed, weight_decay

    def _build(self, L, d):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        odd = lambda k, size: max(1, min(k, size) - (1 - min(k, size) % 2))
        kernels = [(odd(kv, L), odd(kh, d)) for kv, kh in self.kernels]

        class Net(nn.Module):
            def __init__(net):
                super().__init__()
                net.norm = nn.LayerNorm(d)
                net.branches = nn.ModuleList([nn.Conv2d(1, self.channels, k, padding='same') for k in kernels])
                net.head = nn.Linear(self.channels * len(kernels), 2)

            def activations(net, x):
                x = net.norm(x).unsqueeze(1)
                return [torch.relu(b(x)) for b in net.branches]

            def forward(net, x):
                pooled = [a.amax(dim=(2, 3)) for a in net.activations(x)]
                return net.head(torch.cat(pooled, dim=1))

        return Net()

    def fit(self, H, y):
        torch = _torch()
        self.scaler_ = Standardizer().fit(H)
        X = torch.tensor(self.scaler_(H), dtype=torch.float32)
        Y = torch.tensor(np.asarray(y).astype(int))
        self.net_ = self._build(X.shape[1], X.shape[2])
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=64, shuffle=True)
        self.net_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                torch.nn.functional.cross_entropy(self.net_(xb), yb).backward()
                optimizer.step()
        self.net_.eval()
        return self

    def logits(self, H):
        torch = _torch()
        with torch.no_grad():
            return self.net_(torch.tensor(self.scaler_(H), dtype=torch.float32)).numpy()

    def score(self, H):
        logits = self.logits(H)
        return logits[:, 1] - logits[:, 0]

    def predict(self, H):
        return (self.score(H) > 0).astype(int)

    def activation_maps(self, H, branch=0):
        """Channel-averaged pre-pooling activations A(x) in R^{L x d} of one branch."""
        torch = _torch()
        with torch.no_grad():
            maps = self.net_.activations(torch.tensor(self.scaler_(H), dtype=torch.float32))[branch]
        return maps.mean(dim=1).numpy()


def balanced_accuracy(probe, H, y):
    return float(balanced_accuracy_score(np.asarray(y).astype(int), probe.predict(H)))


# ------------------------------------------------------------------------- structure


def difference_map(A, y):
    """Class-conditional mean difference, mean(A | y=1) - mean(A | y=0), shape (L, d)."""
    A, y = np.asarray(A, dtype=float), np.asarray(y).astype(int)
    return A[y == 1].mean(axis=0) - A[y == 0].mean(axis=0)


def _corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def summary_metrics(delta, percentile=95, late_fraction=0.5, eps=1e-12):
    """V_corr, H_corr, SNR_dB and LateRatio of |delta| (Appendix B of the paper)."""
    M = np.abs(np.asarray(delta, dtype=float))
    L, d = M.shape
    v_corr = _corr(M[:-1], M[1:]) if L > 1 else float('nan')
    h_corr = _corr(M[:, :-1], M[:, 1:]) if d > 1 else float('nan')
    energy = M.mean(axis=0)
    cut = np.percentile(energy, percentile)
    signal, noise = energy[energy > cut], energy[energy <= cut]
    snr = 10 * np.log10((signal ** 2).sum() / len(signal) / (noise.var() + eps) + eps) if len(signal) and len(noise) > 1 else float('nan')
    start = int(np.floor((1 - late_fraction) * L))
    late = M[start:].sum() / (M.sum() + eps)
    return dict(v_corr=v_corr, h_corr=h_corr, snr_db=float(snr), late_ratio=float(late))


def dimension_energy(delta):
    return np.abs(np.asarray(delta, dtype=float)).mean(axis=0)


def rank_dimensions(delta):
    """Importance s_j = sum_l |delta_lj|, descending."""
    return np.argsort(-np.abs(np.asarray(delta, dtype=float)).sum(axis=0), kind='stable')


def knee(values):
    """Knee of a decreasing ranked profile: the point farthest from the chord."""
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return 0
    x = np.linspace(0, 1, len(v))
    span = v[0] - v[-1]
    yv = (v - v[-1]) / span if span > 0 else np.zeros_like(v)
    chord = 1 - x
    return int(np.argmax(np.abs(yv - chord)))


# ------------------------------------------------------------------- knob predictor


class KnobPredictor:
    """Layer-attention pooling plus the paper's log-primitive closed form.

    z_i = sum_l softmax(w)_l H[l, J_i]
    Phi(z) = c + sum_i sum_m w_im log(a_im + b_im z_i)
    The log argument is clamped at `floor` to stay defined; with standardized z and
    the initialization used here it is rarely active.
    """

    def __init__(self, terms=2, epochs=300, lr=0.02, seed=0, floor=1e-2, l2=1e-4):
        self.terms, self.epochs, self.lr, self.seed, self.floor, self.l2 = terms, epochs, lr, seed, floor, l2

    def _pool(self, H):
        torch = _torch()
        alpha = torch.softmax(self.w_, dim=0)
        return torch.einsum('l,nlk->nk', alpha, H)

    def _phi(self, z):
        torch = _torch()
        argument = torch.clamp(self.a_ + self.b_ * z.unsqueeze(-1), min=self.floor)
        return self.c_ + (self.weight_ * torch.log(argument)).sum(dim=(1, 2))

    def fit(self, H, y, dims):
        torch = _torch()
        torch.manual_seed(self.seed)
        self.dims_ = np.asarray(dims)
        sub = np.asarray(H, dtype=float)[:, :, self.dims_]
        self.scaler_ = Standardizer().fit(sub)
        X = torch.tensor(self.scaler_(sub), dtype=torch.float32)
        Y = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)
        n, L, k = X.shape
        self.w_ = torch.zeros(L, requires_grad=True)
        self.a_ = torch.full((k, self.terms), 5.0, requires_grad=True)
        signs = torch.tensor([[1.0, -1.0][m % 2] for m in range(self.terms)]).repeat(k, 1)
        self.b_ = (signs * 1.0).clone().requires_grad_(True)
        self.weight_ = torch.zeros(k, self.terms, requires_grad=True)
        self.c_ = torch.zeros(1, requires_grad=True)
        parameters = [self.w_, self.a_, self.b_, self.weight_, self.c_]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self._phi(self._pool(X))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y)
            loss = loss + self.l2 * sum((p ** 2).sum() for p in (self.a_, self.b_, self.weight_))
            loss.backward()
            optimizer.step()
        return self

    def pooled(self, H):
        torch = _torch()
        with torch.no_grad():
            return self._pool(torch.tensor(self.scaler_(np.asarray(H)[:, :, self.dims_]), dtype=torch.float32)).numpy()

    def score(self, H):
        torch = _torch()
        with torch.no_grad():
            return self._phi(torch.tensor(self.pooled(H), dtype=torch.float32)).numpy()

    def predict(self, H):
        return (self.score(H) > 0).astype(int)

    def alpha(self):
        torch = _torch()
        with torch.no_grad():
            return torch.softmax(self.w_, dim=0).numpy()

    def gradient(self, z):
        """dPhi/dz at pooled coordinates z, shape (n, k)."""
        torch = _torch()
        zt = torch.tensor(np.asarray(z, dtype=float), dtype=torch.float32, requires_grad=True)
        self._phi(zt).sum().backward()
        return zt.grad.numpy()

    def closed_form(self, names=None):
        names = names or [f'x{i + 1}' for i in range(len(self.dims_))]
        terms = []
        for i, name in enumerate(names):
            for m in range(self.terms):
                w, a, b = (float(t.detach()[i, m]) for t in (self.weight_, self.a_, self.b_))
                terms.append(f'{w:+.3f}*log({a:.3f} {b:+.3f}*{name})')
        return f'Phi = {float(self.c_):+.3f} ' + ' '.join(terms)


# ------------------------------------------------------------------------- knobs


@dataclass
class KnobSearch:
    k_values: tuple
    accuracies: dict
    full_accuracy: float
    knob_k: int
    knob_dims: np.ndarray
    random_control: dict


def nested_knob_search(H_train, y_train, H_test, y_test, ranking, full_accuracy, delta=0.01, k_max=10,
                       make_predictor=None, random_trials=50, seed=0):
    """Smallest top-k subset (by `ranking`) whose predictor is within `delta` of the full probe."""
    make_predictor = make_predictor or (lambda: KnobPredictor())
    accuracies, knob_k = {}, None
    for k in range(1, k_max + 1):
        dims = ranking[:k]
        predictor = make_predictor().fit(H_train, y_train, dims)
        accuracies[k] = balanced_accuracy(predictor, H_test, y_test)
        if knob_k is None and accuracies[k] >= full_accuracy - delta:
            knob_k = k
    chosen_k = knob_k if knob_k is not None else max(accuracies, key=accuracies.get)
    rng = np.random.default_rng(seed)
    d = np.asarray(H_train).shape[2]
    random_scores = []
    for _ in range(random_trials):
        dims = rng.choice(d, chosen_k, replace=False)
        random_scores.append(balanced_accuracy(make_predictor().fit(H_train, y_train, dims), H_test, y_test))
    random_scores = np.asarray(random_scores)
    control = dict(mean=float(random_scores.mean()), std=float(random_scores.std()), min=float(random_scores.min()),
                   max=float(random_scores.max()), fraction_at_least_topk=float((random_scores >= accuracies[chosen_k]).mean()),
                   trials=random_trials)
    return KnobSearch(tuple(accuracies), accuracies, full_accuracy, chosen_k, np.asarray(ranking[:chosen_k]), control)


# ------------------------------------------------------------------- interventions


def intervention_directions(predictor, z, seed=0, eps=1e-6):
    """Unit normal n(z), a unit tangent orthogonal to it and a random unit direction, per row."""
    rng = np.random.default_rng(seed)
    gradient = predictor.gradient(z)
    norm = np.linalg.norm(gradient, axis=1, keepdims=True)
    normal = gradient / (norm + eps)
    random_direction = rng.normal(size=normal.shape)
    random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True) + eps
    tangent = rng.normal(size=normal.shape)
    tangent -= (tangent * normal).sum(axis=1, keepdims=True) * normal
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.where(tangent_norm > eps, tangent / (tangent_norm + eps), 0.0)
    return dict(normal=normal, tangent=tangent, random=random_direction), norm[:, 0]


def layer_patches(predictor, delta_z):
    """Minimum-norm layer-wise patch: dH[l, J] = alpha_l / sum_m alpha_m^2 * delta_z (Eq. 10).

    Returns an array (n, L, k) in standardized knob units; callers rescale by the
    predictor's standardizer to hidden-state units.
    """
    alpha = predictor.alpha()
    share = alpha / (alpha ** 2).sum()
    return np.einsum('l,nk->nlk', share, np.asarray(delta_z, dtype=float))


def pooled_shift(tau, gradient_norm, direction, eps=1e-6):
    """Delta z(tau, d) = tau / (||grad Phi|| + eps) * d (Eq. 8)."""
    return tau / (gradient_norm[:, None] + eps) * direction


def hidden_patch(predictor, H, tau, direction_name, seed=0):
    """Full patch in hidden-state units, shape (n, L, d_model), nonzero only on knob dims."""
    z = predictor.pooled(H)
    directions, gradient_norm = intervention_directions(predictor, z, seed)
    delta_z = pooled_shift(tau, gradient_norm, directions[direction_name])
    patch_knob = layer_patches(predictor, delta_z) * predictor.scaler_.std_[None]
    patch = np.zeros(np.asarray(H).shape, dtype=float)
    patch[:, :, predictor.dims_] = patch_knob
    return patch
