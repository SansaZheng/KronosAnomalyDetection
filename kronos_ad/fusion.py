"""Evidence fusion for Kronos forecast residuals (label-free, causal, frozen before test).

Pipeline (each step is fitted on calibration rows only):

1. Forecast stacking: per feature, an L1-optimal convex blend of Kronos and naive
   forecasts. A feature on which Kronos has no skill receives weight near 0, so a
   degenerate forecast cannot dominate the residual.
2. Conditional scale: a pinball-loss model of |residual| on robust and non-robust
   window scales. The fit decides how much a recent shock should widen the scale, so
   volatility clustering is learned rather than assumed.
3. Evidence channels: `forecast` (one-step onset), `local` (deviation from the window
   median, which persists through an interval) and `global` (deviation from the
   reference distribution, for stationary features only).
4. Conformal p-values with a GPD tail, so ranks stay continuous past 1/n.
5. Fusion: a row's score is its strongest evidence, 1 / min p over features and
   channels. The harmonic mean is also available; on calibration data it was not better.
6. One joint threshold from a held-out partition, so the alarm rate does not inflate
   with dimension as a per-feature OR does. It can also be tracked online.
"""
from collections import deque
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from scipy.optimize import linprog

MAD_TO_STD = 1.4826
CHANNELS = ("forecast", "local", "global")


def mad(values, axis):
    median = np.median(values, axis=axis, keepdims=True)
    return MAD_TO_STD * np.median(np.abs(values - median), axis=axis)


def robust_reference(reference):
    """Median and scale of a reference block. A zero MAD falls back to std, then 1."""
    reference = np.asarray(reference, dtype=float)
    center = np.median(reference, axis=0)
    scale = mad(reference, 0)
    std = reference.std(axis=0)
    scale = np.where(scale > 1e-12, scale, std)
    return center, np.where(scale > 1e-12, scale, 1.0)


def simplex_grid(k, step=0.05):
    units = int(round(1 / step))
    points = [c + (units - sum(c),) for c in product(range(units + 1), repeat=k - 1) if sum(c) <= units]
    return np.asarray(points, dtype=float) / units


def fit_stack(target, candidates, step=0.05):
    """Weights minimizing mean absolute error; ties favor the first candidate."""
    grid = simplex_grid(candidates.shape[1], step)
    errors = np.abs(target[:, None] - candidates @ grid.T).mean(axis=0)
    return grid[int(np.argmin(np.round(errors, 12)))]


def fit_pinball(design, target, quantile=0.75):
    """Nonnegative linear quantile regression solved as a linear program."""
    n, k = design.shape
    cost = np.r_[np.zeros(k), quantile * np.ones(n), (1 - quantile) * np.ones(n)] / n
    equality = np.hstack([design, np.eye(n), -np.eye(n)])
    result = linprog(cost, A_eq=equality, b_eq=target, bounds=(0, None), method="highs")
    if not result.success:
        raise ValueError("Scale model failed: " + result.message)
    return result.x[:k]


class TailPValue:
    """Conformal p-value; beyond the initial quantile, a Hosking-Wallis GPD tail."""

    def __init__(self, scores, tail_quantile=0.9, shape_limits=(0.0, 0.75)):
        self.sorted_ = np.sort(np.asarray(scores, dtype=float))
        self.n_ = len(self.sorted_)
        self.u_ = float(np.quantile(self.sorted_, tail_quantile))
        excess = self.sorted_[self.sorted_ > self.u_] - self.u_
        self.n_tail_ = len(excess)
        self.shape_ = 0.0
        self.scale_ = 0.0
        if self.n_tail_ >= 5 and excess.mean() > 0:
            plot = (np.arange(1, self.n_tail_ + 1) - 0.35) / self.n_tail_
            a0, a1 = excess.mean(), (excess * (1 - plot)).mean()
            shape = 2 - a0 / (a0 - 2 * a1) if a0 != 2 * a1 else 0.0
            self.shape_ = float(np.clip(shape, *shape_limits))
            self.scale_ = float(a0 * (1 - self.shape_))

    def __call__(self, scores):
        scores = np.asarray(scores, dtype=float)
        ge = self.n_ - np.searchsorted(self.sorted_, scores, side="left")
        p = (1 + ge) / (self.n_ + 1)
        if self.scale_ > 0:
            tail = scores > self.u_
            z = (scores[tail] - self.u_) / self.scale_
            sf = np.exp(-z) if self.shape_ < 1e-9 else (1 + self.shape_ * z) ** (-1 / self.shape_)
            p[tail] = (self.n_tail_ + 1) / (self.n_ + 1) * sf
        return np.clip(p, 1e-300, 1.0)


@dataclass(frozen=True)
class FusionConfig:
    channels: tuple = CHANNELS
    naive: tuple = ("last", "median")
    tail_quantile: float = 0.9
    scale_quantile: float = 0.75
    alpha: float = 0.01
    stack_step: float = 0.05
    scale_terms: tuple = ("level", "step", "level_std", "step_std")
    aggregate: str = "max"

    def __post_init__(self):
        if not self.channels or not set(self.channels) <= set(CHANNELS):
            raise ValueError("Unknown evidence channel")
        if self.aggregate not in {"max", "harmonic"} or not 0 < self.alpha < 1:
            raise ValueError("Invalid aggregate or alpha")


class EvidenceFusion:
    """x has shape (n, d); windows (n, L, d), oldest first and strictly before x."""

    def __init__(self, config=None):
        self.config = config or FusionConfig()

    @staticmethod
    def _context(windows):
        windows = np.asarray(windows, dtype=float)
        steps = np.diff(windows, axis=1)
        return dict(last=windows[:, -1], median=np.median(windows, axis=1),
                    level=mad(windows, 1), step=mad(steps, 1),
                    level_std=windows.std(axis=1), step_std=steps.std(axis=1))

    def _forecast(self, context, forecasts, j):
        names = [name for name in self.config.naive] + sorted(forecasts)
        return names, np.column_stack([context[n][:, j] if n in context else forecasts[n][:, j] for n in names])

    def _design(self, context, j):
        n = len(context["level"])
        terms = [context[name][:, j] for name in self.config.scale_terms]
        return np.column_stack(terms + [np.full(n, self.reference_scale_[j])])

    def _raw(self, x, context, forecasts):
        """Nonconformity per channel, shape (n, d, channels)."""
        x = np.asarray(x, dtype=float)
        columns = []
        for name in self.config.channels:
            if name == "global":
                columns.append(np.abs(x - self.reference_center_) / self.reference_scale_)
                continue
            out = np.empty_like(x)
            for j in range(x.shape[1]):
                if name == "forecast":
                    _, candidates = self._forecast(context, forecasts, j)
                    prediction = candidates @ self.weights_[j]
                else:
                    prediction = context["median"][:, j]
                scale = self._design(context, j) @ self.scale_coef_[name][j]
                out[:, j] = np.abs(x[:, j] - prediction) / np.maximum(scale, 1e-3 * self.reference_scale_[j])
            columns.append(out)
        return np.stack(columns, axis=2)

    def fit(self, x, windows, forecasts, reference, global_mask=None):
        c = self.config
        x = np.asarray(x, dtype=float)
        self.reference_center_, self.reference_scale_ = robust_reference(reference)
        context = self._context(windows)
        self.feature_count_ = x.shape[1]
        names, self.weights_ = None, []
        for j in range(x.shape[1]):
            names, candidates = self._forecast(context, forecasts, j)
            self.weights_.append(fit_stack(x[:, j], candidates, c.stack_step))
        self.weights_ = np.asarray(self.weights_)
        self.forecast_names_ = names
        self.scale_coef_ = {}
        for name in ("forecast", "local"):
            coefficients = []
            for j in range(x.shape[1]):
                if name == "forecast":
                    prediction = self._forecast(context, forecasts, j)[1] @ self.weights_[j]
                else:
                    prediction = context["median"][:, j]
                coefficients.append(fit_pinball(self._design(context, j), np.abs(x[:, j] - prediction), c.scale_quantile))
            self.scale_coef_[name] = np.asarray(coefficients)
        self.global_mask_ = np.ones(x.shape[1], bool) if global_mask is None else np.asarray(global_mask, bool)
        raw = self._raw(x, context, forecasts)
        self.tails_ = [[TailPValue(raw[:, j, k], c.tail_quantile) for k in range(raw.shape[2])]
                       for j in range(x.shape[1])]
        return self

    def pvalues(self, x, windows, forecasts):
        raw = self._raw(x, self._context(windows), forecasts)
        p = np.empty_like(raw)
        for j in range(raw.shape[1]):
            for k in range(raw.shape[2]):
                p[:, j, k] = self.tails_[j][k](raw[:, j, k])
        return p

    def active_mask(self):
        """Channel-feature pairs that vote; `global` skips nonstationary features."""
        mask = np.ones((self.feature_count_, len(self.config.channels)), bool)
        if "global" in self.config.channels:
            mask[~self.global_mask_, self.config.channels.index("global")] = False
        return mask

    def score(self, x, windows, forecasts):
        p = self.pvalues(x, windows, forecasts)[:, self.active_mask()]
        return (1 / p).mean(axis=1) if self.config.aggregate == "harmonic" else 1 / p.min(axis=1)

    def explain(self, x, windows, forecasts):
        """Feature and channel index of each row's smallest active p-value."""
        p = self.pvalues(x, windows, forecasts)
        p = np.where(self.active_mask()[None], p, np.inf)
        flat = p.reshape(len(p), -1).argmin(axis=1)
        return np.unravel_index(flat, p.shape[1:])

    def calibrate(self, x, windows, forecasts, alpha=None):
        alpha = self.config.alpha if alpha is None else alpha
        scores = np.sort(self.score(x, windows, forecasts))
        self.calibration_scores_ = scores
        rank = min(int(np.ceil((len(scores) + 1) * (1 - alpha))), len(scores))
        self.threshold_ = float(scores[rank - 1])
        return self

    def state(self):
        return dict(config=asdict(self.config), forecast_names=self.forecast_names_,
                    weights=self.weights_.tolist(), threshold=self.threshold_)


def track_threshold(scores, calibration, alpha=0.01, gamma=0.005, window=512):
    """Adaptive conformal alarms: each decision uses only earlier scores.

    The miscoverage level follows a_{t+1} = a_t + gamma (alpha - alarm_t), so the
    long-run alarm rate tends to alpha under any drift (Gibbs and Candes, 2021). That
    is control of the alarm budget, not of false alarms. Alarmed scores are kept out
    of the buffer so a burst of anomalies does not raise the threshold on its own.
    """
    buffer = deque(np.asarray(calibration, dtype=float), maxlen=window)
    level, flags, thresholds = alpha, [], []
    for score in np.asarray(scores, dtype=float):
        if level <= 0:
            threshold = np.inf
        elif level >= 1:
            threshold = -np.inf
        else:
            threshold = float(np.quantile(np.fromiter(buffer, float), 1 - level, method="higher"))
        alarm = bool(score > threshold)
        flags.append(alarm)
        thresholds.append(threshold)
        level += gamma * (alpha - alarm)
        if not alarm:
            buffer.append(score)
    return np.asarray(flags), np.asarray(thresholds)
