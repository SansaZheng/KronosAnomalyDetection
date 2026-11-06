"""Evidence fusion for Kronos forecast residuals (label-free, causal, frozen before test).

Pipeline (each step is fitted on calibration rows only):

1. Forecast stacking: per feature, an L1-optimal convex blend of Kronos and naive
   forecasts. A feature on which Kronos has no skill receives weight near 0, so a
   degenerate forecast cannot dominate the residual.
2. Conditional scale: a pinball-loss model of |residual| on robust and non-robust
   window scales. The fit decides how much a recent shock should widen the scale, so
   volatility clustering is learned rather than assumed. A forecaster's predictive
   spread (a `forecasts` key ending in `:spread`, e.g. the std over Kronos draws) is
   one more nonnegative term, so a calibrated predictive distribution tightens or
   widens the scale only to the extent that it helped on the fit partition.
3. Evidence channels: `forecast` (one-step onset), `local` (deviation from the window
   median, which persists through an interval), `long` (deviation from the median of a
   longer causal history, keys `long:center` and `long:scale` in `forecasts`) and
   `global` (deviation from the reference distribution, for stationary features only;
   either in scale units or as a two-sided rank rarity). Row-level evidence from a
   multivariate detector enters through keys `row:<name>`.
4. Skill gating: sequential channels vote on a feature only when the stacked forecast
   beats the level predictors on the fit partition; otherwise they add noise to the
   maximum without adding information. Contextual features (calendar codes) are
   restricted to `global` through `sequential_mask`.
5. Conformal p-values with a GPD tail, so ranks stay continuous past 1/n. They are
   kept as log p, so an extreme row cannot underflow to a tie at the floor.
6. Fusion: `max` takes the strongest evidence, -log min p over features and channels;
   `multi` computes Tippett and Fisher statistics per channel, calibrates each on the
   fit partition, and takes the strongest calibrated statistic, so diffuse evidence
   across many features counts as well as one extreme feature.
7. One joint threshold from a held-out partition, so the alarm rate does not inflate
   with dimension as a per-feature OR does. It can also be tracked online.
"""
from collections import deque
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import logsumexp

MAD_TO_STD = 1.4826
CHANNELS = ("forecast", "local", "long", "global")
SEQUENTIAL = ("forecast", "local")
SPECIAL_SUFFIXES = (":spread", ":center", ":scale")


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
    identity = sparse.identity(n, format="csc")
    equality = sparse.hstack([sparse.csc_matrix(np.asarray(design, dtype=float)), identity, -identity], format="csc")
    result = linprog(cost, A_eq=equality, b_eq=target, bounds=(0, None), method="highs")
    if not result.success:
        raise ValueError("Scale model failed: " + result.message)
    return result.x[:k]


class TailPValue:
    """Conformal log p-value; beyond the initial quantile, a Hosking-Wallis GPD tail."""

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
        """Natural log of the p-value; finite for every finite score."""
        scores = np.asarray(scores, dtype=float)
        ge = self.n_ - np.searchsorted(self.sorted_, scores, side="left")
        log_p = np.log((1 + ge) / (self.n_ + 1))
        if self.scale_ > 0:
            tail = scores > self.u_
            z = (scores[tail] - self.u_) / self.scale_
            log_sf = -z if self.shape_ < 1e-9 else -np.log1p(self.shape_ * z) / self.shape_
            log_p[tail] = np.log((self.n_tail_ + 1) / (self.n_ + 1)) + log_sf
        return np.minimum(log_p, 0.0)


class RankRarity:
    """Two-sided reference rarity: -log min(F(x), 1 - F(x-)), ties counted on both sides."""

    def __init__(self, reference):
        self.sorted_ = np.sort(np.asarray(reference, dtype=float))
        self.n_ = len(self.sorted_)

    def __call__(self, values):
        values = np.asarray(values, dtype=float)
        below = np.searchsorted(self.sorted_, values, side="right")
        above = self.n_ - np.searchsorted(self.sorted_, values, side="left")
        tail = np.minimum(below, above)
        return -np.log((tail + 1) / (self.n_ + 1))


@dataclass(frozen=True)
class FusionConfig:
    channels: tuple = ("forecast", "local", "global")
    naive: tuple = ("last", "median")
    tail_quantile: float = 0.9
    scale_quantile: float = 0.75
    alpha: float = 0.01
    stack_step: float = 0.05
    scale_terms: tuple = ("level", "step", "level_std", "step_std")
    aggregate: str = "max"
    gate_skill: float = 0.0
    global_rarity: str = "scale"
    statistics: tuple = ("tippett", "fisher")
    row_evidence: bool = True

    def __post_init__(self):
        if not self.channels or not set(self.channels) <= set(CHANNELS):
            raise ValueError("Unknown evidence channel")
        if self.aggregate not in {"max", "harmonic", "multi"} or not 0 < self.alpha < 1:
            raise ValueError("Invalid aggregate or alpha")
        if self.global_rarity not in {"scale", "rank"} or not 0 <= self.gate_skill < 1:
            raise ValueError("Invalid global_rarity or gate_skill")
        if not self.statistics or not set(self.statistics) <= {"tippett", "fisher", "meanz", "maxz"}:
            raise ValueError("Unknown statistic")


def is_forecaster(name):
    return not any(name.endswith(s) for s in SPECIAL_SUFFIXES) and not name.startswith("row:")


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

    @staticmethod
    def _spreads(forecasts):
        return sorted(name for name in forecasts if name.endswith(":spread"))

    @staticmethod
    def row_sources(forecasts):
        return sorted(name for name in forecasts if name.startswith("row:"))

    def _forecast(self, context, forecasts, j):
        names = [name for name in self.config.naive] + sorted(n for n in forecasts if is_forecaster(n))
        return names, np.column_stack([context[n][:, j] if n in context else forecasts[n][:, j] for n in names])

    def _prediction(self, name, context, forecasts, j):
        if name == "forecast":
            return self._forecast(context, forecasts, j)[1] @ self.weights_[j]
        if name == "local":
            return context["median"][:, j]
        if name == "long":
            return forecasts["long:center"][:, j]
        raise ValueError(name)

    def _design(self, name, context, forecasts, j):
        n = len(context["level"])
        terms = [context[term][:, j] for term in self.config.scale_terms]
        terms += [np.abs(forecasts[s][:, j]) for s in self._spreads(forecasts)]
        if name == "long":
            terms.append(forecasts["long:scale"][:, j])
        return np.column_stack(terms + [np.full(n, self.reference_scale_[j])])

    def _raw(self, x, context, forecasts):
        """Nonconformity per channel, shape (n, d, channels)."""
        x = np.asarray(x, dtype=float)
        columns = []
        for name in self.config.channels:
            if name == "global":
                if self.config.global_rarity == "rank":
                    columns.append(np.column_stack([self.rarity_[j](x[:, j]) for j in range(x.shape[1])]))
                else:
                    columns.append(np.abs(x - self.reference_center_) / self.reference_scale_)
                continue
            out = np.empty_like(x)
            for j in range(x.shape[1]):
                prediction = self._prediction(name, context, forecasts, j)
                scale = self._design(name, context, forecasts, j) @ self.scale_coef_[name][j]
                out[:, j] = np.abs(x[:, j] - prediction) / np.maximum(scale, 1e-3 * self.reference_scale_[j])
            columns.append(out)
        return np.stack(columns, axis=2)

    def _row_raw(self, forecasts):
        names = self.row_sources(forecasts)
        if not self.config.row_evidence or not names:
            return np.empty((0, 0))
        return np.column_stack([np.asarray(forecasts[name], dtype=float) for name in names])

    def _gate(self, x, context, forecasts):
        """Feature-channel mask from forecast skill on the fit partition; label-free."""
        d = x.shape[1]
        mask = np.ones((d, len(self.config.channels)), bool)
        if "global" in self.config.channels:
            mask[~self.global_mask_, self.config.channels.index("global")] = False
        for name in SEQUENTIAL + ("long",):
            if name in self.config.channels:
                mask[~self.sequential_mask_, self.config.channels.index(name)] = False
        self.skill_ = {}
        if self.config.gate_skill > 0:
            errors = {}
            for name in self.config.channels:
                if name == "global":
                    errors[name] = np.abs(x - self.reference_center_).mean(axis=0)
                else:
                    errors[name] = np.column_stack([np.abs(x[:, j] - self._prediction(name, context, forecasts, j))
                                                    for j in range(d)]).mean(axis=0)
            level = [n for n in ("long", "global") if n in errors]
            for j in range(d):
                base = min((errors[n][j] for n in level if mask[j, self.config.channels.index(n)]), default=np.inf)
                if "forecast" in errors:
                    sequential_on = np.isfinite(base) and base > 0 and errors["forecast"][j] < self.config.gate_skill * base
                    for name in SEQUENTIAL:
                        if name in self.config.channels:
                            mask[j, self.config.channels.index(name)] &= bool(sequential_on)
                if "long" in errors and "global" in errors and mask[j, self.config.channels.index("global")]:
                    long_on = errors["global"][j] > 0 and errors["long"][j] < self.config.gate_skill * errors["global"][j]
                    mask[j, self.config.channels.index("long")] &= bool(long_on)
            self.skill_ = {name: value.tolist() for name, value in errors.items()}
        if not mask.any():
            mask = np.ones_like(mask)
            if "global" in self.config.channels:
                mask[~self.global_mask_, self.config.channels.index("global")] = False
        return mask

    def fit(self, x, windows, forecasts, reference, global_mask=None, sequential_mask=None):
        c = self.config
        x = np.asarray(x, dtype=float)
        reference = np.asarray(reference, dtype=float)
        self.reference_center_, self.reference_scale_ = robust_reference(reference)
        if c.global_rarity == "rank":
            self.rarity_ = [RankRarity(reference[:, j]) for j in range(x.shape[1])]
        context = self._context(windows)
        self.feature_count_ = x.shape[1]
        self.global_mask_ = np.ones(x.shape[1], bool) if global_mask is None else np.asarray(global_mask, bool)
        self.sequential_mask_ = np.ones(x.shape[1], bool) if sequential_mask is None else np.asarray(sequential_mask, bool)
        names, self.weights_ = None, []
        for j in range(x.shape[1]):
            names, candidates = self._forecast(context, forecasts, j)
            self.weights_.append(fit_stack(x[:, j], candidates, c.stack_step))
        self.weights_ = np.asarray(self.weights_)
        self.forecast_names_ = names
        self.scale_coef_ = {}
        for name in self.config.channels:
            if name == "global":
                continue
            coefficients = []
            for j in range(x.shape[1]):
                prediction = self._prediction(name, context, forecasts, j)
                coefficients.append(fit_pinball(self._design(name, context, forecasts, j),
                                                np.abs(x[:, j] - prediction), c.scale_quantile))
            self.scale_coef_[name] = np.asarray(coefficients)
        self.scale_terms_ = list(c.scale_terms) + self._spreads(forecasts) + ["reference"]
        self.mask_ = self._gate(x, context, forecasts)
        raw = self._raw(x, context, forecasts)
        self.tails_ = [[TailPValue(raw[:, j, k], c.tail_quantile) for k in range(raw.shape[2])]
                       for j in range(x.shape[1])]
        row_raw = self._row_raw(forecasts)
        self.row_names_ = self.row_sources(forecasts) if row_raw.shape[1] else []
        self.row_tails_ = [TailPValue(row_raw[:, r], c.tail_quantile) for r in range(row_raw.shape[1])]
        if c.aggregate == "multi":
            statistics = self._statistics(self._log_p(raw), raw, self._row_log_p(row_raw))
            self.statistic_names_ = list(statistics)
            self.statistic_tails_ = {k: TailPValue(v, c.tail_quantile) for k, v in statistics.items()}
        return self

    def _log_p(self, raw):
        log_p = np.empty_like(raw)
        for j in range(raw.shape[1]):
            for k in range(raw.shape[2]):
                log_p[:, j, k] = self.tails_[j][k](raw[:, j, k])
        return log_p

    def _row_log_p(self, row_raw):
        return np.column_stack([self.row_tails_[r](row_raw[:, r]) for r in range(row_raw.shape[1])]) \
            if row_raw.shape[1] else np.empty((len(row_raw), 0))

    def log_pvalues(self, x, windows, forecasts):
        return self._log_p(self._raw(x, self._context(windows), forecasts))

    def active_mask(self):
        """Channel-feature pairs that vote."""
        return self.mask_.copy()

    def _statistics(self, log_p, raw, row_log_p):
        """Named test statistics over the masked evidence grid; larger is more anomalous.

        `tippett` and `fisher` work on calibrated log p-values (rank evidence);
        `meanz` and `maxz` work on the scale-normalized nonconformities themselves
        (magnitude evidence), which keeps the ordering among extreme rows that a
        heavy GPD tail compresses.
        """
        out = {}
        groups = {name: self.mask_[:, k] for k, name in enumerate(self.config.channels)}
        groups["all"] = None
        for name, features in groups.items():
            if name == "all":
                block, magnitude = log_p[:, self.mask_], raw[:, self.mask_]
            else:
                k = self.config.channels.index(name)
                block, magnitude = log_p[:, features, k], raw[:, features, k]
            if block.shape[1] == 0:
                continue
            if "tippett" in self.config.statistics:
                out[name + ":tippett"] = -block.min(axis=1)
            if "maxz" in self.config.statistics and name != "all":
                out[name + ":maxz"] = magnitude.max(axis=1)
            if block.shape[1] > 1:
                if "fisher" in self.config.statistics:
                    out[name + ":fisher"] = -block.sum(axis=1)
                if "meanz" in self.config.statistics and name != "all":
                    out[name + ":meanz"] = magnitude.mean(axis=1)
        for r, name in enumerate(self.row_names_):
            out[name] = -row_log_p[:, r]
        return out

    def score(self, x, windows, forecasts):
        """Fused evidence: -log min p (max), log mean 1/p (harmonic) or max calibrated statistic (multi)."""
        raw = self._raw(x, self._context(windows), forecasts)
        log_p = self._log_p(raw)
        row_raw = self._row_raw(forecasts) if self.row_names_ else np.empty((len(log_p), 0))
        if self.config.aggregate == "multi":
            statistics = self._statistics(log_p, raw, self._row_log_p(row_raw))
            calibrated = np.column_stack([-self.statistic_tails_[k](statistics[k]) for k in self.statistic_names_])
            return calibrated.max(axis=1)
        masked = log_p[:, self.mask_]
        if self.row_names_:
            masked = np.hstack([masked, self._row_log_p(row_raw)])
        if self.config.aggregate == "harmonic":
            return logsumexp(-masked, axis=1) - np.log(masked.shape[1])
        return -masked.min(axis=1)

    def explain(self, x, windows, forecasts):
        """Feature and channel index of each row's smallest active p-value."""
        log_p = self.log_pvalues(x, windows, forecasts)
        log_p = np.where(self.mask_[None], log_p, np.inf)
        flat = log_p.reshape(len(log_p), -1).argmin(axis=1)
        return np.unravel_index(flat, log_p.shape[1:])

    def calibrate(self, x, windows, forecasts, alpha=None):
        alpha = self.config.alpha if alpha is None else alpha
        scores = np.sort(self.score(x, windows, forecasts))
        self.calibration_scores_ = scores
        rank = min(int(np.ceil((len(scores) + 1) * (1 - alpha))), len(scores))
        self.threshold_ = float(scores[rank - 1])
        return self

    def state(self):
        return dict(config=asdict(self.config), forecast_names=self.forecast_names_,
                    weights=self.weights_.tolist(), scale_terms=self.scale_terms_,
                    scale_coefficients={k: v.tolist() for k, v in self.scale_coef_.items()},
                    active=self.mask_.tolist(), skill=self.skill_, row_sources=self.row_names_,
                    statistics=getattr(self, "statistic_names_", []), threshold=self.threshold_)


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
