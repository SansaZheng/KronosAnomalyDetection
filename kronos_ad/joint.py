"""Frozen robust residual scoring and one joint empirical threshold.

Fitting consumes calibration residuals only. No test observations or labels are
used in the fitted bias, feature scaling, or threshold.
"""
import numpy as np


class JointResidualCalibrator:
    def __init__(self, floor=0.1, aggregate="top3", bias=True, risk=0.01):
        if floor <= 0 or not 0 < risk < 1 or aggregate not in {"max", "top3", "mean"}:
            raise ValueError("Invalid floor, aggregate, or risk")
        self.floor, self.aggregate, self.bias, self.risk = floor, aggregate, bias, risk

    def _features(self, residual, local_std):
        residual, local_std = np.asarray(residual), np.asarray(local_std)
        if residual.ndim != 2 or residual.shape != local_std.shape:
            raise ValueError("Residuals and local standard deviations must have equal 2D shapes")
        if not np.isfinite(residual).all() or not np.isfinite(local_std).all() or (local_std < 0).any():
            raise ValueError("Inputs must be finite; standard deviations must be nonnegative")
        return np.abs(residual - self.offset_) / np.maximum(local_std, self.floor * self.reference_scale_)

    def score(self, residual, local_std):
        features = self._features(residual, local_std) / self.feature_scale_
        if self.aggregate == "max":
            return features.max(axis=1)
        if self.aggregate == "mean":
            return features.mean(axis=1)
        k = min(3, features.shape[1])
        return np.sort(features, axis=1)[:, -k:].mean(axis=1)

    def fit(self, residual, local_std, reference_scale):
        residual = np.asarray(residual, dtype=float)
        scale = np.asarray(reference_scale, dtype=float)
        if residual.ndim != 2 or len(residual) < 20 or scale.shape != (residual.shape[1],):
            raise ValueError("Need at least 20 calibration rows and one reference scale per feature")
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Reference scales must be finite and positive")
        self.reference_scale_ = scale.copy()
        self.offset_ = np.median(residual, axis=0) if self.bias else np.zeros(residual.shape[1])
        # Balance heterogeneous residual scales without extreme-tail GPD fitting.
        feature = self._features(residual, local_std)
        self.feature_scale_ = np.maximum(np.median(feature, axis=0), 1e-8)
        scores = self.score(residual, local_std)
        # Order-statistic quantile with finite-sample correction. Exchangeability
        # is not assumed for financial streams, so risk is a nominal target only.
        rank = int(np.ceil((len(scores) + 1) * (1 - self.risk)))
        if rank > len(scores):
            raise ValueError("Calibration sample too small for requested risk")
        self.threshold_ = float(np.sort(scores)[rank - 1])
        return self

    def predict(self, residual, local_std):
        # Strict > avoids flagging every row when a score distribution has ties.
        return self.score(residual, local_std) > self.threshold_

    def state(self):
        return dict(floor=self.floor, aggregate=self.aggregate, bias=self.bias,
                    risk=self.risk, threshold=self.threshold_, offset=self.offset_.tolist(),
                    reference_scale=self.reference_scale_.tolist(), feature_scale=self.feature_scale_.tolist())
