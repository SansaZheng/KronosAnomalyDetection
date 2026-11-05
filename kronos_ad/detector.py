"""Causal single-step scoring and frozen, independent calibration."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pot import fit_pot

PRICES = ("open", "high", "low", "close")
FEATURES = PRICES + ("volume", "amount")


@dataclass(frozen=True)
class Config:
    lookback: int = 128
    features: tuple = FEATURES
    epsilon: tuple = (1e-5,) * 6
    quantile: float = 0.95
    risk: float = 0.001
    min_exceedances: int = 30

    def __post_init__(self):
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if (not self.features or len(set(self.features)) != len(self.features)
                or tuple(f for f in FEATURES if f in self.features) != tuple(self.features)):
            raise ValueError("features must be an ordered subset of OHLCVA")
        eps = np.asarray(self.epsilon, dtype=float)
        if eps.shape != (len(self.features),) or not np.isfinite(eps).all() or (eps <= 0).any():
            raise ValueError("Provide one finite positive epsilon per diagnostic feature")
        if not 0 < self.quantile < 1 or not 0 < self.risk < 1 - self.quantile:
            raise ValueError("Require 0 < risk < 1 - quantile < 1")
        if self.min_exceedances < 3:
            raise ValueError("min_exceedances must be >= 3")


def validate(frame):
    frame = frame.copy().reset_index(drop=True)
    if "timestamps" not in frame or not set(PRICES).issubset(frame.columns):
        raise ValueError("Input requires timestamps, open, high, low, close")
    frame["timestamps"] = pd.to_datetime(frame.timestamps, errors="raise", utc=True).dt.tz_localize(None)
    if frame.empty or frame.timestamps.isna().any() or not frame.timestamps.is_monotonic_increasing or frame.timestamps.duplicated().any():
        raise ValueError("Timestamps must be nonempty, unique and strictly increasing")
    cols = [f for f in FEATURES if f in frame]
    frame = frame[["timestamps", *cols]]
    for f in cols:
        frame[f] = pd.to_numeric(frame[f], errors="raise")
    if not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
        raise ValueError("NaN/inf observations require explicit upstream handling")
    return frame


class Detector:
    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = config or Config()
        if self.config.lookback > getattr(backend, "max_context", self.config.lookback):
            raise ValueError("lookback exceeds backend max_context")
        self.state = None

    def score(self, frame, context=None):
        target = validate(frame)
        if context is None:
            combined, offset = target, 0
        else:
            context = validate(context)
            if context.timestamps.iloc[-1] >= target.timestamps.iloc[0]:
                raise ValueError("Context must be strictly earlier than the target period")
            if list(context.columns) != list(target.columns):
                raise ValueError("Context and target feature schemas must match")
            combined = pd.concat([context, target], ignore_index=True)
            offset = len(context)
        c = self.config
        if not set(c.features).issubset(target.columns):
            raise ValueError("Diagnostic features must be real input columns")
        # Copy only raw available columns. The adapter may synthesize V/A for the
        # model, but these never become diagnostic targets unless explicitly chosen.
        rows = []
        for j in range(len(target)):
            pos = offset + j
            row = dict(timestamps=target.timestamps.iloc[j], scored=pos >= c.lookback)
            for f in c.features:
                row.update({f"prediction_{f}": np.nan, f"score_{f}": np.nan})
            if row["scored"]:
                window = combined.iloc[pos-c.lookback:pos].copy().reset_index(drop=True)
                pred = self.backend.forecast(window, target.timestamps.iloc[j])
                p = np.asarray([pred[f] for f in c.features], dtype=float)
                if not np.isfinite(p).all():
                    raise ValueError("Backend produced nonfinite predictions")
                sigma = window[list(c.features)].std(ddof=0).to_numpy(dtype=float)
                observed = target.loc[j, list(c.features)].to_numpy(dtype=float)
                with np.errstate(over="raise", invalid="raise"):
                    scores = np.square((observed - p) / (sigma + np.asarray(c.epsilon)))
                for f, prediction, score in zip(c.features, p, scores):
                    row[f"prediction_{f}"] = prediction
                    row[f"score_{f}"] = score
            rows.append(row)
        return pd.DataFrame(rows)

    def fit(self, calibration, context=None):
        self.state = None
        scores = self.score(calibration, context)
        c = self.config
        tails = {f: fit_pot(scores.loc[scores.scored, f"score_{f}"].to_numpy(),
                            c.quantile, c.risk, c.min_exceedances) for f in c.features}
        flags = np.column_stack([scores.loc[scores.scored, f"score_{f}"].to_numpy() >= tails[f]["threshold"]
                                 for f in c.features])
        self.state = dict(config=asdict(c), tails=tails,
                          calibration_end=str(validate(calibration).timestamps.iloc[-1]),
                          calibration_joint_alarm_rate=float(flags.any(axis=1).mean()),
                          backend=getattr(self.backend, "metadata", {}))
        return self

    def predict(self, test, context=None):
        if self.state is None:
            raise ValueError("Fit calibration thresholds before prediction")
        if validate(test).timestamps.iloc[0] <= pd.Timestamp(self.state["calibration_end"]):
            raise ValueError("Test must start strictly after calibration ends")
        result = self.score(test, context)
        for f in self.config.features:
            threshold = self.state["tails"][f]["threshold"]
            result[f"threshold_{f}"] = threshold
            result[f"anomaly_{f}"] = pd.array(
                np.where(result.scored, result[f"score_{f}"] >= threshold, None), dtype="boolean")
        columns = [f"anomaly_{f}" for f in self.config.features]
        result["anomaly"] = result[columns].any(axis=1).astype("boolean")
        result.loc[~result.scored, "anomaly"] = pd.NA
        return result

    def save(self, path):
        if self.state is None:
            raise ValueError("No calibration state to save")
        Path(path).write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")
