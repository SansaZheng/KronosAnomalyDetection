"""Generate synthetic pipeline inputs, not evidence of Kronos performance."""
from pathlib import Path
import numpy as np
import pandas as pd

out = Path("runs/demo-data")
out.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(42)
n = 1600
close = 100 + np.cumsum(rng.normal(0, 0.1, n))
volume = rng.lognormal(8, 0.2, n)
frame = pd.DataFrame(dict(timestamps=pd.date_range("2024-01-01", periods=n, freq="h"),
                         open=close, high=close+0.2, low=close-0.2, close=close,
                         volume=volume, amount=close*volume))
frame.iloc[:128].to_csv(out / "reference.csv", index=False)
frame.iloc[128:1400].to_csv(out / "calibration.csv", index=False)
# Synthetic turnover here is provided for model input only; diagnose OHLC in this demo.
frame.loc[1500, "close"] += 10
frame.iloc[1400:].to_csv(out / "test.csv", index=False)
print(out)
