import argparse
import json
import hashlib
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .backend import KronosBackend
from .detector import Config, Detector, FEATURES


def main():
    p = argparse.ArgumentParser(description="Rolling Kronos forecast-error detection with offline POT")
    p.add_argument("--reference", required=True, help="Earlier history CSV")
    p.add_argument("--calibration", required=True, help="Independent primarily normal calibration CSV")
    p.add_argument("--test", required=True, help="Later test CSV; labels never read")
    p.add_argument("--kronos-repo", required=True)
    p.add_argument("--output", required=True, help="New output directory")
    p.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    p.add_argument("--epsilon", type=float, nargs="+", default=None)
    p.add_argument("--lookback", type=int, default=128)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--risk", type=float, default=0.001, help="Per-feature unconditional tail risk")
    p.add_argument("--min-exceedances", type=int, default=30)
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--model-revision")
    p.add_argument("--tokenizer-revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-context", type=int, default=512)
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    c = Config(a.lookback, tuple(a.features), tuple(a.epsilon or [1e-5]*len(a.features)),
               a.quantile, a.risk, a.min_exceedances)
    reference, calibration, test = [pd.read_csv(path) for path in (a.reference, a.calibration, a.test)]
    out = Path(a.output)
    if out.exists():
        p.error("Output directory already exists; select a new path")
    backend = KronosBackend(a.kronos_repo, a.model, a.tokenizer, a.device, a.max_context,
                            a.sample_count, a.temperature, a.top_p, a.seed,
                            a.model_revision, a.tokenizer_revision)
    detector = Detector(backend, c).fit(calibration, context=reference)
    result = detector.predict(test, context=pd.concat([reference, calibration], ignore_index=True))
    out.mkdir(parents=True)
    result.to_csv(out / "detections.csv", index=False)
    detector.save(out / "calibration.json")
    try:
        commit = subprocess.check_output(["git", "-C", a.kronos_repo, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(arguments=vars(a), kronos_commit=commit, python=sys.version,
                    dependencies={name: version(name) for name in ("numpy", "pandas", "scipy", "torch", "huggingface_hub")},
                    input_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for name, path in (("reference", a.reference), ("calibration", a.calibration), ("test", a.test))},
                    scored_test_rows=int(result.scored.sum()),
                    test_alarm_rate=float(result.loc[result.scored, "anomaly"].mean()) if result.scored.any() else None)
    (out / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(result)} rows to {out}; calibration joint alarm rate="
          f"{detector.state['calibration_joint_alarm_rate']:.6f}")


if __name__ == "__main__":
    main()
