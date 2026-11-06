# BRD/BTC rerun with real weights: protocol frozen before inference

Written 2026-11-06 while the Kronos weights and the Binance archive were unreachable from this session. `frozen_config.json` fixes the arms, sampling, partitions, features, scoring and endpoints; `frozen_manifest.json` carries the SHA-256 of the configuration and of every script involved.

The question is narrow: does the saved native run's forecast failure come from the official predictor's `(std + 1e-5)` normalization flattening a 5e-5-priced series (E1), and once that is fixed, does Kronos-mini carry one-step skill (E2) or a usable predictive spread (E3) on BRD/BTC 5-minute bars? Detection numbers (E4) are descriptive: the test period has one annotated bar.

Rules: one run per arm with the frozen parameters; thresholds from the validation half only; nothing tuned on the test period; every deviation listed with before/after numbers.
