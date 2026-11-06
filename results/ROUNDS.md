# Test-set looks for the evidence-fusion method

Each row is one evaluation of a frozen configuration on the test rows. The test rows are the same in every round (the published Kronos-proxy positions and the BRD/BTC test period), so later rounds are paired with earlier ones.

| Round | Date | Directory | Looks | Designed on | Outcome |
|---|---|---|---|---|---|
| 1 | 2026-11-06 | `2026-11-06/kronos_fusion` | 1 (audit corrections changed no ranking) | subsampled calibration partitions (injection power) | beat published POT on 5/5 primary datasets; no Kronos contribution; level baselines ahead on BankSim, SD-stock, S-FFSD |
| 2 | 2026-11-07 | `2026-11-07/kronos_fusion_r2` | 2 (pure method, ensemble arm) | all calibration rows, 14 configurations (C7–C11 added after the first ten had been scored on calibration rows; all listed), rule written into the script before the ladder ran | AP above round 1 on 4/5 primary datasets, paired intervals excluding 0 on 3 (S-FFSD, SD-stock, SD-synthetic); SD-prices regressed (fit partition is one 128-row case); Kronos still no positive contribution; level baselines still ahead on BankSim, SD-stock, S-FFSD |

Rules: a configuration is frozen with a SHA-256 manifest before its test look; every configuration tried on calibration rows is listed; changes after a look are recorded with before/after numbers; intervals are not corrected for the number of looks, so treat them as descriptive across rounds.
