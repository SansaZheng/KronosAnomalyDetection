# Test-set looks for the evidence-fusion method

Each row is one evaluation of a frozen configuration on the test rows. The test rows are the same in every round (the published Kronos-proxy positions and the BRD/BTC test period), so later rounds are paired with earlier ones.

| Round | Date | Directory | Looks | Designed on | Outcome |
|---|---|---|---|---|---|
| 1 | 2026-11-06 | `2026-11-06/kronos_fusion` | 1 (audit corrections changed no ranking) | subsampled calibration partitions (injection power) | beat published POT on 5/5 primary datasets; no Kronos contribution; level baselines ahead on BankSim, SD-stock, S-FFSD |
| 2 | 2026-11-07 | `2026-11-07/kronos_fusion_r2` | 2 (pure method, ensemble arm) | all calibration rows, 14 configurations (C7–C11 added after the first ten had been scored on calibration rows; all listed), rule written into the script before the ladder ran | one reliable, small AP gain over round 1 (SD-synthetic, 200 units); two suggestive (S-FFSD 7 blocks, SD-stock 6 cases); one reliable regression (SD-prices, fit partition is one 128-row case); Kronos still no positive contribution; Robust MAD ahead on 4/5 and Isolation Forest on 3/5 primary datasets. Independent audit: numbers reproduce, no leakage, selection story and post-freeze changes corrected in the README |

Rules: a configuration is frozen with a SHA-256 manifest before its test look; every configuration tried on calibration rows is listed; changes after a look are recorded with before/after numbers; intervals are not corrected for the number of looks, so treat them as descriptive across rounds.
