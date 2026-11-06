# Round 2: protocol frozen before the second test evaluation

Round 1 (`results/2026-11-06/kronos_fusion`) scored the test rows once. Its weaknesses, recorded there, were level-type anomalies, tabular streams where sequential channels add noise, and the maximum over single pairs ignoring diffuse evidence.

Round 2 was designed on calibration rows only:

1. `dev_ladder.csv`: the subsampled calibration partitions used in round 1 (0–11 labelled positives per dataset, so only the counterfactual injection power is informative there).
2. `dev_full.csv`: every calibration row with a 32-row history, scored by the pipeline without Kronos (whose stacking weight is 0–0.2 everywhere), fit on the earlier half and evaluated on the later half with real labels: 25–714 positives per dataset.

Fourteen configurations were tried in total (C0–C11 in `scripts/dev_ladder_r2.py`); every one is in the two CSV files. The selection rule was written into `scripts/dev_full_r2.py` before it ran: best mean rank of AP and recall at 1% FPR over datasets with at least 20 dev positives, ties broken by the injection ladder. It chose C1+multi (`frozen_config.json`). The two closest configurations (C3+multi, within 0.1 rank; C6 with Isolation Forest) are not tested as separate arms; the ensemble arm is the selected configuration plus the Isolation Forest row channel.

Test looks after this freeze: two (the selected configuration and the ensemble arm), on the same test rows as round 1. Round-1 scores are kept as a paired comparison. Rules from round 1 apply: later changes are listed with before and after numbers, datasets with fewer than 5 test positives are descriptive, every arm is reported.

Known limits going in: the dev sets for ULB (34 positives) and Sapienza (25 pumps shared by three resolutions) are small; S-FFSD's dev period differs from its test period (Robust MAD reaches AP 0.16 on dev and 0.61 on test); no new Kronos inference.
