# Round 2: protocol frozen before the second test evaluation

Round 1 (`results/2026-11-06/kronos_fusion`) scored the test rows once. Its weaknesses, recorded there, were level-type anomalies, tabular streams where sequential channels add noise, and the maximum over single pairs ignoring diffuse evidence.

Round 2 was designed on calibration rows only:

1. `dev_ladder.csv`: the subsampled calibration partitions used in round 1 (0–11 labelled positives per dataset, so only the counterfactual injection power is informative there).
2. `dev_full.csv`: every calibration row with a 32-row history, scored by the pipeline without Kronos (whose stacking weight is 0–0.2 everywhere), fit on the earlier half and evaluated on the later half with real labels: 25–714 positives per dataset.

Fifteen configurations were tried in total (C0–C11 in `scripts/dev_ladder_r2.py`; dev_ladder.csv holds the first ten, dev_full.csv all fifteen). The sequence, corrected after an independent audit of the file timestamps: the subsampled ladder came first with its own rule (which would have kept the round-1 configuration, and whose alarm check no configuration could pass on 128-row halves); that rule was replaced by the one in `scripts/dev_full_r2.py` (best mean rank of AP and recall at 1% FPR over datasets with at least 20 dev positives, ties broken by the injection ladder), written before the full ladder ran; C7–C10 were added after the first batch had been seen (BankSim motivated the magnitude statistics) and C11 after that; at the moment of the freeze the C7–C10 results for three datasets and all C11 results were still being computed. The audit checked that C1+multi wins the rule under the partial table, the final table, and with or without the extra configurations, so the selection stands, but it was not as clean as "rule fixed before running" implies. C1+multi is `frozen_config.json`; the ensemble arm is the same configuration plus the Isolation Forest row channel.

Test looks after this freeze: two (the selected configuration and the ensemble arm), on the same test rows as round 1. Round-1 scores are kept as a paired comparison. Rules from round 1 apply: later changes are listed with before and after numbers, datasets with fewer than 5 test positives are descriptive, every arm is reported.

Known limits going in: the dev sets for ULB (34 positives) and Sapienza (25 pumps shared by three resolutions) are small; S-FFSD's dev period differs from its test period (Robust MAD reaches AP 0.16 on dev and 0.61 on test); no new Kronos inference.
