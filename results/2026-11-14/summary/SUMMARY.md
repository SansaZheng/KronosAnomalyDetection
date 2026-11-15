# Kronos model-size study

Models: Kronos-mini (4.1 M parameters), Kronos-small (24.7 M parameters), Kronos-base (102.3 M parameters). Same rows for every model.

## Forecast skill (median over features of MAE / baseline MAE; below 1 = better than the baseline)

| dataset | model | features | mae_ratio_vs_last | mae_ratio_vs_median | beats_last | beats_median | fusion_weight |
|---|---|---|---|---|---|---|---|
| K-lines (Xu & Livshits + B0R0koko) | mini | 4 | 2.048 | 3.411 | 1 | 0 | 0.000 |
| K-lines (Xu & Livshits + B0R0koko) | small | 4 | 2.371 | 3.000 | 0 | 0 | 0.000 |
| K-lines (Xu & Livshits + B0R0koko) | base | 4 | 2.336 | 2.954 | 0 | 0 | 0.000 |
| BankSim | mini | 6 | 0.975 | 1.043 | 3 | 1 | 0.000 |
| BankSim | small | 6 | 1.007 | 1.041 | 3 | 3 | 0.017 |
| BankSim | base | 6 | 0.968 | 1.033 | 3 | 1 | 0.017 |
| S-FFSD | mini | 5 | 1.006 | 1.091 | 1 | 1 | 0.000 |
| S-FFSD | small | 5 | 1.039 | 1.150 | 2 | 1 | 0.080 |
| S-FFSD | base | 5 | 1.067 | 1.219 | 2 | 1 | 0.000 |
| Sapienza 15 s | mini | 8 | 1.006 | 1.063 | 3 | 3 | 0.000 |
| Sapienza 15 s | small | 8 | 0.998 | 1.059 | 5 | 3 | 0.006 |
| Sapienza 15 s | base | 8 | 0.995 | 1.056 | 5 | 3 | 0.013 |
| ULB credit card | mini | 29 | 0.844 | 1.195 | 29 | 0 | 0.060 |
| ULB credit card | small | 29 | 0.834 | 1.174 | 29 | 0 | 0.043 |
| ULB credit card | base | 29 | 0.815 | 1.156 | 29 | 0 | 0.045 |

## Detection (test rows, 1 % FPR)

| dataset | model | detector | positives | ap | auroc | recall_1pct_fpr | ap_vs_no_forecaster |
|---|---|---|---|---|---|---|---|
| K-lines (Xu & Livshits + B0R0koko) | mini | fusion_r1_kronos | 96 | 0.263 | 0.931 | 0.677 | [-0.035, +0.033] |
| K-lines (Xu & Livshits + B0R0koko) | - | fusion_r1_no_forecaster | 96 | 0.265 | 0.932 | 0.719 | nan |
| K-lines (Xu & Livshits + B0R0koko) | mini | published_pot_kronos | 96 | 0.064 | 0.890 | 0.073 | [-0.273, -0.141] |
| K-lines (Xu & Livshits + B0R0koko) | mini | kronos_nll_surprise | 96 | 0.078 | 0.903 | 0.302 | [-0.255, -0.122] |
| K-lines (Xu & Livshits + B0R0koko) | - | robust_mad | 96 | 0.682 | 0.941 | 0.781 | [+0.336, +0.497] |
| K-lines (Xu & Livshits + B0R0koko) | small | fusion_r1_kronos | 96 | 0.244 | 0.931 | 0.677 | [-0.042, -0.000] |
| K-lines (Xu & Livshits + B0R0koko) | small | published_pot_kronos | 96 | 0.094 | 0.878 | 0.208 | [-0.235, -0.113] |
| K-lines (Xu & Livshits + B0R0koko) | small | kronos_nll_surprise | 96 | 0.087 | 0.898 | 0.323 | [-0.246, -0.115] |
| K-lines (Xu & Livshits + B0R0koko) | base | fusion_r1_kronos | 96 | 0.246 | 0.929 | 0.552 | [-0.058, +0.022] |
| K-lines (Xu & Livshits + B0R0koko) | base | published_pot_kronos | 96 | 0.094 | 0.871 | 0.219 | [-0.235, -0.113] |
| K-lines (Xu & Livshits + B0R0koko) | base | kronos_nll_surprise | 96 | 0.086 | 0.898 | 0.333 | [-0.247, -0.117] |
| BankSim | mini | fusion_r1_kronos | 1440 | 0.466 | 0.872 | 0.070 | [-0.019, -0.005] |
| BankSim | - | fusion_r1_no_forecaster | 1440 | 0.478 | 0.873 | 0.088 | nan |
| BankSim | mini | published_pot_kronos | 1440 | 0.217 | 0.606 | 0.038 | [-0.281, -0.239] |
| BankSim | mini | kronos_nll_surprise | 1440 | 0.461 | 0.810 | 0.115 | [-0.044, +0.013] |
| BankSim | - | robust_mad | 1440 | 0.952 | 0.990 | 0.762 | [+0.445, +0.499] |
| BankSim | small | fusion_r1_kronos | 1440 | 0.472 | 0.872 | 0.083 | [-0.012, -0.000] |
| BankSim | small | published_pot_kronos | 1440 | 0.224 | 0.604 | 0.042 | [-0.275, -0.233] |
| BankSim | small | kronos_nll_surprise | 1440 | 0.446 | 0.802 | 0.074 | [-0.061, -0.004] |
| BankSim | base | fusion_r1_kronos | 1440 | 0.475 | 0.875 | 0.078 | [-0.009, +0.006] |
| BankSim | base | published_pot_kronos | 1440 | 0.224 | 0.601 | 0.040 | [-0.276, -0.233] |
| BankSim | base | kronos_nll_surprise | 1440 | 0.454 | 0.814 | 0.078 | [-0.052, +0.004] |
| S-FFSD | mini | fusion_r1_kronos | 2394 | 0.526 | 0.274 | 0.000 | [-0.002, +0.002] |
| S-FFSD | - | fusion_r1_no_forecaster | 2394 | 0.527 | 0.269 | 0.000 | nan |
| S-FFSD | mini | published_pot_kronos | 2394 | 0.470 | 0.393 | 0.010 | [-0.085, -0.034] |
| S-FFSD | mini | kronos_nll_surprise | 2394 | 0.428 | 0.310 | 0.001 | [-0.128, -0.073] |
| S-FFSD | - | robust_mad | 2394 | 0.684 | 0.763 | 0.000 | [+0.119, +0.195] |
| S-FFSD | small | fusion_r1_kronos | 2394 | 0.529 | 0.274 | 0.000 | [-0.002, +0.007] |
| S-FFSD | small | published_pot_kronos | 2394 | 0.460 | 0.371 | 0.010 | [-0.095, -0.042] |
| S-FFSD | small | kronos_nll_surprise | 2394 | 0.437 | 0.336 | 0.003 | [-0.116, -0.065] |
| S-FFSD | base | fusion_r1_kronos | 2394 | 0.534 | 0.282 | 0.000 | [+0.003, +0.012] |
| S-FFSD | base | published_pot_kronos | 2394 | 0.443 | 0.322 | 0.010 | [-0.112, -0.059] |
| S-FFSD | base | kronos_nll_surprise | 2394 | 0.436 | 0.331 | 0.002 | [-0.117, -0.067] |
| Sapienza 15 s | mini | fusion_r1_kronos | 64 | 0.618 | 0.992 | 0.781 | [-0.087, -0.005] |
| Sapienza 15 s | - | fusion_r1_no_forecaster | 64 | 0.662 | 0.995 | 0.797 | nan |
| Sapienza 15 s | mini | published_pot_kronos | 64 | 0.182 | 0.967 | 0.109 | [-0.634, -0.369] |
| Sapienza 15 s | mini | kronos_nll_surprise | 64 | 0.439 | 0.959 | 0.594 | [-0.395, -0.085] |
| Sapienza 15 s | - | robust_mad | 64 | 0.893 | 0.999 | 0.969 | [+0.138, +0.285] |
| Sapienza 15 s | small | fusion_r1_kronos | 64 | 0.653 | 0.994 | 0.797 | [-0.045, +0.023] |
| Sapienza 15 s | small | published_pot_kronos | 64 | 0.185 | 0.976 | 0.109 | [-0.633, -0.366] |
| Sapienza 15 s | small | kronos_nll_surprise | 64 | 0.303 | 0.956 | 0.562 | [-0.524, -0.231] |
| Sapienza 15 s | base | fusion_r1_kronos | 64 | 0.625 | 0.992 | 0.766 | [-0.080, +0.003] |
| Sapienza 15 s | base | published_pot_kronos | 64 | 0.183 | 0.976 | 0.109 | [-0.634, -0.368] |
| Sapienza 15 s | base | kronos_nll_surprise | 64 | 0.290 | 0.951 | 0.594 | [-0.536, -0.244] |
| ULB credit card | mini | fusion_r1_kronos | 77 | 0.196 | 0.913 | 0.117 | [-0.013, +0.001] |
| ULB credit card | - | fusion_r1_no_forecaster | 77 | 0.201 | 0.914 | 0.156 | nan |
| ULB credit card | mini | published_pot_kronos | 77 | 0.229 | 0.888 | 0.273 | [-0.036, +0.085] |
| ULB credit card | mini | kronos_nll_surprise | 77 | 0.427 | 0.925 | 0.494 | [+0.129, +0.330] |
| ULB credit card | - | robust_mad | 77 | 0.247 | 0.937 | 0.208 | [-0.007, +0.098] |
| ULB credit card | small | fusion_r1_kronos | 77 | 0.204 | 0.913 | 0.143 | [-0.003, +0.016] |
| ULB credit card | small | published_pot_kronos | 77 | 0.238 | 0.880 | 0.286 | [-0.029, +0.097] |
| ULB credit card | small | kronos_nll_surprise | 77 | 0.423 | 0.924 | 0.532 | [+0.133, +0.328] |
| ULB credit card | base | fusion_r1_kronos | 77 | 0.205 | 0.915 | 0.156 | [-0.003, +0.017] |
| ULB credit card | base | published_pot_kronos | 77 | 0.226 | 0.884 | 0.273 | [-0.035, +0.081] |
| ULB credit card | base | kronos_nll_surprise | 77 | 0.472 | 0.932 | 0.545 | [+0.175, +0.375] |

## Knowledge boundary (balanced accuracy; label = pump or fraud)

| dataset | model | boundary | logistic | mlp | cnn | knob_k | knob_dims | knob_acc | random_k_mean | random_at_least_knob | dlogp_tau_minus1 | dlogp_tau_plus1 | intervention_rows | passes_controls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K-lines (Xu & Livshits + B0R0koko) | mini | error | 0.573 | 0.616 | 0.611 | 1 | [193] | 0.679 | 0.580 | 0.020 | -0.002 | 0.026 | 600 | False |
| K-lines (Xu & Livshits + B0R0koko) | mini | label | 0.708 | 0.724 | 0.693 | 3 | [131, 91, 55] | 0.688 | 0.623 | 0.040 | -0.026 | 0.041 | 192 | False |
| K-lines (Xu & Livshits + B0R0koko) | small | error | 0.605 | 0.590 | 0.617 | 1 | [324] | 0.610 | 0.583 | 0.260 | 0.024 | 0.049 | 600 | False |
| K-lines (Xu & Livshits + B0R0koko) | small | label | 0.708 | 0.740 | 0.651 | 2 | [312, 307] | 0.682 | 0.632 | 0.120 | -0.476 | -0.237 | 192 | False |
| K-lines (Xu & Livshits + B0R0koko) | base | error | 0.608 | 0.632 | 0.611 | 1 | [674] | 0.645 | 0.580 | 0.020 | -0.017 | -0.073 | 600 | False |
| K-lines (Xu & Livshits + B0R0koko) | base | label | 0.734 | 0.766 | 0.646 | 1 | [391] | 0.651 | 0.581 | 0.060 | 0.056 | -0.005 | 192 | False |
| BankSim | mini | error | 0.559 | 0.557 | 0.542 | 1 | [178] | 0.562 | 0.537 | 0.040 | 0.037 | -0.300 | 200 | True |
| BankSim | mini | label | 0.653 | 0.670 | 0.653 | 1 | [39] | 0.685 | 0.618 | 0.000 | -0.978 | 0.567 | 200 | True |
| BankSim | small | error | 0.553 | 0.572 | 0.554 | 1 | [230] | 0.579 | 0.548 | 0.100 | 0.184 | -0.339 | 200 | True |
| BankSim | small | label | 0.616 | 0.668 | 0.661 | 1 | [85] | 0.695 | 0.624 | 0.040 | -0.021 | 0.023 | 200 | False |
| BankSim | base | error | 0.551 | 0.566 | 0.537 | 1 | [292] | 0.568 | 0.550 | 0.200 | 0.226 | -0.248 | 200 | True |
| BankSim | base | label | 0.626 | 0.680 | 0.648 | 1 | [240] | 0.702 | 0.641 | 0.000 | -0.393 | 0.027 | 200 | False |
| S-FFSD | mini | error | 0.572 | 0.548 | 0.487 | 1 | [254] | 0.571 | 0.497 | 0.040 | 0.930 | -0.624 | 200 | False |
| S-FFSD | mini | label | 0.577 | 0.583 | 0.355 | 1 | [22] | 0.540 | 0.509 | 0.440 | -0.016 | 0.005 | 200 | False |
| S-FFSD | small | error | 0.536 | 0.611 | 0.519 | 9 | [168, 67, 77, 398, 312, 125, 184, 508, 481] | 0.520 | 0.551 | 0.640 | 0.085 | -0.089 | 200 | True |
| S-FFSD | small | label | 0.473 | 0.637 | 0.565 | 8 | [493, 294, 205, 359, 230, 168, 83, 172] | 0.648 | 0.556 | 0.280 | 0.008 | -0.011 | 200 | False |
| S-FFSD | base | error | 0.533 | 0.572 | 0.485 | 1 | [277] | 0.488 | 0.521 | 0.800 | -0.078 | 0.042 | 200 | True |
| S-FFSD | base | label | 0.581 | 0.631 | 0.585 | 1 | [58] | 0.694 | 0.520 | 0.000 | -0.021 | 0.017 | 200 | True |
| Sapienza 15 s | mini | error | 0.580 | 0.591 | 0.570 | 2 | [177, 206] | 0.616 | 0.572 | 0.000 | 0.271 | -0.219 | 200 | True |
| Sapienza 15 s | mini | label | 0.680 | 0.758 | 0.695 | 5 | [91, 200, 19, 123, 74] | 0.734 | 0.699 | 0.140 | -0.151 | 0.200 | 128 | True |
| Sapienza 15 s | small | error | 0.543 | 0.601 | 0.578 | 1 | [404] | 0.593 | 0.563 | 0.100 | -0.146 | 0.146 | 200 | True |
| Sapienza 15 s | small | label | 0.719 | 0.680 | 0.680 | 1 | [324] | 0.742 | 0.650 | 0.080 | 0.120 | -0.128 | 128 | True |
| Sapienza 15 s | base | error | 0.575 | 0.599 | 0.561 | 1 | [247] | 0.597 | 0.573 | 0.140 | 0.018 | -0.019 | 200 | False |
| Sapienza 15 s | base | label | 0.727 | 0.703 | 0.680 | 1 | [216] | 0.773 | 0.648 | 0.000 | -0.096 | 0.093 | 128 | False |
| ULB credit card | mini | error | 0.508 | 0.462 | 0.485 | 1 | [136] | 0.563 | 0.511 | 0.000 | 0.010 | -0.434 | 100 | False |
| ULB credit card | mini | label | 0.481 | 0.494 | 0.487 | 8 | [52, 250, 46, 122, 96, 253, 133, 100] | 0.468 | 0.499 | 0.760 | 0.052 | -0.062 | 100 | False |
| ULB credit card | small | error | 0.521 | 0.495 | 0.483 | 1 | [57] | 0.531 | 0.519 | 0.240 | -0.020 | -0.018 | 100 | False |
| ULB credit card | small | label | 0.506 | 0.513 | 0.552 | 1 | [296] | 0.532 | 0.494 | 0.480 | -0.378 | 0.029 | 100 | False |
| ULB credit card | base | error | 0.502 | 0.497 | 0.514 | 5 | [203, 267, 257, 659, 443] | 0.505 | 0.508 | 0.500 | -0.094 | 0.061 | 100 | True |
| ULB credit card | base | label | 0.532 | 0.513 | 0.474 | 2 | [379, 570] | 0.474 | 0.485 | 0.600 | -0.098 | 0.086 | 100 | False |

## Recall at 1 % FPR per channel and combination (miss rate = 1 - recall)

| dataset | model | positives | recall B | recall L | recall K | recall P logistic | recall P cnn | recall B or K | B or K vs B | recall B or P | B or P vs B | recall B+K Fisher | B+K Fisher vs B | recall B or L | recall B or L or K | subtle half: B -> B or K | K recovers of B-missed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K-lines (Xu & Livshits + B0R0koko) | mini | 96 | 0.260 | 0.781 | 0.083 | 0.073 | 0.198 | 0.167 | +6/-15 [-0.178, -0.011] | 0.146 | +4/-15 [-0.206, -0.030] | 0.302 | +6/-2 [-0.011, +0.102] | 0.750 | 0.688 | 0.167 -> 0.146 [-0.125, +0.085] | 0.113 |
| K-lines (Xu & Livshits + B0R0koko) | small | 96 | 0.302 | 0.781 | 0.198 | 0.073 | 0.177 | 0.240 | +9/-15 [-0.160, +0.032] | 0.177 | +2/-14 [-0.206, -0.052] | 0.323 | +3/-1 [-0.021, +0.062] | 0.750 | 0.677 | 0.208 -> 0.167 [-0.170, +0.089] | 0.284 |
| K-lines (Xu & Livshits + B0R0koko) | base | 96 | 0.302 | 0.781 | 0.083 | 0.094 | 0.104 | 0.104 | +1/-20 [-0.274, -0.118] | 0.146 | +5/-20 [-0.253, -0.053] | 0.323 | +4/-2 [-0.023, +0.074] | 0.750 | 0.667 | 0.188 -> 0.083 [-0.188, -0.023] | 0.119 |
| BankSim | mini | 1440 | 0.115 | 0.551 | 0.120 | 0.269 | 0.260 | 0.150 | +92/-42 [+0.020, +0.050] | 0.279 | +278/-42 [+0.126, +0.200] | 0.113 | +3/-7 [-0.007, +0.001] | 0.574 | 0.578 | 0.114 -> 0.155 [+0.016, +0.065] | 0.136 |
| BankSim | small | 1440 | 0.074 | 0.551 | 0.142 | 0.283 | 0.254 | 0.138 | +131/-40 [+0.049, +0.076] | 0.278 | +333/-40 [+0.177, +0.228] | 0.125 | +114/-41 [+0.037, +0.066] | 0.561 | 0.570 | 0.073 -> 0.138 [+0.045, +0.085] | 0.153 |
| BankSim | base | 1440 | 0.078 | 0.551 | 0.174 | 0.200 | 0.190 | 0.165 | +168/-43 [+0.072, +0.102] | 0.208 | +231/-43 [+0.107, +0.151] | 0.078 | +11/-10 [-0.006, +0.007] | 0.560 | 0.578 | 0.076 -> 0.150 [+0.056, +0.093] | 0.189 |
| S-FFSD | mini | 2394 | 0.001 | 0.000 | 0.003 | 0.013 | 0.001 | 0.001 | +2/-2 [-0.002, +0.002] | 0.008 | +18/-2 [+0.001, +0.014] | 0.001 | +0/0 [+0.000, +0.000] | 0.000 | 0.001 | 0.010 -> 0.003 [-0.018, +0.000] | 0.003 |
| S-FFSD | small | 2394 | 0.003 | 0.000 | 0.009 | 0.032 | 0.012 | 0.008 | +18/-6 [-0.001, +0.013] | 0.025 | +60/-6 [+0.004, +0.053] | 0.002 | +0/-2 [-0.002, +0.000] | 0.000 | 0.008 | 0.014 -> 0.003 [-0.022, -0.003] | 0.009 |
| S-FFSD | base | 2394 | 0.002 | 0.000 | 0.102 | 0.080 | 0.005 | 0.024 | +56/-4 [+0.012, +0.032] | 0.042 | +99/-4 [+0.017, +0.069] | 0.003 | +1/0 [+0.000, +0.001] | 0.000 | 0.017 | 0.010 -> 0.003 [-0.015, +0.000] | 0.102 |
| Sapienza 15 s | mini | 64 | 0.594 | 0.906 | 0.312 | 0.297 | 0.344 | 0.797 | +16/-3 [+0.078, +0.328] | 0.797 | +15/-2 [+0.094, +0.328] | 0.656 | +4/0 [+0.016, +0.125] | 0.859 | 0.891 | 0.344 -> 0.719 [+0.219, +0.531] | 0.731 |
| Sapienza 15 s | small | 64 | 0.562 | 0.906 | 0.172 | 0.250 | 0.297 | 0.641 | +9/-4 [-0.031, +0.188] | 0.672 | +11/-4 [+0.000, +0.219] | 0.672 | +7/0 [+0.047, +0.188] | 0.859 | 0.812 | 0.344 -> 0.562 [+0.062, +0.375] | 0.393 |
| Sapienza 15 s | base | 64 | 0.594 | 0.906 | 0.094 | 0.219 | 0.328 | 0.516 | +3/-8 [-0.172, +0.016] | 0.578 | +8/-9 [-0.141, +0.109] | 0.609 | +1/0 [+0.000, +0.047] | 0.844 | 0.797 | 0.344 -> 0.406 [-0.062, +0.188] | 0.154 |
| ULB credit card | mini | 77 | 0.494 | 0.039 | 0.000 | 0.000 | 0.000 | 0.325 | +0/-13 [-0.244, -0.103] | 0.325 | +0/-13 [-0.244, -0.103] | 0.468 | +1/-3 [-0.078, +0.026] | 0.325 | 0.247 | 0.263 -> 0.184 [-0.171, +0.000] | 0.000 |
| ULB credit card | small | 77 | 0.532 | 0.039 | 0.026 | 0.013 | 0.026 | 0.429 | +1/-9 [-0.195, -0.026] | 0.429 | +1/-9 [-0.192, -0.026] | 0.519 | +0/-1 [-0.039, +0.000] | 0.416 | 0.260 | 0.289 -> 0.211 [-0.205, +0.028] | 0.028 |
| ULB credit card | base | 77 | 0.545 | 0.039 | 0.013 | 0.026 | 0.000 | 0.455 | +0/-7 [-0.156, -0.038] | 0.468 | +1/-7 [-0.143, -0.013] | 0.532 | +0/-1 [-0.039, +0.000] | 0.455 | 0.338 | 0.316 -> 0.211 [-0.211, -0.024] | 0.000 |
