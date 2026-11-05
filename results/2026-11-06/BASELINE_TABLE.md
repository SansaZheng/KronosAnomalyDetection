# 无监督基线检测结果

| Dataset | Model | N evaluated | N positive | Precision | Recall | F1 | AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ulb_creditcard | IsolationForest | 60140 | 77 | 0.031 | 0.234 | 0.055 | 0.027 |
| ulb_creditcard | RobustMAD | 60140 | 77 | 0.008 | 0.039 | 0.013 | 0.020 |
| banksim | IsolationForest | 133231 | 1440 | 0.576 | 0.566 | 0.571 | 0.595 |
| banksim | RobustMAD | 133231 | 1440 | 0.830 | 0.545 | 0.658 | 0.729 |
| s_ffsd | IsolationForest | 6159 | 2394 | 0.068 | 0.002 | 0.003 | 0.398 |
| s_ffsd | RobustMAD | 6159 | 2394 | 0.036 | 0.001 | 0.002 | 0.540 |
| sapienza_5s | IsolationForest | 131401 | 64 | 0.033 | 0.984 | 0.064 | 0.541 |
| sapienza_5s | RobustMAD | 131401 | 64 | 0.043 | 0.984 | 0.082 | 0.492 |
| sapienza_15s | IsolationForest | 97179 | 64 | 0.048 | 1.000 | 0.091 | 0.230 |
| sapienza_15s | RobustMAD | 97179 | 64 | 0.052 | 0.969 | 0.099 | 0.671 |
| sapienza_25s | IsolationForest | 82445 | 64 | 0.061 | 1.000 | 0.115 | 0.526 |
| sapienza_25s | RobustMAD | 82445 | 64 | 0.067 | 0.984 | 0.126 | 0.758 |
| sd_crypto | IsolationForest | 10880 | 68 | 0.229 | 0.603 | 0.332 | 0.248 |
| sd_crypto | RobustMAD | 10880 | 68 | 0.312 | 0.574 | 0.404 | 0.260 |
| sd_stock | IsolationForest | 960 | 91 | 0.895 | 0.187 | 0.309 | 0.660 |
| sd_stock | RobustMAD | 960 | 91 | 0.880 | 0.242 | 0.379 | 0.681 |
| sd_prices | IsolationForest | 640 | 76 | 0.188 | 0.039 | 0.065 | 0.167 |
| sd_prices | RobustMAD | 640 | 76 | 0.240 | 0.079 | 0.119 | 0.195 |
| sd_synthetic | IsolationForest | 32000 | 1573 | 0.974 | 0.241 | 0.386 | 0.692 |
| sd_synthetic | RobustMAD | 32000 | 1573 | 0.962 | 0.289 | 0.445 | 0.700 |
