# BRD/BTC 真实权重复跑（协议冻结于 2026-11-06，于 2026-11-10 用离线加载的真实权重执行）

本目录是冻结协议、代码与 2026-11-10 的真实权重结果。协议见 [PREREGISTRATION.md](PREREGISTRATION.md)、[frozen_config.json](frozen_config.json)、[frozen_manifest.json](frozen_manifest.json)。

## 结果（2026-11-10，真实 Kronos-mini + Tokenizer-2k，CPU，每根 K 线 20 次采样）

权重与官方源码从本地离线包加载（SHA256 与 `results/2026-11-06/input_manifest.json` 一致，见 `draws_*.json`）。Binance 归档仍不可达，K 线取自已发布原生实验保存的 2,880 根（`--bars-csv`，见 `DEVIATIONS.md`）：前 64 根只作历史，实际评分 2,816 根（校准 1,664、测试 1,152），不是协议写的 2,880 根。

| 终点 | 预注册判据 | 结果 | 判定 |
|---|---|---|---|
| E1 尺度塌缩 | prescale 臂四个价格维的 MAE / 同字段上一值 均 < 1.5 | 校准段 open 0.78、high 1.06、low 1.09、close 1.17（raw 臂 2.9–4.1，已发布 3.7–4.9） | **成立**：塌缩来自 (std + 1e-5) 标准化 |
| E2 技能 | log_return 或 log_range 上堆叠权重 ≥ 0.3 且 MAE 低于最优朴素预测 | prescale 臂权重 log_return 0.00、log_range 0.25、log_volume 0.30、log_amount 0.40 | **不成立**：修复后价格维仍无一步技能，量/额维有部分权重 |
| E3 预测分布 | prescale 臂 `kronos:spread` 的 pinball 系数在 ≥ 2 个特征上 > 0 | log_return 1.44、log_range 0.73、log_volume 0.00、log_amount 0.04 | **成立**：采样离散度进入尺度模型 |
| E4 检测（描述性） | — | 见下表 | pump K 线在三个变体中都被告警；零成交量告警 0 |

`forecast_skill.csv`（MAE / 同字段上一值）：

| 臂 | 分区 | open | high | low | close | volume | amount |
|---|---|---|---|---|---|---|---|
| prescale | 校准 | 0.776 | 1.056 | 1.091 | 1.170 | 0.865 | 0.824 |
| prescale | 测试 | 0.794 | 1.069 | 1.087 | 1.153 | 0.872 | 0.806 |
| raw | 校准 | 2.943 | 4.120 | 3.882 | 4.017 | 0.763 | 0.769 |
| raw | 测试 | 2.709 | 3.725 | 3.356 | 3.346 | 0.754 | 0.748 |
| 已发布 s3 | 校准 | 3.666 | 4.611 | 4.900 | 4.947 | 0.790 | 0.793 |
| 已发布 s3 | 测试 | 3.333 | 4.411 | 4.417 | 4.456 | 0.761 | 0.754 |

`detection.csv`（1,152 根测试 K 线，名义告警率 1%）：

| 臂 | 变体 | 告警 | 告警率 | 在线告警 | 零成交量告警 | pump 告警 | pump 名次 | Kronos 平均权重 | 注入功效 δ=3 / 6 / 12 |
|---|---|---|---|---|---|---|---|---|---|
| prescale | 融合 + Kronos 均值 + 离散度 | 16 | 1.4% | 12 | 0 | 是 | 3 | 0.24 | 0.18 / 0.92 / 1.00 |
| prescale | 融合 + Kronos 均值 | 24 | 2.1% | 11 | 0 | 是 | 2 | 0.24 | 0.17 / 0.95 / 1.00 |
| prescale | 融合，无 Kronos | 26 | 2.3% | 11 | 0 | 是 | 2 | 0 | 0.19 / 0.95 / 1.00 |
| raw | 融合 + Kronos 均值 + 离散度 | 8 | 0.7% | 12 | 0 | 是 | 4 | 0.36 | 0.21 / 0.79 / 1.00 |
| raw | 融合 + Kronos 均值 | 25 | 2.2% | 11 | 0 | 是 | 2 | 0.36 | 0.17 / 0.95 / 1.00 |
| raw | 融合，无 Kronos | 26 | 2.3% | 11 | 0 | 是 | 2 | 0 | 0.19 / 0.95 / 1.00 |

读法：prescale 把价格预测从"比上一值差 3–5 倍"修到"与上一值相当"（open 略好，high/low/close 略差），但这只是消除了一个预处理缺陷，没有带来一步预测技能——与原生 K 线数据集（`results/2026-11-10/detector_study_xu_livshits_1h`）上的发现一致。采样离散度是真实的信息（E3），用它的变体告警最少、注入功效略低；pump K 线的名次在 2–4 之间，与是否使用 Kronos 无关。已发布流程的 54 次告警（其中 14 次在零成交量 K 线上）在三个变体中都降到 ≤ 26 次、零成交量 0 次。

## Results in brief (English)

Real weights, loaded offline, 20 draws per bar, bars from the published run (first 64 as history; 2,816 scored instead of 2,880, see `DEVIATIONS.md`).  E1 **holds**: with `prescale` the OHLC forecast MAE falls from 2.9–4.9 × persistence to 0.78–1.17 ×, so the published failure was the `(std + 1e-5)` normalization flattening a 5e-5-priced series.  E2 **fails**: the stacking weight on log return / log range is 0.00 / 0.25 (< 0.3), i.e. no one-step price skill even after the fix; volume / amount get 0.30 / 0.40.  E3 **holds**: the spread of the 20 draws gets positive scale coefficients on log return (1.44) and log range (0.73).  E4: 16 / 24 / 26 alarms at nominal 1% (with spread / mean only / no Kronos; published 54), 0 on zero-volume bars (published 14), the pump bar alarmed in every variant (rank 3 / 2 / 2), pump-type injection power 0.18 / 0.92 / 1.00 at δ = 3 / 6 / 12.

## 要回答的问题

1. **E1 尺度塌缩**：已保存的原生实验中，Kronos 的 OHLC 预测误差是“同字段上一值”的 3.7–5.0 倍。假设成因是官方预测器的 (std + 1e-5) 标准化把 5e-5 量级的价格压平。`prescale` 臂先按窗口 std 缩放再送入官方预测器；若四个价格维的 MAE 比值都降到 1.5 以下，假设成立。
2. **E2 技能**：修复后 Kronos-mini 在 log 收益 / log 高低幅上是否有一步预测技能（堆叠权重 ≥ 0.3 且 MAE 低于最优朴素预测）。
3. **E3 预测分布**：保留 20 次采样，其在特征空间的稳健离散度作为条件尺度模型的一项；系数是否为正。
4. **E4 检测**（描述性，只有 1 根标注 K 线）：1% 名义告警率下的告警数、零成交量 K 线告警、pump K 线名次、pump 型注入功效，有/无 Kronos。

## 需要的输入（SHA256 由 `results/2026-11-06/input_manifest.json` 核对）

| 路径 | 来源 |
|---|---|
| `vendor/Kronos/model/{__init__,kronos,module}.py` | 官方仓库 commit `67b630e`（本会话已取得并核对） |
| `weights/Kronos-mini/{config.json,model.safetensors}` | https://huggingface.co/NeoQuasar/Kronos-mini |
| `weights/Kronos-Tokenizer-2k/{config.json,model.safetensors}` | https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k |
| `runs/market/BRDBTC-5m-2018-12.zip{,.CHECKSUM}` | https://data.binance.vision/data/spot/monthly/klines/BRDBTC/5m/ |

## 运行

```bash
pip install -e ".[test,experiments,kronos]"
PYTHONPATH=.:scripts python scripts/run_native_v2.py --device cpu     # 或 mps / cuda
PYTHONPATH=.:scripts python scripts/run_native_v2.py --score-only     # 用缓存的采样重放评分，不需要 Torch
python -m pytest -q tests/test_sampling.py tests/test_native_v2.py
```

两个臂（`prescale`、`raw`）各做 2,880 次单步预测、每次 20 个采样，采样缓存在 `draws_<arm>.npz`，附参数与权重哈希。输出 `forecast_skill.csv`（E1/E2）、`detection.csv`（E3/E4）、`native_rows_<arm>.csv.gz`。

`kronos_ad/sampling.py` 复用官方 tokenizer/模型/采样函数，只是不对采样取平均；`tests/test_sampling.py` 用随机初始化的小模型验证：同一种子下，保留采样的均值与官方 `predict_batch` 输出逐位一致。
