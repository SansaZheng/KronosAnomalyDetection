# BRD/BTC 真实权重复跑（已冻结，尚未执行）

本目录是下一步实验的冻结协议与代码；**没有任何推理结果**，因为本会话无法访问 Hugging Face 与 Binance。协议见 [PREREGISTRATION.md](PREREGISTRATION.md)、[frozen_config.json](frozen_config.json)、[frozen_manifest.json](frozen_manifest.json)。

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
