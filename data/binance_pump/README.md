# Binance pump-and-dump 原生 K 线基准（数据待下载）

目的：把原生 OHLCVA 评估从 1 个标注事件扩展到 Sapienza 列表中 **520 个 Binance 事件**（85 个币种，294 个月度档案）。`events.csv` 为事件表，`urls.txt` 为档案与 CHECKSUM 的下载清单（约 300 个 ZIP，预计 < 100 MB）。

```bash
wget -q -P data/binance_pump/raw -i data/binance_pump/urls.txt
python scripts/binance_pump_manifest.py --verify
```

已退市交易对的档案可能返回 404，`--verify` 会列出缺失项；缺失事件直接剔除，不做替代。

## 拟定协议（数据到位后冻结，再写评估代码）

- 每个事件：参考段 = 公告日前第 21 天至前第 7 天；校准段 = 前 6 天（到公告日 00:00 UTC）；测试段 = 公告当日 288 根 K 线。全部按 K 线开盘时间，只用公告前的历史拟合与取阈值。
- 标签：覆盖公告分钟的 K 线及其后 2 根为正；其前 12 根、后 12 根为未知（不参与评估）；当日其余为负。
- 指标：事件级命中（1% 告警率下 pump K 线是否告警）、pump K 线在当日的名次、汇总 AP、每日误报数；按年份分层报告；配对 bootstrap 按事件重采样。
- 方法：证据融合（有/无 Kronos，`prescale` 开）、已发布逐维 POT、Robust MAD。模型推理量：520 × 288 测试 + 校准 ≈ 1.2M 次单步预测，建议在 GPU/MPS 上运行，或先抽样 100 个事件。
