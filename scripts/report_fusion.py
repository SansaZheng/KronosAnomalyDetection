#!/usr/bin/env python3
"""Render results/2026-11-06/kronos_fusion/README.md from the saved tables."""
import json

import numpy as np
import pandas as pd

from fusion_data import NATIVE, PROXY_NAMES, RESULTS
from run_fusion import OUT

FROZEN = json.loads((OUT / 'frozen_config.json').read_text())
PRIMARY_MIN = FROZEN['endpoints']['primary_datasets_min_test_positives']
LABEL = {'fusion': '证据融合（Kronos+朴素）', 'fusion_no_kronos': '证据融合（无 Kronos）', 'per_feature_pot': '逐维 POT（已发布）',
         'joint_top3': '联合 top3（已发布）', 'isolation_forest': 'Isolation Forest', 'robust_mad': 'Robust MAD',
         'fusion_online': '证据融合 + 在线阈值', 'fusion_no_kronos_online': '无 Kronos + 在线阈值'}


def f(v, digits=3):
    return 'NA' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{digits}f}'


def pct(v):
    return 'NA' if np.isnan(v) else f'{100 * v:.2f}%'


def main():
    metrics = pd.read_csv(OUT / 'metrics.csv')
    boot = pd.read_csv(OUT / 'bootstrap.csv').set_index(['dataset', 'method'])
    skill = pd.read_csv(OUT / 'forecast_skill.csv')
    power = pd.read_csv(OUT / 'injection_power.csv')
    coverage = pd.read_csv(OUT / 'full_coverage.csv')
    online = pd.read_csv(OUT / 'posthoc_online_gamma.csv')
    native = json.loads((OUT / 'native_summary.json').read_text())
    zero = json.loads((OUT / 'posthoc_zero_volume_audit.json').read_text())
    weights = json.loads((OUT / 'stack_weights.json').read_text())
    m = metrics.set_index(['dataset', 'method'])
    positives = {n: int(m.loc[(n, 'fusion'), 'positives']) for n in [NATIVE] + PROXY_NAMES}
    primary = [n for n in PROXY_NAMES if positives[n] >= PRIMARY_MIN]
    secondary = [n for n in [NATIVE] + PROXY_NAMES if n not in primary]

    L = ['# Kronos 证据融合：冻结协议下的回顾性实验（2026-11-06）', '',
         '本目录在**不新增任何 Kronos 推理**的前提下，复用仓库已保存的真实 Kronos-mini 预测（原生 BRD/BTC 2,880 根 K 线，以及 10 个代理版本共 126,592 次逐特征预测），重新设计误差评分、校准与阈值，并与已发布结果在**完全相同的测试位置**上比较。协议在首次接触测试行之前冻结，见 [PREREGISTRATION.md](PREREGISTRATION.md)、[frozen_config.json](frozen_config.json) 与 [frozen_manifest.json](frozen_manifest.json)（代码 SHA256）。', '',
         '## 结论', '',
         f'1. **评分与校准的改造有效。** 在 {len(primary)} 个主要数据集（测试正例 ≥ {PRIMARY_MIN}）上，证据融合的 AP 均高于已发布的逐维 POT；配对 bootstrap 95% 区间全部不含 0，但重采样单元很少（sd_prices 4 个 case、sd_stock 6 个 case、s_ffsd 7 个块），sd_prices 的区间只建立在 4 个 case 上（其中 3 个为正），应视为描述性结果。相对已发布的联合 top3，{sum(boot.loc[(n, "joint_top3"), "ap_diff_lo"] > 0 for n in primary)}/{len(primary)} 个区间不含 0。1% 名义告警率下的实际告警率落在 {pct(m.loc[[(n, "fusion") for n in PROXY_NAMES + [NATIVE]], "alarm_rate"].min())}–{pct(m.loc[[(n, "fusion") for n in PROXY_NAMES + [NATIVE]], "alarm_rate"].max())}，已发布逐维 POT 为 {pct(m.loc[[(n, "per_feature_pot") for n in PROXY_NAMES + [NATIVE]], "alarm_rate"].min())}–{pct(m.loc[[(n, "per_feature_pot") for n in PROXY_NAMES + [NATIVE]], "alarm_rate"].max())}。',
         '2. **Kronos 预测没有正向贡献。** 同一管线去掉 Kronos、只用“上一值”和“窗口中位数”两种朴素预测，AP 的配对差异区间全部含 0；唯一可测量的影响是负的：BankSim 上 1% FPR 召回从无 Kronos 的 ' + f(m.loc[("banksim", "fusion_no_kronos"), "recall_1pct_fpr"]) + ' 降到有 Kronos 的 ' + f(m.loc[("banksim", "fusion"), "recall_1pct_fpr"]) + '（配对差异区间 [' + f(boot.loc[("banksim", "fusion_no_kronos"), "r1_diff_lo"]) + ', ' + f(boot.loc[("banksim", "fusion_no_kronos"), "r1_diff_hi"]) + ']）。L1 最优堆叠给 Kronos 的平均权重为 ' + '、'.join(f'{w["dataset"]} {w["mean_weight"][-1]:.2f}' for w in weights) + '。按各维中位数计，Kronos 一步预测的 MAE 在每个数据集上都高于最好的朴素预测（原生实验的量、额两维是例外，见表 2）。',
         '3. **原生实验的价格预测存在尺度塌缩。** 官方预测器按 (窗口 std + 1e-5) 标准化输入；BRD/BTC 价格约 5e-5，64 根窗口的价格 std 中位数约 2.6–2.8e-7，模型实际看到的价格序列有效跨度中位数只有 0.11 个标准化单位，输出再乘回 1e-5 量级的常数，与真实波动无关。校准段内 Kronos 的 OHLC 预测 MAE 是“同字段上一值”的 3.7–5.0 倍。`kronos_ad/backend.py` 新增 `prescale`（先按窗口 std 缩放再送入官方预测器，默认开启），有单元测试，但**本轮无权重，无法重跑验证其检测效果**。',
         '4. **水平型基线并未被超越。** Isolation Forest / Robust MAD 在 BankSim、SD-stock 上更好，Robust MAD 在 S-FFSD 上更好（表 1 配对区间不含 0）；融合管线在 ULB、SD-crypto、SD-prices、SD-synthetic 上点估计更高（表 4，全覆盖，无区间）。预测残差类方法擅长一步突变，不擅长持续抬升的水平型异常。',
         f'5. **原生 BRD/BTC：** 融合在 1,152 根测试 K 线上告警 {native["fusion_alarms"]} 次（已发布 {native["original_alarms"]} 次），标注 pump K 线在全部测试 K 线中排第 {native["pump_rank_fusion"]}（已发布第 {native["pump_rank_original"]}），16:55–17:15 形成连续 5 根告警。已发布 54 次告警中有 {zero["original_price_only_alarms_on_zero_volume"]} 次落在**零成交量**（没有任何成交）的 K 线上且全部只由价格维触发；融合为 {zero["fusion_alarms_on_zero_volume"]} 次。',
         '6. **S-FFSD 上融合分数是反向的，不是随机的**：AUROC ' + f(m.loc[("s_ffsd", "fusion"), "auroc"]) + '（全覆盖 ' + f(coverage[(coverage.dataset == "s_ffsd") & (coverage.method == "fusion_no_kronos")].auroc.iloc[0]) + '），欺诈行的分数系统性低于正常行，AP≈正例占比 0.39 只是巧合；Robust MAD 在同样的行上 AUROC ' + f(m.loc[("s_ffsd", "robust_mad"), "auroc"]) + '。此外该数据集的固定阈值由两条极端验证行决定（频率编码的 Location 特征在 `local` 通道上的偏离，尺度下限使其 −log p 达到 7.8e3–9.2e3），测试段没有任何行超过它，0.00% 的告警率是方法在频率编码类别特征上的失败，不是告警控制。该数据集相对已发布方法的“提升”不应视为成功。', '',
         '## 方法：证据融合', '',
         '实现见 `kronos_ad/fusion.py`，所有步骤只在拟合分区上估计，阈值只在验证分区上取，测试期间全部冻结：', '',
         '1. **预测堆叠**：每个特征上，对 {Kronos, 上一值, 窗口中位数} 取 L1 最优凸组合（步长 0.05 的单纯形网格）。没有技能的预测器自动得到接近 0 的权重，退化预测不再主导残差。',
         '2. **条件尺度**：用 0.75 分位 pinball 回归把 |残差| 回归到窗口的 MAD、步长 MAD、std、步长 std 和参考尺度上（系数非负）。波动聚集由数据决定，而不是直接除以窗口 std。',
         '3. **三条证据通道**：`forecast`（一步残差，捕捉起点）、`local`（相对窗口中位数的偏离，异常进入窗口后仍保持）、`global`（相对参考分布的偏离，只对平稳特征开启；SD-FMM 各 case 已被整体 min-max 归一化，故对其关闭）。',
         '4. **保形 p 值 + GPD 尾部**：每个特征-通道的非一致性分数在拟合分区上转换为保形 p 值，超过 0.9 分位后用 Hosking–Wallis 矩估计的 GPD 延伸，使排序在 1/n 之外仍连续；形状参数限制在 [0, 0.75]。',
         '5. **融合**：行分数 = −log min p（最强证据，对数域计算避免下溢），并返回触发的特征与通道用于定位。',
         '6. **一个联合阈值**：验证分区分数的 ⌈(n+1)(1−α)⌉ 次序统计量，α=0.01。逐维“或”随维数膨胀的问题由此消失。',
         '7. **在线阈值（可选）**：自适应保形（Gibbs & Candès 2021）跟踪 1% 告警预算；本轮结果见“事后分析”。', '',
         '不属于本方法的部分：校准标签用于设计阶段的全局选择（通道组合、尺度项、聚合方式），没有按数据集调参；测试标签只在评估时读取。排除的特征：' + '、'.join(FROZEN['excluded_from_detection']) + '（时间编码与品种代码）。', '',
         '## 协议', '',
         '- 测试位置、校准位置、拟合/验证二分与 [kronos_tuning](../kronos_tuning/README.md) 完全一致（每版本 ≤512 校准、≤1024 测试，分组保持完整）。原生实验：校准段前半拟合（去掉 65 根预热后 831 根）、后半验证（832 根），1,152 根测试；量、额的对数单位取自校准段前 864 根（全部在拟合分区内）。',
         '- 主要终点：AP 与 1% FPR 下的召回；主要比较：融合 vs 已发布逐维 POT、vs 已发布联合 top3、vs 去掉 Kronos 的同一管线。配对 bootstrap 2,000 次（按整组重采样，单一流按 64 行移动块），seed 20261106。',
         f'- 主要数据集（测试正例 ≥ {PRIMARY_MIN}）：' + '、'.join(primary) + '；其余只作描述。',
         '- 已发布的旧方法测试指标在研究仓库时已被看到，因此测试位置不是严格意义上的“从未接触”；新方法自身的测试结果在冻结前未计算过。',
         '- 冻结之后的改动全部列在“冻结后改动”一节，`frozen_manifest.json` 保留冻结时的 SHA256 并记录每次修订（`scripts/verify_fusion.py` 检查当前修订）。', '',
         '## 表 1：相同测试位置上的结果', '']
    L += ['| 数据集 | 正例/已知 | 方法 | AP [95% CI] | R@1%FPR | AUROC | 实际告警率 | F1 |', '|---|---:|---|---|---:|---:|---:|---:|']
    order = ['fusion', 'fusion_no_kronos', 'per_feature_pot', 'joint_top3', 'isolation_forest', 'robust_mad']
    for n in primary + secondary:
        for method in order:
            if (n, method) not in m.index:
                continue
            r = m.loc[(n, method)]
            ci = f' [{f(boot.loc[(n, method), "ap_lo"])}, {f(boot.loc[(n, method), "ap_hi"])}]' if (n, method) in boot.index else ''
            L.append(f'| {n} | {positives[n]}/{int(r.known)} | {LABEL[method]} | {f(r.ap)}{ci} | {f(r.recall_1pct_fpr)} | {f(r.auroc)} | {pct(r.alarm_rate)} | {f(r.f1)} |')
    L += ['', '配对差异（融合 − 对照，AP，95% bootstrap 区间；正值表示融合更好；“单元”为重采样单元数：整组或 64 行块，少于 10 个单元的区间不可靠）：', '',
          '| 数据集 | 单元 | vs 逐维 POT | vs 联合 top3 | vs 无 Kronos | vs Isolation Forest | vs Robust MAD |', '|---|---:|---|---|---|---|---|']
    for n in primary:
        cells = [str(int(boot.loc[(n, 'fusion'), 'units']))]
        for method in ['per_feature_pot', 'joint_top3', 'fusion_no_kronos', 'isolation_forest', 'robust_mad']:
            lo, hi = boot.loc[(n, method), ['ap_diff_lo', 'ap_diff_hi']]
            mark = '**' if lo > 0 or hi < 0 else ''
            cells.append(f'{mark}[{lo:+.3f}, {hi:+.3f}]{mark}')
        L.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    L += ['', '![表1图](fusion_labelled.png)', '',
          '只有 1–3 个正例的 ULB、Sapienza-25s、SD-crypto 和原生 BRD/BTC 不做统计比较；Sapienza-5s/15s 测试子集无正例。',
          '1% FPR 召回：融合在 ' + '、'.join(n for n in primary if m.loc[(n, 'fusion'), 'recall_1pct_fpr'] > m.loc[(n, 'per_feature_pot'), 'recall_1pct_fpr']) + ' 上高于逐维 POT，'
          + '、'.join(n for n in primary if m.loc[(n, 'fusion'), 'recall_1pct_fpr'] < m.loc[(n, 'per_feature_pot'), 'recall_1pct_fpr']) + ' 上更低，其余持平。', '',
          '## 表 2：Kronos 一步预测技能（正常行，MAE 比值，<1 表示 Kronos 更好）', '',
          '| 数据集 | 分区 | 行数 | 维数 | Kronos/上一值 | Kronos/窗口中位数 | Kronos/最优朴素 | Kronos 胜出维数占比 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for _, r in skill.iterrows():
        L.append(f'| {r.dataset} | {r.part} | {int(r.rows)} | {int(r.dims)} | {f(r.kronos_vs_last)} | {f(r.kronos_vs_median)} | {f(r.kronos_vs_best_naive)} | {f(r.share_dims_kronos_beats_best, 2)} |')
    L += ['', '比值取各维中位数。原生实验用 log 收益、log 高低幅、log1p 成交量、log1p 成交额四个平稳特征；代理实验保留 1 次采样、32 窗口的原设置。', '',
          '## 表 3：反事实注入功效', '',
          '在标签为正常的测试行上，把 20% 的维度（至少 1 维）按步长 MAD 的 δ 倍随机正负扰动（3 个种子平均；原生实验改为收益、幅度、量、额同向的 pump 型扰动）。因为 t 时刻的预测只依赖 t 之前的历史，扰动 x_t 不影响预测，可以直接比较评分器。功效 = 扰动后超过该方法自身未扰动分数 99% 分位的比例。', '',
          '| 数据集 | δ | 逐维 POT | 仅 Kronos | 仅上一值 | 仅中位数 | 堆叠（含 Kronos） | 堆叠（无 Kronos） | 融合 | 融合（无 Kronos） |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cols = ['per_feature_pot', 'kronos_only', 'last_only', 'median_only', 'stacked_with_kronos', 'stacked_no_kronos', 'fusion', 'fusion_no_kronos']
    pw = power.pivot_table(index=['dataset', 'delta'], columns='method', values='power')
    for n in [NATIVE] + PROXY_NAMES:
        for delta in (3, 6):
            r = pw.loc[(n, delta)]
            L.append(f'| {n} | {delta} | ' + ' | '.join(f(r[c]) for c in cols) + ' |')
    wins3 = sum(pw.loc[(n, 3), 'fusion'] > pw.loc[(n, 3), 'per_feature_pot'] for n in [NATIVE] + PROXY_NAMES)
    k3 = pw.xs(3, level='delta')
    L += ['', f'δ=3 时融合功效高于已发布逐维 POT 的数据集：{wins3}/11；“仅 Kronos”高于“仅上一值”{int((k3.kronos_only > k3.last_only).sum())}/11、高于“仅中位数”{int((k3.kronos_only > k3.median_only).sum())}/11，但堆叠后有无 Kronos 的功效相当。已发布逐维 POT 在 Sapienza 上的 max 包含本方法排除的 5 个时间/品种维度；只用相同特征重算的逐维 POT（injection_power.csv 中 `per_feature_pot_same_features`，δ=3：' + '、'.join(f'{n} {f(pw.loc[(n, 3), "per_feature_pot_same_features"])}' for n in ['sapienza_5s', 'sapienza_15s', 'sapienza_25s']) + '）几乎不变，差距来自评分而非特征选择。完整 δ∈{2,3,4,6,8,12} 见 injection_power.csv 与下图。', '',
          '![注入功效](fusion_injection_power.png)', '',
          '## 表 4：全覆盖（无 Kronos）对比已发布基线', '',
          '代理实验只抽样了每版本 ≤1024 个测试位置。去掉 Kronos 后同一管线不需要推理，可以在**每一个拥有 32 行组内历史的测试行**上评分，与已发布的 Isolation Forest / Robust MAD 在相同行上比较（基线分数直接取自已发布 detections 文件）。', '',
          '| 数据集 | 可评分测试行 / 已发布测试行 | 正例 | 方法 | AP | R@1%FPR | AUROC | 实际告警率 | F1 |', '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for n in PROXY_NAMES:
        for method in ['fusion_no_kronos', 'IsolationForest', 'RobustMAD']:
            r = coverage[(coverage.dataset == n) & (coverage.method == method)].iloc[0]
            L.append(f'| {n} | {int(r.test_rows_scorable)} / {int(r.test_rows_published)} | {int(r.positives)} | {LABEL.get(method, method)} | {f(r.ap)} | {f(r.recall_1pct_fpr)} | {f(r.auroc)} | {pct(r.alarm_rate)} | {f(r.f1)} |')
    better = [n for n in PROXY_NAMES if coverage[(coverage.dataset == n) & (coverage.method == 'fusion_no_kronos')].ap.iloc[0] > coverage[(coverage.dataset == n) & (coverage.method.isin(['IsolationForest', 'RobustMAD']))].ap.max()]
    L += ['', 'AP 点估计高于两种基线：' + '、'.join(better) + '；其余数据集基线更好。本表没有区间。SD-FMM 的每个 case 都被整体 min–max 归一化到 [0,1]（最大值常常落在标注异常内），水平型检测器因此获得在线场景下不存在的优势；融合管线在这些数据上关闭了 `global` 通道。', '',
          '![全覆盖](fusion_full_coverage.png)', '',
          '## 原生 BRD/BTC 审计', '',
          f'| 项目 | 已发布逐维 POT | 证据融合 |', '|---|---:|---:|',
          f'| 测试告警 / 1,152 | {native["original_alarms"]} | {native["fusion_alarms"]} |',
          f'| 其中只由价格维触发 | {native["original_price_only"]} | — |',
          f'| 落在零成交量 K 线上的告警 | {zero["original_alarms_on_zero_volume"]} | {zero["fusion_alarms_on_zero_volume"]} |',
          f'| 标注 pump K 线（2018-12-30 17:00 UTC）是否告警 | {"是" if native["pump_original_alarm"] else "否"} | {"是" if native["pump_fusion_alarm"] else "否"} |',
          f'| 该 K 线在全部测试 K 线中的分数名次 | {native["pump_rank_original"]} | {native["pump_rank_fusion"]} |',
          f'| 触发该 K 线的证据 | — | {native["pump_top_feature"]} / {native["pump_top_channel"]} |', '',
          '零成交量 K 线占测试段 ' + f'{100 * native["zero_volume_share_test"]:.1f}%' + '，这些 K 线没有发生任何成交，价格维告警只能来自退化的预测。pump K 线由 `global` 通道（成交额相对参考分布的水平）触发，而不是预测残差。堆叠权重（上一值/中位数/Kronos）：' + '；'.join(f'{k} {v}' for k, v in native['weights'].items()) + '。', '',
          '![原生时间线](fusion_native_timeline.png)', '',
          '## 事后分析（冻结之后，按预注册规则列出）', '',
          '**在线阈值步长。** 冻结配置 γ=0.005 取自自适应保形文献，那里的 α 是 0.1。α=0.01 时一次告警使水平从 0.01 降到 0.00505，恢复需要约 100 个安静行；恢复期内再出现 2 次告警，水平降到 0 以下，阈值变为 +∞（“失明”），直到水平回到正数。下表给出不同 γ 的实际告警率与失明行数：', '',
          '| 数据集 | 固定阈值 | γ=0.005（冻结） | γ=0.001 | γ=0.0005 | γ=0.0001 |', '|---|---:|---:|---:|---:|---:|']
    for n in [NATIVE] + PROXY_NAMES:
        sub = online[online.dataset == n].set_index('gamma')
        cells = [pct(sub.loc['fixed', 'alarm_rate'])] + [f'{pct(sub.loc[str(g), "alarm_rate"])}（失明 {int(sub.loc[str(g), "blind_rows"])}）' for g in (0.005, 0.001, 0.0005, 0.0001)]
        L.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    sub = online[online.dataset == NATIVE].set_index('gamma')
    L += ['', f'原生 pump K 线：固定阈值告警；γ=0.005 与 0.001 不告警（刚好处于失明期），γ≤0.0005 告警。较小的 γ 消除失明，但在 1,024 行测试内无法把告警率拉回 1%（2–3%）。**结论：在线变体在 α=0.01、千行量级测试下不可用；表 1 的主结果使用固定阈值。** 表 1 中 `_online` 行保留冻结配置的数字。', '',
          '## 冻结后改动（按预注册规则 1 列出）', '',
          '| 时间（UTC+8） | 文件 | 改动 | 对结果的影响 |', '|---|---|---|---|',
          '| 2026-11-06 22:06 | scripts/run_fusion.py | 增加 `save_rows`（行级输出）与 `source_row` 列 | 无：指标在 `methods` 构造之后才写出；改动前生成的 posthoc_online_gamma.csv 与改动后的 metrics.csv 逐值相同（独立审计核对） |',
          '| 2026-11-06 22:40 | kronos_ad/fusion.py | p 值改为对数域：`TailPValue` 返回 log p，分数 = −log min p | 审计发现 S-FFSD 两条验证行的 p 值下溢到 1e-300 下限，分数并列为 1e300，阈值不可触发。改动后：表 1 所有数据集的 AP/AUROC/召回/告警率/F1 与改动前逐值相同（单调变换保持排序；改动前数值保存在 before_logp/）；S-FFSD 阈值由 ∞ 变为 7771.2，测试段仍无行超过，告警率两次均为 0.00%；其他阈值为原值的自然对数（原生 97.65 → 4.58） |',
          '| 2026-11-06 22:40 | scripts/run_fusion.py | bootstrap.csv 增加 `units` 列；增加 `per_feature_pot_same_features` 对照（只对 Sapienza 有效） | 不改变任何已有数字 |',
          '| 2026-11-06 22:40 | kronos_ad/backend.py, kronos_ad/cli.py | 丢弃官方预测器合成的量/额列；新增 `--no-prescale` | 与本实验无关（无推理） |',
          '| 2026-11-06 22:40 | 本 README | 结论 2、4、6，在线阈值机制，尺度塌缩数字，拟合行数按审计修正 | 文字 |', '',
          '审计还指出：评估脚本 scripts/run_fusion.py 不在冻结清单内，`frozen_manifest.json` 的修订记录已补入全部脚本。', '',
          '## 局限', '',
          '- 没有新的 Kronos 推理：无法测试多次采样、更长上下文、`prescale` 修复或 Kronos-small/base。表 2 的“无技能”结论只对本仓库保存的预测成立（原生：mini、64 窗口、3 次采样、价格尺度塌缩；代理：域外表示、1 次采样、32 窗口）。',
          '- 测试位置已在旧实验中被查看；这是回顾性比较，不是独立前瞻验证。',
          '- 校准标签参与了全局设计选择（通道、尺度项、max 聚合）；没有逐数据集调参，但设计不是完全无监督。',
          '- 主要数据集只有 5 个，正例 17–154 个，区间很宽；Sapienza 三个分辨率、SD 四个版本互相不独立。',
          '- 2,000 次 bootstrap 的块长/分组是预先固定的，不处理序列相关之外的依赖；单元数见表 1。',
          '- 频率编码的类别特征（S-FFSD 的 Location/Source 等）不适合 `local` 通道：窗口内常量会使尺度落到下限，一次类别切换被放大成极端证据。', '',
          '## 文件', '',
          '| 文件 | 内容 |', '|---|---|',
          '| metrics.csv / bootstrap.csv | 表 1 及配对区间 |', '| `*__rows.csv.gz` | 每个数据集拟合/验证/测试行的分数、阈值、标签，以及测试行上的对照方法分数 |',
          '| forecast_skill.csv | 表 2 |', '| injection_power.csv | 表 3（全部 δ 与方法） |', '| full_coverage.csv | 表 4 |',
          '| native_rows.csv.gz / native_summary.json | 原生逐 K 线分数、告警、触发特征与通道 |',
          '| stack_weights.json | 各数据集堆叠权重与阈值 |', '| posthoc_online_gamma.csv / posthoc_zero_volume_audit.json | 事后分析 |',
          '| before_logp/ | 对数域改动前的 metrics/bootstrap/injection/coverage/native/posthoc 文件 |',
          '| frozen_config.json / frozen_manifest.json / PREREGISTRATION.md | 冻结协议与修订记录 |', '',
          '## 复现', '', '```bash', 'pip install -e ".[test,experiments]"', 'PYTHONPATH=.:scripts python scripts/run_fusion.py',
          'PYTHONPATH=.:scripts python scripts/posthoc_fusion.py', 'PYTHONPATH=.:scripts python scripts/plot_fusion.py',
          'PYTHONPATH=.:scripts python scripts/report_fusion.py', 'PYTHONPATH=.:scripts python scripts/verify_fusion.py', 'python -m pytest -q', '```', '',
          '不需要 Torch 或模型权重；全部输入是仓库内已保存的预测缓存与原始数据。']
    (OUT / 'README.md').write_text('\n'.join(L) + '\n')
    print('wrote', OUT / 'README.md', len(L), 'lines')


if __name__ == '__main__':
    main()
