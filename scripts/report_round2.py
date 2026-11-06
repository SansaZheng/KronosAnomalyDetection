#!/usr/bin/env python3
"""Render results/2026-11-07/kronos_fusion_r2/README.md from the saved tables (round 2 vs round 1)."""
import json

import numpy as np
import pandas as pd

import run_fusion
from fusion_data import NATIVE, PROXY_NAMES

LABEL = {'fusion': '第 2 轮证据融合', 'fusion_r1': '第 1 轮证据融合', 'fusion_no_kronos': '第 2 轮（无 Kronos）',
         'fusion_ensemble': '集成臂（+Isolation Forest 行证据）', 'per_feature_pot': '逐维 POT（已发布）',
         'joint_top3': '联合 top3（已发布）', 'isolation_forest': 'Isolation Forest', 'robust_mad': 'Robust MAD'}
ORDER = ['fusion', 'fusion_r1', 'fusion_no_kronos', 'fusion_ensemble', 'per_feature_pot', 'joint_top3', 'isolation_forest', 'robust_mad']


def f(v, digits=3):
    return 'NA' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{digits}f}'


def pct(v):
    return 'NA' if np.isnan(v) else f'{100 * v:.2f}%'


def interval(lo, hi):
    mark = '**' if lo > 0 or hi < 0 else ''
    return f'{mark}[{lo:+.3f}, {hi:+.3f}]{mark}'


def main():
    OUT = run_fusion.use_round('r2')
    R1 = run_fusion.ROUNDS['r1']
    frozen = run_fusion.FROZEN
    m = pd.read_csv(OUT / 'metrics.csv').set_index(['dataset', 'method'])
    m1 = pd.read_csv(R1 / 'metrics.csv').set_index(['dataset', 'method'])
    boot = pd.read_csv(OUT / 'bootstrap.csv').set_index(['dataset', 'method'])
    ens = pd.read_csv(OUT / 'posthoc_ensemble_bootstrap.csv').set_index(['dataset', 'comparison'])
    power = pd.read_csv(OUT / 'injection_power.csv').pivot_table(index=['dataset', 'delta'], columns='method', values='power')
    power1 = pd.read_csv(R1 / 'injection_power.csv').pivot_table(index=['dataset', 'delta'], columns='method', values='power')
    coverage = pd.read_csv(OUT / 'full_coverage.csv')
    coverage1 = pd.read_csv(R1 / 'full_coverage.csv')
    native = json.loads((OUT / 'native_summary.json').read_text())
    native1 = json.loads((R1 / 'native_summary.json').read_text())
    zero = json.loads((OUT / 'posthoc_zero_volume_audit.json').read_text())
    weights = json.loads((OUT / 'stack_weights.json').read_text())
    dev = pd.read_csv(OUT / 'dev_full.csv')
    positives = {n: int(m.loc[(n, 'fusion'), 'positives']) for n in [NATIVE] + PROXY_NAMES}
    primary = [n for n in PROXY_NAMES if positives[n] >= run_fusion.PRIMARY_MIN]
    secondary = [n for n in [NATIVE] + PROXY_NAMES if n not in primary]
    wins_r1 = [n for n in primary if m.loc[(n, 'fusion'), 'ap'] > m.loc[(n, 'fusion_r1'), 'ap']]
    sig_r1 = [n for n in primary if boot.loc[(n, 'fusion_r1'), 'ap_diff_lo'] > 0]
    worse_r1 = [n for n in primary if boot.loc[(n, 'fusion_r1'), 'ap_diff_hi'] < 0]
    ens_wins = [n for n in primary if m.loc[(n, 'fusion_ensemble'), 'ap'] > m.loc[(n, 'fusion'), 'ap']]
    ens_vs_if = {n: ens.loc[(n, 'ensemble minus isolation_forest'), ['ap_diff_lo', 'ap_diff_hi']].tolist() for n in primary}
    alarm = m.loc[[(n, 'fusion') for n in PROXY_NAMES + [NATIVE]], 'alarm_rate']

    L = ['# Kronos 证据融合：第 2 轮（2026-11-07，冻结协议下的第二次测试评估）', '',
         '在第 1 轮（[results/2026-11-06/kronos_fusion](../../2026-11-06/kronos_fusion/README.md)）之后，本轮只在**校准行**上设计（14 个配置，见 [dev_full.csv](dev_full.csv) 与 [dev_ladder.csv](dev_ladder.csv)），按 [PREREGISTRATION.md](PREREGISTRATION.md) 预先写下的规则选出配置并冻结（[frozen_config.json](frozen_config.json)、[frozen_manifest.json](frozen_manifest.json)），然后在**与第 1 轮完全相同的测试行**上评估一次。本轮有两次测试查看：纯方法与集成臂。仍然没有新的 Kronos 推理。', '',
         '## 第 2 轮改了什么', '',
         '1. **`long` 通道**：相对同组内最多 512 行因果历史（严格早于当前行）的中位数的偏离，尺度为该历史的 IQR/1.349 加入条件尺度模型。32 行窗口在异常区间中会被污染，长历史保持灵敏。',
         '2. **多统计量聚合**：对每个通道以及全部活跃证据分别计算 Tippett（−min log p）与 Fisher（−Σ log p）统计量，各自在拟合分区上做保形校准，取最强的校准统计量作为行分数。第 1 轮只取单个最强证据，分散在多个特征上的中等证据被忽略。',
         '3. 其余不变：预测堆叠、pinball 条件尺度、GPD 尾部保形 p 值、验证分区上的单一联合阈值（α=0.01）、排除时间/品种编码、SD 数据关闭 `global`。', '',
         '在校准行上尝试但**未被选中**的部分（数字见 dev_full.csv）：按预测技能的通道门控（BankSim 上有利、ULB 上不利）、双侧秩稀有度（BankSim 上显著变差）、时间/品种编码进入 `global`（无影响）、幅度型统计量 meanz/maxz（BankSim 上达到 Robust MAD 水平，但 ULB 与 Sapienza 上崩溃）。', '',
         '## 结论', '',
         f'1. **相对第 1 轮：** 在 {len(primary)} 个主要数据集上，第 2 轮 AP 点估计高于第 1 轮的有 {len(wins_r1)} 个（' + '、'.join(wins_r1) + f'），配对 bootstrap 区间不含 0 的有 {len(sig_r1)} 个（' + '、'.join(f'{n} {interval(*boot.loc[(n, "fusion_r1"), ["ap_diff_lo", "ap_diff_hi"]])}，{int(boot.loc[(n, "fusion_r1"), "units"])} 个重采样单元' for n in sig_r1) + f'）；区间为负的 {len(worse_r1)} 个（' + '、'.join(f'{n} {interval(*boot.loc[(n, "fusion_r1"), ["ap_diff_lo", "ap_diff_hi"]])}，{int(boot.loc[(n, "fusion_r1"), "units"])} 个单元' for n in worse_r1) + '）。按单元数衡量，**可靠的改进只有 sd_synthetic 一个，且效应很小**；s_ffsd（7 个块）与 sd_stock（6 个 case）只是提示性的，而 s_ffsd 的“改进”是从反向（第 1 轮 AUROC ' + f(m1.loc[("s_ffsd", "fusion"), "auroc"]) + '）回到接近随机（' + f(m.loc[("s_ffsd", "fusion"), "auroc"]) + '，R@1%FPR ' + f(m.loc[("s_ffsd", "fusion"), "recall_1pct_fpr"]) + '）；sd_prices 的退化是可靠的。本轮在第 1 轮之上又有两次测试查看、校准行上 15 个配置，区间未做任何多重比较校正。',
         f'2. **相对已发布方法与水平型基线：** AP 高于已发布逐维 POT 的主要数据集 {sum(m.loc[(n, "fusion"), "ap"] > m.loc[(n, "per_feature_pot"), "ap"] for n in primary)}/{len(primary)}，区间不含 0 的 {sum(boot.loc[(n, "per_feature_pot"), "ap_diff_lo"] > 0 for n in primary)}/{len(primary)}。但**水平型基线仍然领先**：第 2 轮 AP 显著低于 Robust MAD 的主要数据集 {sum(boot.loc[(n, "robust_mad"), "ap_diff_hi"] < 0 for n in primary)}/{len(primary)}（' + '、'.join(n for n in primary if boot.loc[(n, "robust_mad"), "ap_diff_hi"] < 0) + f'），显著低于 Isolation Forest 的 {sum(boot.loc[(n, "isolation_forest"), "ap_diff_hi"] < 0 for n in primary)}/{len(primary)}（' + '、'.join(n for n in primary if boot.loc[(n, "isolation_forest"), "ap_diff_hi"] < 0) + f'）。1% 名义告警率下的实际告警率：sd_prices {pct(m.loc[("sd_prices", "fusion"), "alarm_rate"])}（结论 6）；s_ffsd {pct(m.loc[("s_ffsd", "fusion"), "alarm_rate"])} 是与第 1 轮相同的退化——同样两条频率编码特征上的极端验证行（源行 60291、62150）决定了阈值（第 1 轮 7771，本轮 85.8），测试段无行超过，不是告警控制；其余为 {pct(alarm.drop([("sd_prices", "fusion"), ("s_ffsd", "fusion")]).min())}–{pct(alarm.drop([("sd_prices", "fusion"), ("s_ffsd", "fusion")]).max())}。',
         f'3. **Kronos 仍无正向贡献：** 去掉 Kronos 后的 AP 配对差异区间，{sum(boot.loc[(n, "fusion_no_kronos"), "ap_diff_hi"] < 0 for n in primary)} 个为负（Kronos 有害：' + '、'.join(n for n in primary if boot.loc[(n, "fusion_no_kronos"), "ap_diff_hi"] < 0) + f'），{sum(boot.loc[(n, "fusion_no_kronos"), "ap_diff_lo"] > 0 for n in primary)} 个为正，其余含 0（' + '、'.join(f'{n} {interval(*boot.loc[(n, "fusion_no_kronos"), ["ap_diff_lo", "ap_diff_hi"]])}' for n in primary) + '）。BankSim 上 1% FPR 召回 ' + f(m.loc[("banksim", "fusion"), "recall_1pct_fpr"]) + '（有 Kronos）对 ' + f(m.loc[("banksim", "fusion_no_kronos"), "recall_1pct_fpr"]) + '（无），配对差异区间 ' + interval(*boot.loc[("banksim", "fusion_no_kronos"), ["r1_diff_lo", "r1_diff_hi"]]) + '（含 0）；AP 的负区间上界仅为 ' + f'{boot.loc[("banksim", "fusion_no_kronos"), "ap_diff_hi"]:+.4f}' + '，属于临界。堆叠平均权重：' + '、'.join(f'{w["dataset"]} {w["mean_weight"][-1]:.2f}' for w in weights) + '。',
         f'4. **集成臂：** 加入 Isolation Forest 行证据后，AP 点估计高于纯方法的主要数据集 {len(ens_wins)}/{len(primary)}，但配对区间不含 0 的只有 ' + ('、'.join(f'{n} {interval(*ens.loc[(n, "ensemble minus fusion"), ["ap_diff_lo", "ap_diff_hi"]])}' for n in primary if ens.loc[(n, "ensemble minus fusion"), "ap_diff_lo"] > 0) or '无') + '；相对 Isolation Forest 单独使用，配对差异区间：' + '、'.join(f'{n} {interval(*ens_vs_if[n])}' for n in primary) + f'，即集成臂在 {sum(ens_vs_if[n][1] < 0 for n in primary)}/{len(primary)} 个数据集上显著不如 Isolation Forest 本身。注意：集成臂中的森林只用检测特征，而作为对照的已发布 Isolation Forest 还使用了时间/品种编码。',
         f'5. **原生 BRD/BTC：** 第 2 轮 {native["fusion_alarms"]} 次告警（第 1 轮 {native1["fusion_alarms"]}，已发布 {native["original_alarms"]}），pump K 线名次 {native["pump_rank_fusion"]}（第 1 轮 {native1["pump_rank_fusion"]}），零成交量 K 线上的告警 {zero["fusion_alarms_on_zero_volume"]} 次（已发布 {zero["original_alarms_on_zero_volume"]}）。',
         f'6. **sd_prices 退化：** AP 从第 1 轮的 {f(m1.loc[("sd_prices", "fusion"), "ap"])} 降到 {f(m.loc[("sd_prices", "fusion"), "ap"])}，实际告警率 {pct(m.loc[("sd_prices", "fusion"), "alarm_rate"])}。该数据集只有 16 个 case，校准段 3 个：拟合分区是**单个** case（128 行，步长 MAD 为 0 的阶梯状序列），验证分区是另一个 case，4 个测试 case 的波动完全不同。多统计量的二级校准在 128 行上外推失效，而第 1 轮的单一最强证据恰好更耐受。全部 109 次告警都来自 `all:tippett` 统计量，它的二级 GPD 尺度（0.49）只有单通道统计量的一半，外推最陡；`long` 通道的原始非一致性在测试段达到 34.7（拟合段最大 2.8），因为尺度模型是在零步长 case 上学到的。这是数据量低于方法要求的失败，按预注册规则如实报告，不做事后修补。一个可能的守卫是“拟合分区少于 2 个组时退回 max 聚合”，但它无法在只有 3 个校准 case 的数据上验证。',
         f'7. **注入功效下降：** δ=3 时第 2 轮功效高于第 1 轮的数据集只有 {int((power.xs(3, level="delta").fusion > power1.xs(3, level="delta").fusion).sum())}/11；Sapienza 与 S-FFSD 上明显更低（表 2）。注入协议与第 1 轮完全相同（同样的扰动、种子与步长尺度，各方法各用自己的 99% 分位），所以这是真实的灵敏度损失；“更多统计量稀释了单点冲击证据”是对原因的假设，未单独验证。', '',
         '## 表 1：相同测试位置上的结果', '',
         '| 数据集 | 正例/已知 | 方法 | AP [95% CI] | R@1%FPR | AUROC | 实际告警率 | F1 |', '|---|---:|---|---|---:|---:|---:|---:|']
    for n in primary + secondary:
        for method in ORDER:
            if (n, method) not in m.index:
                continue
            r = m.loc[(n, method)]
            ci = f' [{f(boot.loc[(n, method), "ap_lo"])}, {f(boot.loc[(n, method), "ap_hi"])}]' if (n, method) in boot.index else ''
            L.append(f'| {n} | {positives[n]}/{int(r.known)} | {LABEL[method]} | {f(r.ap)}{ci} | {f(r.recall_1pct_fpr)} | {f(r.auroc)} | {pct(r.alarm_rate)} | {f(r.f1)} |')
    L += ['', '配对差异（第 2 轮 − 对照，AP，95% bootstrap 区间；正值表示第 2 轮更好；“单元”为重采样单元数，少于 10 的区间不可靠）：', '',
          '| 数据集 | 单元 | vs 第 1 轮 | vs 逐维 POT | vs 联合 top3 | vs 无 Kronos | vs Isolation Forest | vs Robust MAD |', '|---|---:|---|---|---|---|---|---|']
    for n in primary:
        cells = [str(int(boot.loc[(n, 'fusion'), 'units']))] + [interval(*boot.loc[(n, k), ['ap_diff_lo', 'ap_diff_hi']]) for k in ['fusion_r1', 'per_feature_pot', 'joint_top3', 'fusion_no_kronos', 'isolation_forest', 'robust_mad']]
        L.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    L += ['', '1% FPR 召回的配对差异（第 2 轮 − 第 1 轮）：' + '、'.join(f'{n} {interval(*boot.loc[(n, "fusion_r1"), ["r1_diff_lo", "r1_diff_hi"]])}' for n in primary) + '。', '',
          '集成臂的配对差异（集成 − 对照，AP）：', '', '| 数据集 | vs 第 2 轮纯方法 | vs Isolation Forest | vs Robust MAD |', '|---|---|---|---|']
    for n in primary:
        L.append(f'| {n} | ' + ' | '.join(interval(*ens.loc[(n, f'ensemble minus {k}'), ['ap_diff_lo', 'ap_diff_hi']]) for k in ['fusion', 'isolation_forest', 'robust_mad']) + ' |')
    L += ['', '![表1图](fusion_labelled.png)', '',
          '## 表 2：反事实注入功效（δ=3 与 6，1% 告警率）', '',
          '| 数据集 | δ | 第 2 轮 | 第 1 轮 | 第 2 轮无 Kronos | 集成臂 | 逐维 POT |', '|---|---:|---:|---:|---:|---:|---:|']
    for n in [NATIVE] + PROXY_NAMES:
        for delta in (3, 6):
            r, r1 = power.loc[(n, delta)], power1.loc[(n, delta)]
            L.append(f'| {n} | {delta} | {f(r["fusion"])} | {f(r1["fusion"])} | {f(r["fusion_no_kronos"])} | {f(r.get("fusion_ensemble", np.nan))} | {f(r["per_feature_pot"])} |')
    k3 = power.xs(3, level='delta')
    k13 = power1.xs(3, level='delta')
    L += ['', f'δ=3 时第 2 轮功效高于第 1 轮的数据集 {int((k3.fusion > k13.fusion).sum())}/11，高于已发布逐维 POT 的 {int((k3.fusion > k3.per_feature_pot).sum())}/11。注入只扰动序列特征，协议与第 1 轮相同。', '',
          '## 表 3：全覆盖（无 Kronos）', '',
          '| 数据集 | 可评分测试行 | 正例 | 第 2 轮管线 AP | 第 1 轮管线 AP | Isolation Forest | Robust MAD | 第 2 轮 R@1% | 第 1 轮 R@1% |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in PROXY_NAMES:
        c = coverage[coverage.dataset == n].set_index('method')
        c1 = coverage1[coverage1.dataset == n].set_index('method')
        L.append(f'| {n} | {int(c.loc["fusion_no_kronos", "test_rows_scorable"])} | {int(c.loc["fusion_no_kronos", "positives"])} | {f(c.loc["fusion_no_kronos", "ap"])} | {f(c1.loc["fusion_no_kronos", "ap"])} | {f(c.loc["IsolationForest", "ap"])} | {f(c.loc["RobustMAD", "ap"])} | {f(c.loc["fusion_no_kronos", "recall_1pct_fpr"])} | {f(c1.loc["fusion_no_kronos", "recall_1pct_fpr"])} |')
    better = [n for n in PROXY_NAMES if coverage[(coverage.dataset == n) & (coverage.method == 'fusion_no_kronos')].ap.iloc[0] > coverage1[(coverage1.dataset == n) & (coverage1.method == 'fusion_no_kronos')].ap.iloc[0]]
    L += ['', '全覆盖 AP 高于第 1 轮管线的数据集：' + ('、'.join(better) if better else '无') + '。', '',
          '![全覆盖](fusion_full_coverage.png)', '',
          '## 原生 BRD/BTC', '',
          '| 项目 | 已发布逐维 POT | 第 1 轮 | 第 2 轮 |', '|---|---:|---:|---:|',
          f'| 测试告警 / 1,152 | {native["original_alarms"]} | {native1["fusion_alarms"]} | {native["fusion_alarms"]} |',
          f'| 零成交量 K 线上的告警 | {zero["original_alarms_on_zero_volume"]} | 0 | {zero["fusion_alarms_on_zero_volume"]} |',
          f'| pump K 线是否告警 | {"是" if native["pump_original_alarm"] else "否"} | {"是" if native1["pump_fusion_alarm"] else "否"} | {"是" if native["pump_fusion_alarm"] else "否"} |',
          f'| pump K 线名次 | {native["pump_rank_original"]} | {native1["pump_rank_fusion"]} | {native["pump_rank_fusion"]} |',
          f'| 触发证据 | — | {native1["pump_top_feature"]}/{native1["pump_top_channel"]} | {native["pump_top_feature"]}/{native["pump_top_channel"]} |', '',
          '![原生时间线](fusion_native_timeline.png)', '',
          '## 设计阶段：校准行上的阶梯（dev_full.csv，AP）', '',
          '每个数据集的拟合分区为校准段前半、评估为后半（分组保持完整），管线不含 Kronos。', '']
    pivot = dev.pivot(index='dataset', columns='config', values='ap')
    configs = [c for c in ['C0 round1', 'C1 +long', 'C1 +multi', 'C2 +gate', 'C3 +multi', 'C4 +rank', 'C7 gate+long+multi4', 'C10 C7 magnitude only', 'C6 +iforest', 'C11 C1+multi+iforest'] if c in pivot.columns]
    L += ['| 数据集 | 评估正例 | ' + ' | '.join(configs) + ' |', '|---|---:|' + '---:|' * len(configs)]
    counts = dev.drop_duplicates('dataset').set_index('dataset').dev_positives
    for n in pivot.index:
        L.append(f'| {n} | {int(counts[n])} | ' + ' | '.join(f(pivot.loc[n, c]) for c in configs) + ' |')
    L += ['', '选中 `C1 +multi`（本轮配置）。', '',
          '## 设计过程的实际时间线（独立审计后补记）', '',
          '- 05:49：dev_ladder.csv（10 个配置，子采样分区）。按写在 `scripts/dev_ladder_r2.py` 里的第一条规则（含 [0.3%,3%] 的半段告警检查），它会选出 **C0（第 1 轮配置）**，且没有配置通过告警检查——半段只有约 128 行，1% 水平的检查没有意义。该规则随后被 `scripts/dev_full_r2.py` 的规则（全部校准行、AP 与 R@1% 的平均秩）替代，这一替代发生在看到子采样阶梯结果之后。',
          '- 05:57–06:13：dev_full 第一批（C0–C6，10 个配置）逐数据集完成。看到 BankSim 的结果后加入了幅度型统计量 C7–C10（BankSim 的 `__new` 文件 06:02），之后又加入 C11。',
          '- 06:14：冻结。此时 sapienza_5s、sapienza_25s、ulb 的 C7–C10 以及全部 C11 结果尚未产生（06:15–06:20 完成，与测试运行重叠）。',
          '- 审计核对：无论用冻结时的部分表还是最终的完整表、只看纯配置还是全部配置，`C1 +multi` 都是规则的赢家，因此选择本身成立；但 PREREGISTRATION.md 原先“规则在运行前固定、14 个配置全部先于冻结”的说法不准确，已改写。dev_full.csv 由各数据集文件合并并去重（158 行→150 行，去重前后数值相同）。', '',
          '## 冻结后改动', '',
          '| 时间（UTC） | 文件 | 改动 | 对结果的影响 |', '|---|---|---|---|',
          '| 22:16:59 | scripts/run_fusion.py | 无 Kronos 臂评分时传入 `long:*` 键（原来传空字典，在第一个数据集上即报 KeyError） | 在任何指标产生之前；`kronos=True` 的路径不受影响，纯方法数字不可能改变 |',
          '| 22:26:44 | scripts/run_fusion.py | 原生审计节用与其他节相同的 bundle 选项（原来缺少长历史键而崩溃，此时 labelled/injection/coverage 三节已完成） | 原生审计的 15 次告警与 pump 名次 2 已由 `verify_fusion.py` 对照修复前写出的行文件核对一致 |', '',
          '修订前的脚本副本没有保存，第一项只能通过时间（先于 forecast_skill.csv 与所有指标文件）与代码路径核对。`scripts/verify_fusion.py r2` 核对 frozen_manifest.json 当前修订的 SHA256 与各表的重算值；frozen_config.json 的哈希自冻结起未变。', '',
          '## 局限', '',
          '- 两次测试查看（纯方法、集成臂）叠加在第 1 轮之上，校准行上尝试了 15 个配置；本轮的区间没有为多次查看或多重比较校正。',
          '- 设计阶段与测试协议的拟合规模不同：设计阶段在 ≤4000 行上拟合、在 4k–68k 行上评估；冻结的测试协议只在 128–832 行的子采样拟合分区上拟合。',
          '- 多统计量的二级校准用的是拟合分区自身的（样本内）p 值，样本外的行经 GPD 外推后统计量更极端；最终阈值在验证分区上取，告警率因此仍受控，但 sd_prices 的失败正与此相关。',
          '- 设计集与测试集来自不同时段：S-FFSD 的校准段上任何方法都接近随机（Robust MAD AP 0.16），测试段上 Robust MAD 达到 0.61，因此校准段上的选择对该数据集没有指导意义。',
          '- ULB（34 个评估正例）与 Sapienza（25 个 pump，三个分辨率共享）的设计集很小。',
          '- 仍无新的 Kronos 推理；Kronos 侧的所有结论只对仓库保存的预测成立。',
          '- SD 数据的 `global` 通道按协议关闭，水平型基线利用了逐 case 的 min–max 归一化。', '',
          '## 文件', '',
          '| 文件 | 内容 |', '|---|---|',
          '| metrics.csv / bootstrap.csv / posthoc_ensemble_bootstrap.csv | 表 1 及配对区间 |', '| `*__rows.csv.gz` | 行级分数、阈值、标签，测试行上的全部对照分数 |',
          '| forecast_skill.csv / injection_power.csv / full_coverage.csv | 表 2、表 3 与预测技能 |',
          '| native_rows.csv.gz / native_summary.json / posthoc_* | 原生审计与事后分析 |',
          '| dev_full.csv / dev_ladder.csv | 设计阶段全部配置的校准行结果 |',
          '| frozen_config.json / frozen_manifest.json / PREREGISTRATION.md | 冻结协议 |', '',
          '## 复现', '', '```bash', 'PYTHONPATH=.:scripts python scripts/dev_full_r2.py        # 设计阶段（校准行）',
          'PYTHONPATH=.:scripts python scripts/run_fusion.py --round r2', 'PYTHONPATH=.:scripts python scripts/posthoc_fusion.py r2',
          'PYTHONPATH=.:scripts python scripts/posthoc_round2.py', 'PYTHONPATH=.:scripts python scripts/plot_fusion.py r2',
          'PYTHONPATH=.:scripts python scripts/report_round2.py', 'PYTHONPATH=.:scripts python scripts/verify_fusion.py r2', '```']
    (OUT / 'README.md').write_text('\n'.join(L) + '\n')
    print('wrote', OUT / 'README.md', len(L), 'lines')


if __name__ == '__main__':
    main()
