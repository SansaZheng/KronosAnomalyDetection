#!/usr/bin/env python3
"""Render README.md for a knowledge-boundary results directory from its saved files."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def f(v, digits=3):
    return 'NA' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{digits}f}'


def main():
    out = Path(sys.argv[1])
    data = json.loads((out / 'results.json').read_text())
    results = data['boundaries']
    detection = pd.read_csv(out / 'anomaly_detection.csv')
    intervention = pd.read_csv(out / 'intervention.csv')
    standin = 'dataset' in data
    L = []
    if standin:
        m = data['model']
        L += ['# 知识边界分析：替身 Transformer（无 Kronos 权重）', '',
              f'论文流程（探针 → 结构定位 → knob 搜索 → 闭式蒸馏 → 定向干预）在一个**替身**模型上的端到端运行：在 `{data["dataset"]}` 的参考分区上训练的因果 Transformer（{m["n_layers"]} 层、$d={m["d_model"]}$、{m["n_heads"]} 头、上下文 {m["lookback"]} 根、{m["training_windows"]} 个训练窗口、{m["epochs"]} 个 epoch，高斯下一根 K 线预测头）。隐状态 $H$ 为嵌入输出与每个块输出在最后一根 K 线处的残差流（{m["n_layers"] + 1} 层 × {m["d_model"]} 维）。**所有数字只描述这个替身模型，不代表 Kronos。** 换成真实 Kronos 只需换隐状态适配器（`scripts/run_boundary_kronos.py`）。', '',
              '特征：' + '、'.join(data['features']) + '。探针在参考+校准行的平衡子集上拟合，在测试行的平衡子集上评估；预测器只见过参考行。', '',
              '训练曲线：' + '；'.join(f'epoch {h["epoch"]} 训练 NLL {h["train_nll"]:.3f}，校准段末步 NLL {h["val_last_step_nll"]:.3f}' for h in data['training'] if 'epoch' in h) + f'；选用校准段 NLL 最低的 epoch {next((h["selected_epoch"] for h in data["training"] if "selected_epoch" in h), "NA")}。持续预测 + 校准段方差的基线 NLL {data.get("persistence_nll", float("nan")):.3f}。误差边界阈值（校准段正常行 NLL 90 分位）{data["error_threshold"]:.3f}。', '']
    else:
        L += ['# 知识边界分析：Kronos', '', f'模型 {data.get("model_dir")}，tokenizer {data.get("tokenizer_dir")}，K 线 {data["bars"]}，窗口 {data["lookback"]}，prescale {data["prescale"]}，层数 {data["layers"]}，误差边界阈值 {data["error_threshold"]:.3f}。' + ('**随机权重空跑，数字无意义。**' if data.get('tiny_dry_run') else ''), '']
    notes = out / 'NOTES.md'
    if notes.exists():
        L += ['## 结论', '', notes.read_text().strip(), '']
    L += ['## 表 1：边界可学性（平衡精度，机会水平 0.5）', '', '| 边界 | 训练行（正例） | 测试行（正例） | LR | MLP | 多视角 CNN |', '|---|---:|---:|---:|---:|---:|']
    for r in results:
        L.append(f'| {r["boundary"]} | {r["train_rows"]}（{r["train_positives"]}） | {r["test_rows"]}（{r["test_positives"]}） | {f(r["acc_logistic"])} | {f(r["acc_mlp"])} | {f(r["acc_cnn"])} |')
    L += ['', '## 表 2：边界证据的结构（附录 B 统计量）', '', '| 边界 | 激活图 | V_corr（跨层连续） | H_corr（跨维连续） | SNR (dB) | LateRatio（后半层占比） | 能量拐点（维数） |', '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        for label, text in (('cnn', 'CNN 纵向支路'), ('raw', '标准化隐状态')):
            s = r[f'structure_{label}']
            L.append(f'| {r["boundary"]} | {text} | {f(s["v_corr"], 2)} | {f(s["h_corr"], 2)} | {f(s["snr_db"], 1)} | {f(s["late_ratio"], 2)} | {r[f"knee_{label}"]} |')
    L += ['', '![差分图](boundary_difference_maps.png)', '', '![能量曲线](boundary_energy.png)', '',
          '## 表 3：knob 搜索（嵌套 top-k，δ = 1%，对照为同样大小的随机子集）', '', '| 边界 | 全维 CNN 精度 | k | 维度 | top-k 精度 | 是否达到 1% 容差 | 随机子集精度 均值 / 最小 / 最大 | 随机 ≥ top-k 的比例 | 闭式精度 |', '|---|---:|---:|---|---:|---|---|---:|---:|']
    for r in results:
        k = r['knob']
        c = k['random_control']
        L.append(f'| {r["boundary"]} | {f(k["full_accuracy"])} | {k["k"]} | {k["dims"]} | {f(k["accuracies"][str(k["k"])] if str(k["k"]) in k["accuracies"] else k["accuracies"][k["k"]])} | {"是" if k["near_optimal"] else "否（取 k≤%d 中最好的）" % max(int(x) for x in k["accuracies"])} | {f(c["mean"])} / {f(c["min"])} / {f(c["max"])} | {f(c["fraction_at_least_topk"], 2)} | {f(k["distilled_accuracy"])} |')
    L += ['', 'top-k 精度随 k 的变化：']
    for r in results:
        acc = r['knob']['accuracies']
        L.append(f'- {r["boundary"]}：' + '，'.join(f'k={k} {float(v):.3f}' for k, v in acc.items()))
    L += ['', '闭式边界分数（log 原语库，池化坐标已标准化）：', '']
    for r in results:
        L.append(f'- {r["boundary"]}：`{r["knob"]["closed_form"]}`；层注意力 α = {np.round(r["knob"]["alpha"], 3).tolist()}')
    L += ['', '![knob 空间](boundary_knob_space.png)', '',
          '## 表 4：定向干预（式 7–11）', '',
          '沿闭式边界的法向以步长 τ 扰动 knob 坐标，按 α 分摊到各层最后位置；度量为真实下一根 K 线对数似然的平均变化；对照为切向与随机方向；p 值为 Wilcoxon 符号秩检验（法向 vs 0、法向 vs 切向、法向 vs 随机）。y=0 / y=1 列是法向效应在两类测试行上的分别均值。', '',
          '| 边界 | τ | 法向 Δlog p | 中位数 | 改善比例 | y=0 | y=1 | 切向 | 随机 | p(法向≠0) | p(法向 vs 切向) | p(法向 vs 随机) |' + (' Δ预测对数方差（法向） |' if 'dlogvar_normal' in intervention else ''),
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|' + ('---:|' if 'dlogvar_normal' in intervention else '')]
    for _, r in intervention.iterrows():
        row = f'| {r.boundary} | {r.tau:+.2f} | {r.mean_normal:+.4f} | {r.median_normal:+.4f} | {100 * r.improved_normal:.1f}% | {r.mean_normal_y0:+.4f} | {r.mean_normal_y1:+.4f} | {r.mean_tangent:+.4f} | {r.mean_random:+.4f} | {r.p_normal_vs_zero:.1e} | {r.p_normal_vs_tangent:.1e} | {r.p_normal_vs_random:.1e} |'
        if 'dlogvar_normal' in intervention:
            row += f' {r.dlogvar_normal:+.3f} |'
        L.append(row)
    L += ['', '![干预](boundary_intervention.png)', '',
          '## 表 5：内部信号作为异常分数（全部抽样测试行，标注异常为正例）', '', '| 边界 | 分数 | 测试行 | 正例 | AP | AUROC | R@1%FPR |', '|---|---|---:|---:|---:|---:|---:|']
    for _, r in detection.iterrows():
        L.append(f'| {r.boundary} | {r.score} | {int(r.test_rows)} | {int(r.positives)} | {f(r.ap)} | {f(r.auroc)} | {f(r.recall_1pct_fpr)} |')
    L += ['', '`nll_surprise` 是黑箱基线（模型输出的意外度）；`cnn_probe` / `logistic_probe` / `knob_closed_form` 是隐状态信号。误差边界的探针没有用任何异常标签。', '',
          '## 文件', '', '| 文件 | 内容 |', '|---|---|', '| results.json | 全部数字 |', '| intervention.csv / anomaly_detection.csv | 表 4、表 5 |',
          '| `*_delta_cnn.npy` / `*_delta_raw.npy` | 差分图 |', '| `*_knob_space.npz` | 测试行的池化 knob 坐标、标签、闭式分数 |', '| run.log | 运行日志 |']
    (out / 'README.md').write_text('\n'.join(L) + '\n')
    print('wrote', out / 'README.md')


if __name__ == '__main__':
    main()
