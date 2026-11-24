# S-FFSD: fuse the ordinary methods as far as they go, then add the knob

# S-FFSD：先把普通方法融合到最好，再加入旋钮

Kronos-mini (real weights). Script: `scripts/sffsd_fusion.py`. Results: `results/2026-11-25/sffsd_fusion/`. Test set: 6,159 labelled rows, 2,394 fraud. Means over three seeds.

Kronos-mini（真实权重）。脚本：`scripts/sffsd_fusion.py`。结果：`results/2026-11-25/sffsd_fusion/`。测试集：6,159 条有标签行，其中 2,394 条欺诈。三个种子取平均。

## Result / 结果

| Step / 步骤 | Method / 方法 | AP | AUROC | Recall at 1 % false alarms / 1% 误报率下的召回率 | Recall at 5 % / 5% 时 |
|---|---|---|---|---|---|
| Start / 起点 | Kronos forecast surprise B / Kronos 预测惊讶度 | 0.284 | 0.317 | 0.001 | 0.007 |
| 1 | Best single ordinary detector (HBOS) / 最好的单一普通方法 | 0.674 | 0.681 | 0.003 | 0.008 |
| 1 | All 12 detectors + L, label-free Fisher fusion / 12 种方法 + L，无标签融合 | 0.397 | 0.562 | 0.001 | 0.010 |
| 1 | **Best fusion of ordinary methods**: stacked logistic regression on 12 detectors + L + B / **普通方法的最佳融合**：对 12 种方法 + L + B 做堆叠逻辑回归 | 0.811 | 0.884 | 0.002 | 0.756 |
| 1 | The same with hand-made 32-row context averages / 同上，再加手工的 32 行上下文均值 | 0.740 | 0.823 | 0.271 | 0.294 |
| 2 | Knob K alone / 旋钮 K 单独 | 0.815 | 0.858 | 0.190 | 0.565 |
| 2 | **Best fusion + knob**: the same stacked model with K added / **最佳融合 + 旋钮**：同一堆叠模型加入 K | **0.870** | **0.896** | **0.345** | 0.719 |

**English.**

- **Step 1.** Fusing ordinary methods lifts AP from 0.674 to 0.811, but at 1 % false alarms it still catches 0.2 % of fraud. Fusion improves the overall ranking and leaves the strict-threshold problem untouched.
- **Step 2.** Adding the knob to the same model lifts AP to 0.870, the highest of anything tested on S-FFSD, and recall at 1 % false alarms from 0.2 % to 34.5 % (standard deviation over seeds 4 points). The miss rate at that threshold falls from 99.8 % to 65.5 %.
- **The comparison is fair.** Both stacked models are trained on the same labelled calibration rows (2,217 normal, 772 fraud) and differ only in the knob column.
- **Where the gain comes from.** S-FFSD fraud arrives in runs, so what matters is the recent history. Ordinary detectors look at one row. Giving them hand-made 32-row context averages also raises strict-threshold recall (to 27.1 %) but lowers AP to 0.740; the knob supplies that context in one number and improves both.
- **What is not solved.** About two thirds of fraud is still missed at 1 % false alarms. At 5 % false alarms the model without the knob is slightly ahead (0.756 against 0.719). Gradient-boosted stacking failed in both steps (AP below 0.5), so the result depends on a simple, regularised combiner. One model size; the knob and the stacking both need labelled anomalies.

**中文。**

- **第一步。** 融合普通方法把 AP 从 0.674 提高到 0.811，但在 1% 误报率下仍然只抓到 0.2% 的欺诈。融合改善了整体排序，却没有触及严格阈值下的问题。
- **第二步。** 在同一个模型里加入旋钮，AP 升到 0.870，是 S-FFSD 上所有测试方法中最高的；1% 误报率下的召回率从 0.2% 升到 34.5%（三个种子的标准差为 4 个百分点），该阈值下的漏报率从 99.8% 降到 65.5%。
- **对比是公平的。** 两个堆叠模型用同样的已标注校准行训练（2,217 条正常、772 条欺诈），唯一的区别是有没有旋钮这一列。
- **提升从哪里来。** S-FFSD 的欺诈成串出现，关键信息在近期历史里，而普通方法一次只看一行。给它们加上手工的 32 行上下文均值，也能提高严格阈值下的召回率（到 27.1%），但 AP 降到 0.740；旋钮用一个数字提供了这种上下文，并且两项指标都提高。
- **没有解决的部分。** 在 1% 误报率下仍有约三分之二的欺诈被漏掉。在 5% 误报率下，不带旋钮的模型略高（0.756 对 0.719）。梯度提升堆叠在两步中都失败（AP 低于 0.5），所以结果依赖于简单、带正则的合并器。只验证了一个模型规模；旋钮和堆叠都需要已标注的异常。

## Protocol / 实验设置

**English.** The twelve detectors of Financial_unsuperviesdAD are fitted on up to 8,192 unlabelled reference rows. Every score becomes a p-value against one random half of the normal calibration rows. Subset selection and stacking use the other half plus all calibration fraud rows. The test rows are used once, for evaluation. The knob is the frozen one-dimensional fraud knob of `results/2026-11-10`.

**中文。** Financial_unsuperviesdAD 的十二种检测器在最多 8,192 条无标签参考期数据上拟合。每个分数都相对于正常校准行的随机一半转成 p 值；子集选择和堆叠使用另一半正常行加全部校准期欺诈行。测试行只用一次，用于评估。旋钮是 `results/2026-11-10` 中已冻结的一维欺诈旋钮。

```
export PYTHONPATH=.:scripts KRONOS_MODEL=mini KRONOS_RESULTS=results/2026-11-10 FINAD_REPO=/path/to/Financial_unsuperviesdAD
python scripts/sffsd_fusion.py
```
