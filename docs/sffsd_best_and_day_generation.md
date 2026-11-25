# S-FFSD: how far Kronos detection can be pushed, and whether day-by-day generation helps

# S-FFSD：Kronos 异常检测能做到多好，以及“逐天生成”是否有帮助

Kronos-mini (real weights), seed 42. Scripts: `scripts/sffsd_full_states.py`, `scripts/sffsd_block_generation.py`, `scripts/sffsd_generation_fusion.py`, `scripts/sffsd_best.py`. Results: `results/2026-11-26/sffsd_best/`. Test set: 6,159 labelled rows, 2,394 fraud.

Kronos-mini（真实权重），随机种子 42。脚本同上。结果：`results/2026-11-26/sffsd_best/`。测试集：6,159 条有标签行，其中 2,394 条欺诈。

## 1. Short answer / 结论

**English.**

- **Day-by-day generation helps, as a replacement for the rolling surprise.** Its score reaches AP 0.71 where the rolling surprise B reaches 0.28. Fused with the level channel L and the knob K without any labels it gives AP 0.847 and recall 0.339 at 1 % false alarms (previous label-free best, L + K: 0.843 and 0.269).
- **Day-by-day reconstruction does not help** (AP 0.290 against 0.284).
- **Highest numbers reached:** a logistic regression on ordinary detectors + raw features + Kronos B and K with rolling context reaches AP 0.905, AUROC 0.917 and recall 0.68 at 1 % false alarms. This configuration was picked after looking at test results, so it is an upper bound, not a forecast.
- **What a rule fixed in advance selects:** AP 0.864, AUROC 0.890, recall 0.447 at 1 % (rule 2 below). The other rule selects a boosting model with AP 0.733.
- **Kronos is what makes the difference.** The best model without Kronos reaches AP 0.771 and recall 0.003 at 1 %; Kronos channels alone reach 0.886 and 0.60.
- **Caveat.** 1,889 of the 2,394 test fraud rows sit in one burst. Outside the burst every detector is weak (best AP 0.377 where the fraud share is 0.127).

**中文。**

- **逐天生成有帮助，可以替代滚动惊讶度。** 生成分数的 AP 达到 0.71，而滚动惊讶度 B 只有 0.28。不用任何标签，把它与水平通道 L、旋钮 K 融合，AP 为 0.847，1% 误报率下召回率 0.339（此前无标签最佳 L + K 为 0.843 和 0.269）。
- **逐天重现没有帮助**（AP 0.290 对 0.284）。
- **达到的最高数字：** 对“普通方法 + 原始特征 + 带滚动上下文的 Kronos B 和 K”做逻辑回归，AP 0.905，AUROC 0.917，1% 误报率下召回率 0.68。这个配置是看过测试结果后挑出来的，所以它是上限，不是预期值。
- **事先定好的规则选出的结果：** AP 0.864，AUROC 0.890，1% 时召回率 0.447（下文规则 2）。另一条规则选中了提升树模型，AP 0.733。
- **起决定作用的是 Kronos。** 不用 Kronos 的最好模型 AP 0.771，1% 时召回率 0.003；只用 Kronos 通道就有 0.886 和 0.60。
- **注意。** 测试集 2,394 条欺诈中有 1,889 条集中在一次爆发里。在爆发之外，所有检测器都很弱（最好 AP 0.377，而欺诈占比为 0.127）。

## 2. What a "day" is / “天”的含义

**English.** The S-FFSD `Time` column is a transaction counter, not a clock, so there are no real days. A "day" here is a block of 32 consecutive transactions. For each block Kronos sees only the 32 rows before the block, so fraud inside the block cannot leak into what Kronos conditions on.

- **Reconstruction (F).** Every row of the block is scored by its NLL as the next row after that frozen context.
- **Generation (G).** Kronos samples 8 continuations of 32 steps per feature from the frozen context. Each real row is scored by how far it lies in the tails of the generated values of its block. **Gb** is the mean of G over the block.

**中文。** S-FFSD 的 `Time` 列是交易序号，不是时钟，所以没有真实的“天”。这里的一“天”是连续 32 笔交易组成的一块。对每一块，Kronos 只能看到块之前的 32 行，因此块内的欺诈不会进入 Kronos 的上下文。

- **重现（F）。** 把块内每一行当作冻结上下文之后的下一行，用它的 NLL 打分。
- **生成（G）。** Kronos 从冻结上下文出发，对每个特征采样 8 条长度为 32 的延续序列。每一条真实行按它落在该块生成值尾部有多远来打分。**Gb** 是 G 在整块上的平均。

## 3. Day-by-day results, no labels used / 逐天方法的结果（不使用标签）

Each channel becomes a p-value against normal calibration rows; channels are combined with Fisher's method. "Outside" = the 3,982 test rows outside the burst (505 fraud).

每个通道先相对校准期的正常行换算成 p 值，再用 Fisher 方法合并。“爆发外”指测试集中爆发之外的 3,982 行（505 条欺诈）。

| Channels / 通道 | AP | AUROC | Recall at 1 % / 1% 召回 | Recall at 5 % / 5% 召回 | AP outside / 爆发外 AP |
|---|---|---|---|---|---|
| B (rolling surprise / 滚动惊讶度) | 0.284 | 0.317 | 0.001 | 0.007 | 0.100 |
| F (day reconstruction / 逐天重现) | 0.290 | – | – | – | – |
| G (day generation, row / 逐天生成，逐行) | 0.660 | 0.786 | 0.009 | 0.219 | 0.161 |
| Gb (day generation, block mean / 逐天生成，整块均值) | 0.713 | 0.805 | 0.000 | 0.291 | 0.175 |
| K (knob / 旋钮) | 0.815 | 0.858 | 0.190 | 0.564 | 0.245 |
| L + K | 0.843 | 0.877 | 0.269 | 0.668 | 0.287 |
| B + L + K | 0.800 | 0.860 | 0.242 | 0.561 | 0.233 |
| G + L + K | 0.845 | 0.873 | 0.295 | 0.647 | 0.277 |
| **Gb + L + K** | **0.847** | 0.871 | **0.339** | 0.673 | 0.281 |

**English.**

- Replacing B by Gb in the three-channel fusion lifts AP from 0.800 to 0.847 and recall at 1 % from 0.242 to 0.339.
- Against L + K the gain in AP is within noise (0.843 to 0.847, one seed); the gain at 1 % false alarms is larger (0.269 to 0.339).
- On the calibration period, which has no burst, G and Gb are at chance (AP 0.108 and 0.118; fraud share 0.118). The generation score detects "this whole day is unlike what Kronos expected", which pays off during a fraud wave and not for isolated fraud.

**中文。**

- 在三通道融合里把 B 换成 Gb，AP 从 0.800 升到 0.847，1% 时召回率从 0.242 升到 0.339。
- 与 L + K 相比，AP 的提升在噪声范围内（0.843 到 0.847，单个种子）；1% 误报率下的提升更明显（0.269 到 0.339）。
- 校准期没有爆发，G 和 Gb 在那里等同于随机（AP 0.108 和 0.118；欺诈占比 0.118）。生成分数检测的是“这一整天不像 Kronos 预期的样子”，在欺诈成批出现时有效，对零散欺诈无效。

## 4. Search for the best detector / 搜索最佳检测器

**English.** 87 candidates: 29 feature sets × 3 models (logistic regression C = 1, C = 0.05, gradient boosting). Feature blocks: 12 ordinary detectors + L, raw features, Kronos B, knob K (dimension 220 at every layer), rolling context means (8 / 32 / 128 rows and an exponential average), a 128-row Kronos context, and day-frozen surprise. Two selection rules, neither of which sees test labels:

- **Rule 1:** train on reference, validate on calibration.
- **Rule 2:** train on reference + first half of calibration, validate on the second half.

Reported test scores come from a refit on reference + calibration.

**中文。** 共 87 个候选：29 组特征 × 3 种模型（逻辑回归 C = 1、C = 0.05、梯度提升树）。特征块包括：12 种普通方法 + L、原始特征、Kronos B、旋钮 K（每一层的第 220 维）、滚动上下文均值（8 / 32 / 128 行和指数平均）、128 行的 Kronos 上下文、逐天冻结的惊讶度。两条选择规则都不接触测试标签：

- **规则 1：** 在参考期训练，在校准期验证。
- **规则 2：** 在参考期 + 校准期前半训练，在校准期后半验证。

报告的测试分数来自在参考期 + 校准期上重新拟合的模型。

| Detector / 检测器 | How chosen / 选择方式 | AP | AUROC | Recall at 1 % / 1% 召回 | Recall at 5 % / 5% 召回 | AP outside / 爆发外 AP |
|---|---|---|---|---|---|---|
| Best without Kronos (ordinary + raw, logistic) / 不用 Kronos 的最好结果 | best on test / 测试集最好 | 0.771 | 0.873 | 0.003 | 0.551 | 0.212 |
| Kronos only: B + K with context, logistic / 仅 Kronos | best on test / 测试集最好 | 0.886 | 0.898 | 0.597 | 0.697 | 0.286 |
| Kronos + ordinary, everything, logistic C = 0.05 / 全部特征 | **rule 2 / 规则 2** | 0.864 | 0.890 | 0.447 | 0.616 | 0.244 |
| Kronos + ordinary, all context, boosting / 提升树 | **rule 1 / 规则 1** | 0.733 | 0.835 | 0.116 | 0.248 | – |
| Ordinary + raw + B + K with Kronos context, logistic C = 1 / 带 Kronos 上下文 | best on test / 测试集最好 | **0.905** | **0.917** | 0.681 | 0.749 | 0.311 |
| Ordinary + raw + B + K, 32- and 128-row contexts, logistic C = 1 / 32 和 128 行上下文 | best outside the burst / 爆发外最好 | 0.905 | 0.915 | 0.653 | **0.768** | **0.377** |

**English.**

- **Validation does not predict test here.** Across the 87 candidates the rank correlation between validation AP and test AP is −0.23 (rule 1) and 0.03 (rule 2). The burst pattern of the test period does not occur earlier, so no earlier period can tell which model will handle it.
- **Rule 2 was added after the first search had been looked at**, so it is not fully independent of the test results either. Its pick (AP 0.864) is the most defensible number; 0.905 is the ceiling.
- **Logistic regression is the robust choice.** Mean test AP over feature sets: logistic 0.82, boosting 0.60. Boosting fits the pre-burst pattern too closely.
- **The 128-row context** leaves AP unchanged (0.905) and gives the best result outside the burst (AP 0.377 against 0.311–0.368).
- **Day-frozen surprise adds nothing** to the supervised models (0.905 with and without).
- Kronos-only models selected by either rule are boosting models that fail on test (AP 0.46–0.47). The Kronos-only logistic models are far better (0.886) but were not selected.

**中文。**

- **在这个数据集上，验证成绩不能预测测试成绩。** 87 个候选中，验证 AP 与测试 AP 的秩相关为 −0.23（规则 1）和 0.03（规则 2）。测试期的爆发模式在此前没有出现过，所以任何更早的时段都无法判断哪个模型能应对它。
- **规则 2 是在看过第一轮搜索结果之后才加入的**，因此它也不完全独立于测试结果。它选出的结果（AP 0.864）是最站得住的数字；0.905 是上限。
- **逻辑回归是稳健的选择。** 各特征组的平均测试 AP：逻辑回归 0.82，提升树 0.60。提升树对爆发前的模式拟合过紧。
- **128 行上下文** 不改变 AP（0.905），但在爆发之外给出最好结果（AP 0.377，对比 0.311–0.368）。
- **逐天冻结的惊讶度** 对有监督模型没有增益（加与不加都是 0.905）。
- 两条规则在“仅 Kronos”一组里选中的都是提升树模型，测试表现很差（AP 0.46–0.47）。仅 Kronos 的逻辑回归模型好得多（0.886），但没有被选中。

## 5. Limits / 局限

**English.** One seed. One dataset with one burst: the high test numbers mostly measure how well a detector recognises that burst. Generation was run with 8 samples and 32-row days only; other block sizes and more samples are untested. Generation costs about 5.5 s per block on two CPU cores (45 minutes for the test period).

**中文。** 只用了一个种子。只有一个数据集、一次爆发：测试集上的高分主要反映检测器识别这次爆发的能力。生成只试了 8 个样本、32 行为一天；其他块大小和更多样本尚未测试。生成在两个 CPU 核上每块约需 5.5 秒（测试期共 45 分钟）。
