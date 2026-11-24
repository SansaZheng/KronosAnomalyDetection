# Kronos against recent unsupervised detectors, and a test of six remedies

# Kronos 与近年无监督方法的对比，以及六种改进方案的实测

Kronos-mini (real weights), S-FFSD and Sapienza 15 s. Scripts: `scripts/compare_recent_methods.py`, `scripts/solutions_study.py`. Results: `results/2026-11-24/`.

Kronos-mini（真实权重），S-FFSD 与 Sapienza 15 秒。脚本：`scripts/compare_recent_methods.py`、`scripts/solutions_study.py`。结果：`results/2026-11-24/`。

## 1. Short answer / 结论

**English.**

- **Kronos' forecast surprise is not competitive.** On both datasets it ranks anomalies worse than most of the twelve detectors of [Financial_unsuperviesdAD](https://github.com/SansaZheng/Financial_unsuperviesdAD) run on the same rows.
- **Sapienza: ordinary methods win.** kNN, HBOS and GMM reach AP 0.92–0.93; the best Kronos combination reaches 0.79, and adding the knob to those methods makes them worse.
- **S-FFSD: the knob is the one thing that works.** Every unsupervised detector stays at AP ≤ 0.67 and catches almost nothing at 1 % false alarms. The knob alone reaches AP 0.815 and lifts every detector it is combined with. The knob uses labelled anomalies from the training period; the twelve detectors use none.
- **Remedies.** Of six remedies, fine-tuning is the strongest on Sapienza (surprise AP 0.567 → 0.841 on the evaluation subset) and none of them repairs the surprise score on S-FFSD.

**中文。**

- **Kronos 的预测惊讶度没有竞争力。** 在两个数据集上，它对异常的排序都不如 [Financial_unsuperviesdAD](https://github.com/SansaZheng/Financial_unsuperviesdAD) 中十二种检测器的大多数（在相同的行上运行）。
- **Sapienza：普通方法胜出。** kNN、HBOS、GMM 的 AP 达到 0.92–0.93；Kronos 最好的组合只有 0.79，而且把旋钮加到这些方法上反而变差。
- **S-FFSD：只有旋钮有效。** 所有无监督检测器的 AP 都不超过 0.67，在 1% 误报率下几乎抓不到异常。旋钮单独就达到 AP 0.815，并且与任何检测器合并都能提升它。旋钮用到了训练期的已标注异常，十二种检测器不使用标签。
- **改进方案。** 六种方案中，微调在 Sapienza 上效果最强（评估子集上惊讶度 AP 0.567 → 0.841）；在 S-FFSD 上没有任何方案能修复惊讶度。

## 2. Comparison with twelve detectors / 与十二种检测器的对比

**English.** The detectors were fitted with that repository's own code, preprocessing and default budgets (at most 8,192 unlabelled rows, seeds 42–44) on the reference period of this project's split, and scored on the same test rows and the same feature columns as Kronos. Values are means over three seeds. "+ K" is the detector Fisher-combined with the Kronos knob.

**中文。** 检测器使用该仓库自己的代码、预处理和默认预算（最多 8,192 条无标签训练行，种子 42–44），在本项目划分的参考期上拟合，并在与 Kronos 相同的测试行、相同的特征列上打分。数值为三个种子的平均。“+ K”表示该检测器与 Kronos 旋钮做 Fisher 合并。

| Detector / 检测器 | Sapienza AP | Sapienza AUROC | Sapienza recall@1% | Sapienza AP + K | S-FFSD AP | S-FFSD AUROC | S-FFSD recall@1% | S-FFSD AP + K |
|---|---|---|---|---|---|---|---|---|
| kNN | 0.933 | 0.999 | 0.974 | 0.593 | 0.314 | 0.370 | 0.001 | 0.748 |
| HBOS | 0.923 | 0.999 | 0.984 | 0.704 | 0.674 | 0.681 | 0.003 | 0.823 |
| GMM | 0.919 | 0.999 | 0.969 | 0.773 | 0.471 | 0.664 | 0.001 | 0.804 |
| Isolation Forest | 0.803 | 0.999 | 0.990 | 0.369 | 0.400 | 0.565 | 0.001 | 0.776 |
| VAE | 0.734 | 0.999 | 0.974 | 0.510 | 0.506 | 0.726 | 0.001 | 0.831 |
| RGLD-ID (2026) | 0.730 | 0.999 | 1.000 | 0.171 | 0.442 | 0.650 | 0.002 | 0.824 |
| One-Class SVM | 0.606 | 0.997 | 0.865 | 0.239 | 0.321 | 0.405 | 0.001 | 0.719 |
| RTTAD-TTA (2026) | 0.587 | 0.998 | 0.964 | 0.540 | 0.456 | 0.528 | 0.001 | 0.783 |
| PCA-SPE | 0.547 | 0.995 | 0.948 | 0.562 | 0.456 | 0.436 | 0.003 | 0.763 |
| Autoencoder | 0.426 | 0.995 | 0.891 | 0.444 | 0.375 | 0.409 | 0.002 | 0.772 |
| Deep SVDD | 0.054 | 0.706 | 0.177 | 0.094 | 0.262 | 0.192 | 0.001 | 0.753 |
| LOF | 0.008 | 0.823 | 0.000 | 0.026 | 0.441 | 0.630 | 0.002 | 0.822 |
| **Kronos B (surprise)** | 0.308 | 0.961 | 0.594 | – | 0.284 | 0.317 | 0.001 | – |
| **Kronos K (knob)** | 0.053 | 0.805 | 0.234 | – | 0.815 | 0.858 | 0.190 | – |
| L (level magnitude) | 0.537 | 0.996 | 0.906 | 0.592 | 0.673 | 0.766 | 0.000 | 0.843 |
| **Kronos B + L + K** | 0.785 | 0.998 | 0.953 | – | 0.800 | 0.860 | 0.242 | – |

Test rows: Sapienza 24,040 (64 pumps); S-FFSD 6,159 labelled (2,394 fraud). / 测试行：Sapienza 24,040 条（64 个拉盘）；S-FFSD 6,159 条有标签（2,394 条欺诈）。

**English.** Two cautions. The detectors see one row at a time; Kronos sees the 32 rows before it. The knob is found with labelled anomalies, so "K" rows are label-assisted while the twelve detectors are fully unsupervised.

**中文。** 两点注意。这些检测器一次只看一行；Kronos 能看到前面 32 行。旋钮是用已标注的异常找出来的，所以带“K”的行借助了标签，而十二种检测器完全无监督。

## 3. Six remedies, tested / 六种改进方案的实测

**English.** Fusion, two-stage screening and rolling thresholds use all labelled test rows. Clean context, joint input and fine-tuning re-run Kronos on an evaluation subset (S-FFSD: all 6,159 labelled rows; Sapienza: all 64 pumps plus 6,000 random normal rows), so their Sapienza AP is not comparable with section 2; compare them with the "unchanged" row.

**中文。** 融合、两级筛选和滚动阈值使用全部有标签测试行。干净上下文、联合输入和微调需要在评估子集上重新运行 Kronos（S-FFSD：全部 6,159 条有标签行；Sapienza：全部 64 个拉盘加 6,000 条随机正常行），因此它们在 Sapienza 上的 AP 不能与第 2 节比较，应与“不改动”一行比较。

| Remedy / 方案 | Sapienza | S-FFSD | Verdict / 结论 |
|---|---|---|---|
| Fusion with the ordinary detector L / 与普通方法 L 融合 | AP 0.537 (L) → 0.785 (B+L+K) | AP 0.673 (L) → 0.843 (L+K) | Works on both / 两者都有效 |
| Two-stage: L screens, Kronos scores the rest / 两级：L 先筛，Kronos 只给可疑行打分 | Kronos on 1.5 % of rows: AP 0.793 (all rows: 0.785) | Kronos on 11 % of rows: AP 0.673; needs 46 % of rows to reach 0.857 | Works on Sapienza only / 仅 Sapienza 有效 |
| Rolling 1 % threshold / 滚动 1% 阈值 | False alarms 0.55 % → 0.79 %, recall 0.906 → 0.922 | False alarms 3.6 % → 1.1 %, recall 0.385 → 0.024 | Corrects the rate; fails when anomalies are common / 能校正误报率；异常很多时失效 |
| Clean context (drop flagged rows) / 干净上下文 | B: recall@1% 0.594 → 0.875, AP 0.567 → 0.558 | No change (AP 0.284 → 0.279) | Partial, Sapienza only / 部分有效，仅 Sapienza |
| Joint input (features share a sequence) / 多特征联合输入 | 6.6× faster; B AP 0.567 → 0.173 | 6.3× faster; B AP 0.284 → 0.438 | Faster; accuracy mixed / 更快；精度好坏不一 |
| Fine-tuning on unlabelled reference rows / 用无标签参考期数据微调 | B AP 0.567 → 0.841, recall@1% 0.594 → 0.906 | B AP 0.284 → 0.316, AUROC 0.317 → 0.403 | Strong on Sapienza, not on S-FFSD / Sapienza 很有效，S-FFSD 无效 |

**English.**

- **Why S-FFSD resists.** Its fraud rows are neither large in magnitude nor surprising, and fraud is 39 % of the labelled test rows. Screening by magnitude leaves the fraud out, a rolling quantile is pushed up by the fraud itself, and fine-tuning on unlabelled data teaches Kronos the fraud runs as normal.
- **Fine-tuned Kronos against ordinary methods on Sapienza.** After 600 steps (18 minutes on 2 CPUs) the surprise score reaches recall 0.906 and AUROC 0.998 at 1 % false alarms. kNN and HBOS reach 0.974–0.984 and 0.999 without any model. Fine-tuning closes most of the gap; it does not overtake.
- **A correction.** An earlier summary said a 1 % threshold set on the calibration period flagged about 5 % of S-FFSD test rows for the surprise score. With the threshold taken from all calibration rows, the surprise score flags 1.2 %. The drift is in the fused score (8.6 % of rows flagged).
- **Fused numbers differ from earlier reports.** Here p-values are calibrated on all normal calibration rows (11,987 for Sapienza, 4,434 for S-FFSD); the earlier run used a held-out half of the normal pool rows. Sapienza's B + L + K AP is 0.785 here and 0.614 there. With 64 anomalies the fused AP is sensitive to that choice.
- **Limits.** One model size, one fine-tuning run, one seed for the Kronos remedies. The detectors' rankings on Sapienza have large seed spread for some methods (AP standard deviation up to 0.32).

**中文。**

- **S-FFSD 为什么难以改进。** 它的欺诈行幅度不大，也不令模型意外，而且欺诈占有标签测试行的 39%。按幅度筛选会把欺诈筛掉；滚动分位数会被欺诈本身抬高；用无标签数据微调则会让 Kronos 把欺诈串当成正常。
- **Sapienza 上微调后的 Kronos 与普通方法。** 微调 600 步（2 核 CPU 约 18 分钟）后，惊讶度在 1% 误报率下的召回率达到 0.906，AUROC 0.998。kNN 和 HBOS 不需要任何模型就达到 0.974–0.984 和 0.999。微调缩小了大部分差距，但没有超过。
- **一处更正。** 此前的总结说，按校准期设定的 1% 阈值在 S-FFSD 测试期使惊讶度报警了约 5% 的行。改用全部校准行确定阈值后，惊讶度的报警比例是 1.2%。漂移出现在融合分数上（8.6% 的行被报警）。
- **融合数值与此前报告不同。** 这里的 p 值用全部正常校准行校准（Sapienza 11,987 条，S-FFSD 4,434 条）；此前的运行使用正常行的留出一半。Sapienza 的 B + L + K 的 AP 这里是 0.785，此前是 0.614。只有 64 个异常时，融合后的 AP 对这一选择很敏感。
- **局限。** Kronos 的各项方案只用了一个模型规模、一次微调、一个随机种子。部分检测器在 Sapienza 上的排序随种子波动很大（AP 标准差最高 0.32）。

## 4. Reproduce / 复现

```
export PYTHONPATH=.:scripts KRONOS_MODEL=mini KRONOS_RESULTS=results/2026-11-10 HF_HUB_OFFLINE=1
export FINAD_REPO=/path/to/Financial_unsuperviesdAD      # commit 2741cb7; pip install pythresh==1.1.1
python scripts/compare_recent_methods.py --dataset sapienza_15s
python scripts/compare_recent_methods.py --dataset s_ffsd
python scripts/solutions_study.py --dataset sapienza_15s --only offline
python scripts/solutions_study.py --dataset sapienza_15s --only inference
python scripts/solutions_study.py --dataset s_ffsd --only offline
python scripts/solutions_study.py --dataset s_ffsd --only inference
```
