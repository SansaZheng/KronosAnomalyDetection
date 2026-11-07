# 从 LLM 的“知识边界”到 Kronos 的“异常边界”

对论文 *Uncovering the Knowledge Boundaries and Mechanistic "Knob" of LLMs*（Huang, Mao, Jiang, Cambria；NTU / Tongyi Lab）的分析，以及把它的方法移植到本仓库（Kronos 预测误差异常检测）的设计与实现。

## 1. 论文做了什么

**问题。** LLM 会自信地给出错误事实。现有的幻觉检测要么只看输出（自洽性、语义熵），要么把隐状态压成一个不可解释的分数。论文想知道：模型内部**在哪里、以什么结构**区分“知道”与“不知道”。

**定义。** 取最后一个 prompt token 在所有层的隐状态 $H(x)\in\mathbb{R}^{L\times d}$（$L$ 层，含嵌入层），用外部评判给出标签 $y\in\{0,1\}$（答案是否正确）。**知识边界**是隐状态空间里的决策面 $\mathcal{B}=\{H: g(H)=0\}$。**knob** 是一个很小的维度子集 $\mathcal{J}$，只用它训练的探针精度不低于全维探针减去容差 $\delta$（式 2，$\delta=1\%$），取嵌套 top-$k$ 序列里第一个满足条件的 $k$。

**由粗到细的四步。**

| 步骤 | 做法 | 主要发现 |
|---|---|---|
| RQ1 预测 | 多视角 CNN 探针：对 $d$ 维 LayerNorm，三条卷积支路（纵向 9×1 跨层、横向 1×33 跨维、局部 5×5），ReLU + 全局最大池化 + 线性层；与 LR、MLP 对照 | PopQA/TriviaQA × Qwen3-8B/Llama-3.1-8B/Mistral-7B 的平衡精度 0.69–0.81，边界可学 |
| RQ2 结构 | 类条件激活差 $\Delta A=\bar A^{(1)}-\bar A^{(0)}$；四个汇总量：纵向/横向相邻相关 $V_{corr},H_{corr}$，维度集中度 SNR(dB)，后半层占比 LateRatio；按维度能量排序找拐点 | 证据沿深度连续（$V_{corr}\approx0.9\gg H_{corr}$），集中在少数维度（拐点 ≈ 4096 维中的 100–200），集中在后半层 |
| RQ3 knob | 按 $s_j=\sum_\ell|\Delta A_{\ell j}|$ 排序，嵌套 top-$k$ 搜索；层注意力池化 $\bar z=\sum_\ell\alpha_\ell H[\ell,\mathcal{J}]$ 接一个两节点 KAN，再符号蒸馏成闭式 $\Phi(\bar z)$（log 原语库，式 12/23） | Qwen3-8B/PopQA 上 $k=2$（维 2421、1420）即达全维精度的 1% 容差内；50 个随机 2 维子集无一达到 |
| 因果干预 | 沿 $\Phi$ 的梯度方向 $n(\bar z)$ 以步长 $\tau$ 扰动池化坐标，按 $\alpha_\ell/\sum\alpha_m^2$ 分摊到各层最后 token（最小范数补丁，式 10）；对照：切向、随机方向；度量：teacher forcing 下参考答案的对数似然变化（式 11） | 法向干预带来最大、单调的似然变化，切向/随机接近 0；Wilcoxon $p<10^{-25}$；Llama 上独立复现（维 788、2978） |

**我对论文的评价。**

- 可信的部分：探针精度跨模型稳定；差分图的结构统计（尤其 $V_{corr}$ 与 LateRatio）给出了可检验的几何性质；knob 搜索有随机对照；干预有切向与随机对照并做配对检验；附录 H 的 25 次重复表明维 2421 稳定出现。
- 需要小心的部分：(a) 干预度量的是 teacher forcing 下参考答案的似然，不是自由生成是否变对——论文自己在局限里承认；(b) $\tau$ 的定义里除以 $\|\nabla\Phi\|$，等价于在 $\Phi$ 的单位尺度上走，但不同样本的实际隐状态扰动范数差别很大，“$\tau=0.1$”并非统一的小扰动；(c) 符号蒸馏的闭式只在两个池化坐标上成立，与“机制”之间仍隔着注意力池化和层间分摊两步近似；(d) 维度索引不跨模型迁移，论文明确只主张“分析流程”可迁移。
- 与异常检测最相关的一点：论文把“模型知不知道”建模成**隐状态里的一个可定位、可操控的方向**，而不是输出分数。这正是本仓库之前只用 Kronos **输出**（预测误差）所缺的视角。

## 2. 对应关系：LLM → Kronos

| 论文 | Kronos 异常检测 |
|---|---|
| prompt $x$ | 一段 K 线上下文（本仓库原生实验 64 根、代理实验 32 根） |
| 最后一个 prompt token 的隐状态 $H(x)\in\mathbb{R}^{L\times d}$ | 上下文最后一根 K 线处，嵌入输出与每个 Transformer 块输出的残差流（`kronos_ad/hidden.py` 用 forward hook 读取，不改官方代码） |
| 标签 $y$：答案是否正确 | 两种边界：**误差边界** $y=1$ 当下一根真实 K 线在 teacher forcing 下的 NLL 超过校准段 90 分位（“模型不知道这根 K 线”，无需异常标签）；**异常边界** $y=1$ 为标注异常 K 线 |
| 参考答案的 teacher-forced 对数似然 | 真实下一根 K 线的 token 对 $(s_1^*,s_2^*)$ 的对数似然 $\log p(s_1^*\mid\text{ctx})+\log p(s_2^*\mid s_1^*,\text{ctx})$（式 11 的单步情形）。tokenizer 编码器与预测器自注意力都是因果的，追加真实 K 线不改变上下文 token，所以这个目标定义是一致的 |
| 最小范数层间补丁（式 10） | 同一公式，补丁加在各层最后位置（`KronosHiddenStates._hooks`） |
| 探针、差分图、knob 搜索、符号蒸馏、干预 | 与模型无关，全部在 `kronos_ad/boundary.py` |

与检测器的接口：探针分数或闭式 knob 分数是一行一个数，可作为 `row:` 证据源进入第 2 轮的证据融合（`kronos_ad/fusion.py` 的 row evidence），与预测残差、水平通道一起做保形校准。

## 3. 实现

| 文件 | 内容 |
|---|---|
| `kronos_ad/boundary.py` | `LogisticProbe` / `MLPProbe` / `CNNProbe`（论文的三支路 CNN，核尺寸自适应浅/窄模型）；`difference_map`、`summary_metrics`（附录 B 的 $V_{corr}$、$H_{corr}$、SNR、LateRatio）、`dimension_energy`、`knee`、`rank_dimensions`；`nested_knob_search`（式 2 的 $\delta$ 准则 + 随机子集对照）；`KnobPredictor`（层注意力池化 + log 原语闭式 $\Phi$，直接以交叉熵拟合，`closed_form()` 输出表达式）；`intervention_directions`、`layer_patches`、`hidden_patch`（式 7–10） |
| `kronos_ad/hidden.py` | `KronosHiddenStates`：官方 `Kronos`/`KronosTokenizer` 的隐状态读取、补丁注入、teacher-forced 似然；`normalize_windows`（官方归一化规则，可选 `prescale`） |
| `scripts/boundary_pipeline.py` | 五步流程的共享实现（探针 → 定位 → knob → 闭式 → 干预，含 Wilcoxon 检验与 y=0/y=1 分组效应） |
| `scripts/run_boundary_kronos.py` | 真实 Kronos 权重上的运行脚本（协议已写在文件头，`--tiny` 用随机权重小模型做空跑） |
| `scripts/boundary_standin.py` | 无权重时的替身实验：在仓库自己的 Sapienza 15 秒成交块特征行上训练一个小的因果 Transformer，跑同一条流程 |
| `scripts/plot_boundary.py` | 差分图、能量曲线、knob 空间、干预曲线 |
| `tests/test_boundary.py`、`tests/test_hidden.py` | 植入边界的合成隐状态上验证每一步；用固定版本的官方模型代码（`tests/vendor/`）实例化随机权重的微型 Kronos，验证 hook、补丁只作用于最后位置、似然计算与干预链路 |

## 4. 替身实验（无 Kronos 权重）

见 `results/2026-11-07/knowledge_boundary_standin/README.md`。替身模型只是检验流程并给出“在一个真实训练的金融 Transformer 上这套分析会得到什么”的样例；它的任何数字都不代表 Kronos。

## 5. 在真实 Kronos 上运行

```bash
PYTHONPATH=.:scripts python scripts/run_boundary_kronos.py \
    --kronos-src /path/to/shiyu-coder-Kronos \
    --model-dir /path/to/Kronos-mini --tokenizer-dir /path/to/Kronos-Tokenizer-2k \
    --bars results/2026-11-06/kronos/detections.csv.gz --prescale \
    --out results/<date>/knowledge_boundary_kronos
PYTHONPATH=.:scripts python scripts/plot_boundary.py results/<date>/knowledge_boundary_kronos
```

协议固定在脚本文件头：64 根窗口、官方归一化、误差边界阈值为校准段正常行 NLL 的 90 分位、探针在校准行拟合、在平衡的测试行评估、$\delta=1\%$、$k\le10$、50 个随机对照、论文的 8 个 $\tau$、切向与随机对照、Wilcoxon 检验。原生 BRD/BTC 只有一根标注 pump K 线，异常边界需要更多标注（第三个提交准备的 Binance pump 事件表）；误差边界不需要标签。

## 6. knob 能否降低细微异常的漏报率（替身实验，2026-11-08）

把目标改写成检测主张来检验：在替身自身最强的检测器（真实行的 teacher-forced NLL 意外度 B）旁边加上 knob 通道，能否在相同假阳性率下找回 B 漏掉的异常。协议在 `scripts/boundary_missrate.py` 的文件头预先固定（Tippett 为主要组合，Fisher 与"每通道等预算 OR"为次要；细微异常的两种定义：B 漏掉的、水平幅度低于异常中位数的；对异常组的配对自助法与 McNemar 计数）。数据范围限定为金融交易数据集：Sapienza 15S、BankSim、S-FFSD、ULB，每个各训练一个替身（`scripts/boundary_standin.py --dataset`）。首次 Sapienza 运行后做了一次修订（未截断的水平幅度、GPD 尾部 p 值、加入 OR 臂），首次运行保留为 `run_A.log`。结果与解读见 `results/2026-11-08/README.md`。

要点：

- **边界可学时 knob 确有互补性。** 在 Sapienza、BankSim、S-FFSD 上标注边界可学（平衡精度 0.70–0.82），knob 得分与 B 在异常上强负相关（ρ ≈ −0.5 到 −0.8），单独的 knob 在自身 1% FPR 下找回 B 漏掉异常的 60% / 43% / 26%，而对 B 已检出的异常几乎不响应。
- **按每通道等假阳性预算组合时漏报率下降。** BankSim：1% FPR 下 0.666 → 0.540（召回增量 [+0.103, +0.149]）；低幅度欺诈（水平统计量看不到的一半）0.28 → 0.40，叠加在水平基线之上仍有 +0.12 到 +0.18。S-FFSD：1.000 → 0.825（[+0.095, +0.261]）——那里的欺诈比正常行*更*可预测（NLL 意外度 AUROC 0.11），B 一个都抓不到。Sapienza：无净变化（Fisher 找回 10 个中的 3–4 个）。ULB：边界不可学（探针 0.53），knob 是噪声，分半预算反而损失召回。
- **预先指定的 Tippett 组合在所有数据集上失败。** 原因是 NLL 意外度的分布从校准分区到测试分区上移（校准阈值在 7–12% 的测试正常行上告警），p 值汇聚继承了偏移；排序空间的等预算 OR 不受影响。隐状态通道反而是跨分区更稳定的通道（0.9–2.4%）。
- **knob 从不优于同一隐状态上的普通探针**（BankSim OR：探针 +413/−101，knob +282/−101）；knob 的价值在闭式与干预，而非检测质量。knob 与探针都是有监督的；B、L 与融合检测器是无监督的，比较时须注意。
- **干预检验需要方差膨胀对照。** 在 ULB 上边界处于随机水平，沿 knob 法向的干预仍"显著"提高似然，靠的是抬高预测方差；论文的切向与随机对照挡不住这种方向。
