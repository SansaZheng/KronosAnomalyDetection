# KronosAnomalyDetection

基于 **Kronos 单步预测误差 + 逐维 POT 上尾阈值** 的金融时间序列异常检测与异常维度定位。

这是为 Kronos 设计的独立下游异常检测方案，**不是 Kronos 原论文提出的异常定义，也不声称已有检测性能提升**。本仓库提供可运行实现、CSV 命令行入口、校准记录及无未来泄漏测试。原始 Kronos 的模型和权重从官方项目加载，本仓库不复制或重新发布它们。

## 方法

给定以正常行情为主的历史参考段、独立校准段和随后测试段，按时间顺序处理。对每个待评分时刻 t，仅取此前 L 条原始观测 W_t，调用 Kronos 预测一条未来 K 线。预测接口使用 `pred_len=1`、`sample_count=K`；多次采样平均由官方 `KronosPredictor` 完成。

对每个真实可观测维度 i：

\[
\tilde{\mathbf x}_t=f_{\mathrm{Kronos}}(W_t),\qquad
s_{t,i}=\left(\frac{x_{t,i}-\tilde x_{t,i}}{\sigma_i(W_t)+\varepsilon_i}\right)^2.
\]

标准差使用 `ddof=0`，和官方预测器的窗口总体标准差约定一致。评分直接使用真实观测值，不截断极端值；官方预测器内部的输入标准化与截断属于预测过程。epsilon 是每个评分特征的**原始单位**下的正数，不是方差、分位数或 TranAD coefficient。默认 `1e-5` 仅是数值示例，实际数据应按量纲设置。

在校准段按完全相同的滚动方式得到分数。每维取初始分位数 u，只对严格超过 u 的超额值 z=s-u 用 SciPy `genpareto.fit(..., floc=0)` 拟合 GPD。令校准样本数 n，超额数 N_u，GPD 形状 ξ、尺度 β：

\[
D=u+\frac{\beta}{\xi}\left[\left(\frac{N_u}{nq}\right)^\xi-1\right],\quad\xi\ne0;
\qquad D=u+\beta\log\frac{N_u}{nq},\quad\xi=0.
\]

实现用 GPD 的 inverse survival function 计算，以稳定处理 ξ 接近零的情况。q 表示 **单维无条件超越概率**，不是 GPD 条件尾概率。必须满足 `0 < q < 1-quantile`，且小于实测超额比例。每维默认至少 30 个超额样本；上尾不足、分数恒定或拟合无效时直接报错，不悄悄替换为经验阈值。

\[
y_{t,i}=\mathbb I(s_{t,i}\ge D_i),\qquad y_t=\bigvee_i y_{t,i}.
\]

测试期间阈值固定，未实现 SPOT 在线更新。逐维“或”会增大总体告警率；`calibration.json` 报告校准段联合告警比例。它是拟合样本上的诊断值，不能当作独立验证的误报率；需另取主要正常的验证段检查联合误报。尾部 q 也不是有限样本的误报保证，分数自相关和市场状态变化会影响尾部拟合。

## 安装

Python 3.10+：

```bash
git clone https://github.com/SansaZheng/KronosAnomalyDetection.git
cd KronosAnomalyDetection
python -m venv .venv
source .venv/bin/activate
pip install -e '.[kronos,test]'
git clone https://github.com/shiyu-coder/Kronos.git vendor/Kronos
```

`--kronos-repo` 指向官方 checkout。需要联网下载 Hugging Face 权重，或传入已下载权重目录。生产实验应固定官方 checkout 的提交及 `--model-revision`、`--tokenizer-revision`；运行清单记录实际 checkout 提交及传入参数。依赖版本如需严格复现实验，应另外冻结当前环境。CPU 可以运行，滚动逐点推理可能较慢。

## 输入数据与约束

三个 CSV 分别为 `reference.csv`、`calibration.csv`、`test.csv`，每行一根 K 线：

```csv
timestamps,open,high,low,close,volume,amount
2024-01-01T00:00:00Z,100,102,99,101,1000,101000
```

- OHLC 必需；volume、amount 可省略。各段真实输入列及顺序应一致，时间严格递增、无重复，参考段早于校准段，测试段晚于校准段。支持各段长度不同。
- 时区统一转成 UTC 后交给模型。无时区时间被视作 UTC，交易所本地时间请先转换。单次运行只处理一个资产、一种频率；用户需保证相同采样频率与交易日历，本实现不将隔夜/节假日间隔误作缺失，也不自动重采样。
- 不读取标签，不自动填补 NaN/inf，不按测试数据缩放或调阈值。CSV 附加的标签列不会传给模型。未执行 OHLC 合法性修复，以保留数据错误的异常幅度。
- 缺量时官方会补零，缺额时可能由价格与量估算。**这些维度必须从 `--features` 排除**。若 CSV 已含上游填补/推算的量额，程序无法辨别其来源，调用者也必须排除；本实现不支持逐行来源掩码。
- 参考段用于补足校准开头上下文；测试上下文由此前参考段、校准段及已发生的测试观测组成。历史不足 L 时 `scored=False`，预测、分数和异常结果为空，不能视为正常。
- 当前时刻值从不进入自己的预测窗口。历史中的异常会进入下一时刻上下文，这符合滚动定义，也可能导致后续持续高分。
- 预训练权重可能见过历史行情；本实现保证下游评分与校准流程因果性，不保证权重训练语料与评估数据完全无重叠。正式实验应报告模型训练截止时间和数据划分。

## 运行

完整、真实 OHLCVA：

```bash
kronos-ad --reference data/reference.csv \
  --calibration data/calibration.csv --test data/test.csv \
  --kronos-repo vendor/Kronos --output runs/experiment-001 \
  --lookback 128 --sample-count 5 --device cpu \
  --features open high low close volume amount \
  --epsilon 0.001 0.001 0.001 0.001 1 100 \
  --quantile 0.95 --risk 0.001 --min-exceedances 30
```

仅诊断 OHLC（可输入带量额数据，也可不带）：

```bash
kronos-ad --reference data/reference.csv \
  --calibration data/calibration.csv --test data/test.csv \
  --kronos-repo vendor/Kronos --output runs/ohlc-001 \
  --features open high low close --epsilon 0.001 0.001 0.001 0.001
```

示例参数尚未经金融数据调优，不代表最优配置。small/base 通常用 512 上下文，mini 应配套 `NeoQuasar/Kronos-Tokenizer-2k` 并设置对应上下文。程序要求 L 不超过显式 `max_context`，调用者还需保证模型与 tokenizer 匹配且实际模型支持该长度。

输出：

| 文件 | 内容 |
| --- | --- |
| `detections.csv` | 时间戳、是否评分、逐维预测/分数/阈值/异常、时间戳联合异常 |
| `calibration.json` | 配置、每维 GPD 参数与超额数、阈值、校准结束时间、校准联合告警比例、模型参数 |
| `run.json` | 命令参数、Kronos checkout 提交、Python 版本、依赖版本、输入 SHA256、测试告警率 |

测试告警率不等于误报率。没有事件/逐点标签时不报告 precision、recall、F1。维度定位不代表因果诊断；极端行情、状态切换、坏数据和预测失准都可能产生高分。

## 验证与合成示例

```bash
python -m pytest -q
python examples/generate_demo.py
kronos-ad --reference runs/demo-data/reference.csv \
  --calibration runs/demo-data/calibration.csv --test runs/demo-data/test.csv \
  --kronos-repo vendor/Kronos --output runs/demo-result \
  --features open high low close --epsilon 0.001 0.001 0.001 0.001
```

单元测试使用可审计的上一时刻预测桩，验证因果窗口、原始极端值评分、阈值冻结、预热标识、维度排除、尾部阈值与人工注入尖峰。它们不需要权重，也不是 Kronos 检测效果实验。合成示例的 amount 是推算值，因此示例命令只诊断 OHLC。真实 Kronos 权重的端到端效果需要另外运行上述命令并评估。

## 来源

- [Kronos 官方仓库](https://github.com/shiyu-coder/Kronos) 与 [预测器实现](https://github.com/shiyu-coder/Kronos/blob/master/model/kronos.py)
- [用户提供的中文说明](https://www.zdoc.app/zh/shiyu-coder/Kronos)
- [Kronos 论文：A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)
- Siffer et al., [Anomaly Detection in Streams with Extreme Value Theory, KDD 2017](https://doi.org/10.1145/3097983.3098144)
- [SciPy generalized Pareto distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html)

本项目的预测误差异常定义、逐维评分及组合规则来自本任务指定方案；POT 借鉴极值理论。本仓库不直接沿用 TranAD 两阶段重构误差或其参数/实验结论。

## 公开数据目录

新增[金融异常检测数据目录](data/README.md)：检索35个来源/数据家族，实际保存8个来源包、38个文件（约103.5MiB）。包含ULB、BankSim、pump-and-dump发布特征、S-FFSD、SD-FMM数组，以及明确标注为部分数据的SynthAML告警、ADMA事件区间和CardSim代码/输入。完整范围、未下载原因、逐文件SHA256和无监督评估约定均见数据目录；不是全部数据已下载的声明。

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py
python scripts/audit_datasets.py
```
