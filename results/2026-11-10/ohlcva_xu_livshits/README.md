# Hourly OHLCVA K-lines with pump labels (Xu & Livshits 2019 + B0R0koko), 2026-11-10

The first dataset of the study in Kronos's own format (OHLCVA bars).  Kronos weights were
still unreachable, so the Kronos slot of every forecast-based arm holds the stand-in
transformer trained on the reference candles (`results/2026-11-08/knowledge_boundary_standin_xu_livshits_1h/`);
nothing here is a statement about Kronos itself.

## Data

Source: `SansaZheng/Financial_unsuperviesdAD`, branch `claude/ohlcva-kline-data`,
`data/raw/xu_livshits_pumpdump/` converted by its `scripts/prepare_ohlcva_klines.py`
(CryptoCompare hourly candles, BTC-quoted, 2018-05-15 to 2018-10-29; PumpOlymp signals).
Loader: `fusion_data.load_rows('xu_livshits_1h')` (the table lives in the git-ignored
`.local_data/ohlcva/`).

| | |
|---|---|
| Candles | 3,685,842 on 921 coin:exchange series (CCCAGG aggregates dropped; series with < 160 rows dropped) |
| Exchanges | Cryptopia, Yobit, Bittrex, Binance |
| Zero-volume hours | 64.4% |
| Pump hours | 200 = 182 PumpOlymp signal hours + 18 Binance hours matched from the B0R0koko manifest (37 of its 498 announcements fall in the candle window) |
| Hu et al. Telegram log | 709 events from 2019 onward: no candles in hand for that period (Binance unreachable), so label-only, unused |
| Features | log return on the previous close, log high/low range, log1p volume and amount in units of the series' median positive volume / amount over the reference period |
| Split (by time) | reference < 2018-08-01 (1,702,929 candles, 47 pumps); calibration to 2018-09-10 (906,264, 57); test from 2018-09-11 (1,076,649, 96 pumps in 82 series) |

## Detector comparison (`scripts/ohlcva_pump_study.py`)

Test sample: 200,081 candles (uniform thinning + every pump hour; prevalence 0.048%);
calibration sample 24,057 (fit / validation = earlier / later series halves).  Paired
bootstrap over the 82 pump-bearing series, 2,000 resamples.

| Detector | AP | AUROC | R@1% FPR | Alarm rate | AP - fusion r1 (95% CI) |
|---|---|---|---|---|---|
| fusion r1 (stand-in in the Kronos slot) | 0.036 | 0.933 | 0.729 | 0.87% | - |
| fusion r1, no forecaster | 0.036 | 0.933 | 0.729 | 0.87% | identical (stand-in weight 0 on every feature) |
| fusion r2 | 0.072 | 0.935 | 0.740 | 0.89% | [+0.013, +0.105] |
| published POT, persistence forecast | 0.030 | 0.926 | 0.625 | 0.60% | [-0.021, +0.018]; R@1% [-0.220, +0.010] |
| published POT, stand-in forecast | 0.009 | 0.663 | 0.052 | **28.5%** | [-0.044, -0.004] |
| stand-in likelihood surprise | 0.035 | 0.910 | 0.573 | 1.0% (by construction) | [-0.019, +0.013]; R@1% [-0.271, -0.053] |
| Robust MAD | 0.088 | 0.938 | 0.760 | 1.14% | [+0.018, +0.147] |
| Isolation Forest | **0.292** | **0.940** | **0.771** | 1.11% | [+0.167, +0.354] |

Readings.  (1) The stand-in's point forecast gets zero stacking weight on all four bar
features, as the cached Kronos forecasts did on the transaction data; its likelihood is
well calibrated (3.11 vs 4.89 for persistence) but its mean is no better than the window
median.  (2) The published POT structure cannot take a model forecast on these candles:
64% of hours have zero volume, dead windows have zero standard deviation, and any
non-zero model forecast of volume produces an unbounded error score -- 28.5% alarms; with
the last value as forecast the error is zero on dead windows and the rule works (0.6%
alarms, 63% of pumps at 1% FPR), and the GPD tail cannot even be fitted for the volume
and amount features (empirical quantile fallback, logged).  (3) Round 2's long-history
channel helps on bars (AP doubles).  (4) Level baselines lead again; Isolation Forest is
far ahead in ranking quality (its top-ranked candles are mostly pumps) while every method
catches 57-77% of pump hours at 1% FPR, because a pump hour on an illiquid coin is a
|z| ~ 20 spike in range and volume.

## Knowledge boundary and the knob (`boundary_standin.py`, `boundary_missrate.py --dataset xu_livshits_1h`)

Stand-in: 240,040 training windows, calibration NLL 3.11 (epoch 2) vs 4.89 persistence.
Hidden states at the last of 32 context hours.  Error boundary (NLL above the calibration
90th percentile, 4.46): LR / MLP / CNN 0.713 / 0.713 / 0.705; knob k = 2 (dims 2, 15)
0.702, random pairs 0.679 with 12% matching; interventions beat both controls (p ~ 1e-7).
Pump boundary (the 32 hours before a pump hour): 0.677 / 0.662 / 0.708; knob k = 6
0.698, random 6-subsets 0.657 with 2% matching -- but its interventions do not beat the
tangent or random controls (p = 0.23 / 0.76 at tau = 0.1), so on candles the pump knob is
predictive without passing the paper's causal test.

Miss rate at 1% FPR on 24,096 test candles with 96 pump hours (82 bootstrap units):

| Arm | Miss rate | vs B (gained / lost; recall CI) |
|---|---|---|
| B: stand-in surprise | 0.427 | - |
| L: level magnitude | 0.229 | +23 / -4; [+0.098, +0.292] |
| K: pump knob alone | 0.802 | recovers 41% of B's misses, 4% of its catches; rho(K, B) = -0.27 |
| B + K, Tippett / Fisher | 0.427 / 0.427 | 0 / 0 |
| B or K, equal budget | 0.344 | +15 / -7; [0.000, +0.173] |
| B or error knob, equal budget | 0.333 | +17 / -8; [0.000, +0.191] |
| B or logistic probe, equal budget | 0.323 | +18 / -8; [+0.010, +0.204] |
| B or L, equal budget | **0.208** | +22 / -1; [+0.137, +0.301] |
| B or L or K, equal budget | 0.240 | vs B or L: [-0.075, +0.011] |
| low-magnitude half of the pumps (48): B -> B or K | 0.458 -> 0.438 | [-0.146, +0.102] |

Who the recovered pumps are (`explain_missrate.py xu_livshits_1h`, medians): the 55 pump
hours surprise catches and the 17 the knob recovers are equally large (range |z| 20 vs 22,
volume |z| 7.9 vs 8.5) and both sit on illiquid coins (84% zero-volume hours in the
context); they differ in the hours before: the recovered ones show rising volume over the
last eight hours (+0.54 vs 0.00 in z units) and higher context return and range (0.26 /
0.72 vs 0.07 / 0.24) -- a build-up that inflates the forecaster's predicted variance and
makes the pump hour unsurprising, and that the hidden state encodes.  Sixteen of the 17 are
Cryptopia pumps.  The 24 pumps neither channel sees are small (range |z| 1.1, volume 1.8)
on liquid Binance / Bittrex coins with continuous trading: an hourly bar does not resolve
them, which is a granularity problem, not a scoring one.

Why the knob adds less here than on BankSim: the build-up signal is weaker than the pump
hour's own spike, which a level statistic sees directly, so once the level channel is in
the OR there is nothing left for the knob to add; on the low-magnitude pumps the context
carries no build-up either.
