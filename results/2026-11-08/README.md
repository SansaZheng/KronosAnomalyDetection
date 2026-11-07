# Knob as a scoring channel: miss rate on subtle anomalies (stand-ins, 2026-11-08)

Objective 4 of the knowledge-boundary work, stated as a detection claim: *can the
knob lower the miss rate on subtle anomalies?*  Tested on four stand-in transformers,
one per financial-transaction dataset of the repository (`scripts/boundary_standin.py
--dataset ...`), with the protocol fixed in `scripts/boundary_missrate.py` before any
run.  Directories:

| Directory | Contents |
|---|---|
| `knowledge_boundary_standin_<dataset>/` | stand-in forecaster, probes, difference maps, knob search, interventions (same layout as `results/2026-11-07/knowledge_boundary_standin/`, which is the Sapienza 15S run) |
| `knowledge_boundary_missrate[_<dataset>]/` | `missrate.csv` (every arm), `increments.csv` (what a channel adds over B, L and B+L), `complementarity.csv`, `scores_test.csv.gz` (per-row channel scores), `missrate.png`, `run.log`; the Sapienza directory also keeps `run_A.log`, the first run before revision 1 |

Protocol (see the script docstring): baseline **B** = the stand-in's teacher-forced
NLL surprise; **L** = level magnitude max_j |z_j| (Robust-MAD analogue); knob channels
**K_label** (labelled boundary) and **K_error** (label-free error boundary); logistic and
CNN probes as comparators.  Channels combined by Tippett (pre-specified primary), Fisher,
and an equal false-positive budget OR in rank space.  Operating point 1% FPR on test
normal rows; miss rate = 1 - recall over labelled anomalies.  Subtle anomalies: those B
misses (relative) and those below the anomaly-median level magnitude (intrinsic).
Paired bootstrap over anomaly groups (or ~50 blocks of positives on single-stream data).

Revision 1 (after the first Sapienza run): unclipped z-scores for L, GPD-tail p-values
instead of plain conformal ranks (which tied at the minimum p under partition shift), and
the OR arm added.  The knob and probe channels are **supervised**; B and L are not.

## Miss rate at 1% FPR, label knob added to B

| Dataset (anomalies) | B alone | B+K Tippett (primary) | B+K Fisher | B or K, equal budget | K alone | logistic probe alone | L alone |
|---|---|---|---|---|---|---|---|
| Sapienza 15S (64) | 0.156 | 0.156 (+0/-0) | 0.109 (+3/-0; [0.000, +0.094]) | 0.156 (+5/-5) | 0.703 | 0.641 | 0.094 |
| BankSim (1,440) | 0.666 | 0.660 (+11/-3; [+0.001, +0.010]) | 0.634 (+70/-24; [+0.020, +0.043]) | **0.540** (+282/-101; [+0.103, +0.149]) | 0.710 | 0.646 | 0.437 |
| S-FFSD (2,394) | 1.000 | 1.000 | 1.000 | **0.825** (+418/-0; [+0.095, +0.261]) | 0.739 | 0.750 | 1.000 (tie artefact; AP 0.67) |
| ULB (77) | 0.403 | 0.403 | 0.390 (+1/-0) | 0.481 (+1/-7; [-0.169, 0.000]) | 0.987 | 0.987 | 0.974 |

Intrinsically subtle anomalies (below-median level magnitude), recall B -> B or K:
Sapienza 0.719 -> 0.750 [-0.125, +0.188]; **BankSim 0.281 -> 0.404 [+0.085, +0.162]**;
S-FFSD 0.003 -> 0.003; ULB 0.316 -> 0.237 [-0.191, +0.029].  On top of the level baseline
(BankSim, OR): L -> L or K 0.563 -> 0.635 [+0.057, +0.089], subtle half 0.083 -> 0.235
[+0.120, +0.184]; B or L -> B or L or K 0.642 -> 0.682 [+0.024, +0.056], subtle 0.248 ->
0.332 [+0.052, +0.118].

Complementarity: the knob recovers 60% / 43% / 26% / 3% of the anomalies B misses
(Sapienza / BankSim / S-FFSD / ULB) and 0-24% of those B detects; Spearman(K, B) on
anomalies -0.80 / -0.84 / -0.54 / -0.15.

## Reading

- The knob lowers the miss rate where the stand-in's hidden states encode the labelled
  boundary (balanced accuracy 0.70-0.82: Sapienza, BankSim, S-FFSD) **and** the channels
  are combined with an explicit per-channel false-positive budget.  It does nothing on
  ULB, where no boundary is learnable (probe 0.53, knob 0.51), and halving the budget
  there costs recall.
- The pre-specified Tippett combination changed nothing on any dataset: the NLL-surprise
  distribution shifts from calibration to test (the alpha = 0.01 calibration threshold
  alarms on 7-12% of test normals), and pooled p-values inherit the shift; the rank-space
  OR is immune to it.  Hidden-state channels were the stable ones (0.9-2.4% alarm rate).
- A plain logistic or CNN probe on the same hidden states always adds at least as much as
  the one- or two-dimensional knob.  The knob's value is the closed form and the
  interventions, not detection quality.
- On ULB the knob direction still "passes" the paper's intervention test (beats tangent
  and random controls) by inflating the predicted variance; a variance-inflation control
  is needed before an intervention result is read as mechanism.

Stand-in boundary summary (label boundary LR / MLP / CNN; knob k, dims, accuracy, random-k
mean, share of random subsets >= knob): Sapienza 0.820/0.813/0.766, k=1 (1) 0.797, 0.738,
14%; BankSim 0.712/0.712/0.708, k=2 (0, 26) 0.701, 0.698, 34%; S-FFSD 0.751/0.766/0.715,
k=2 (17, 9) 0.733, 0.718, 60%; ULB 0.552/0.506/0.526, k=1 (24) 0.506, 0.492, 32%.
Error boundary LR: 0.678 / 0.512 / 0.597 / 0.501.

None of these numbers describe Kronos; they describe stand-ins trained here.
`scripts/run_boundary_kronos.py` runs the same analysis on real weights.
