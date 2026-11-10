# 2026-11-10 — rerun of every experiment with the real Kronos-mini (offline weights)

Model: Kronos-mini + Kronos-Tokenizer-2k, loaded with `HF_HUB_OFFLINE=1` from the local bundle
(hashes in each `meta.json` / `results.json`). Report: https://claude.ai/artifact/CQ22Q83nJC5wYu8qH7h95Y

| Directory | Content |
|---|---|
| `kronos_rows/<d>/meta.json`, `kronos_rows/logs/` | inference cache metadata and logs (the `.npz` caches are git-ignored); `queue.sh` reruns everything, one process at a time, resumable |
| `detector_study_<d>/` | forecast skill, fusion / published POT / baselines, paired bootstrap |
| `knowledge_boundary_kronos_<d>/` | probes, knob search, interventions |
| `knowledge_boundary_missrate_kronos_<d>/` | channels B / L / K / P, combinations, miss rates at 1 % FPR |
| `subtle_anomalies_kronos.csv` | who the knob recovers (group medians) |
| `summary/SUMMARY.md` | all tables across datasets, real Kronos next to the stand-in |

`<d>` is one of `xu_livshits_1h`, `banksim`, `s_ffsd`, `sapienza_15s`, `ulb_creditcard`.

Headline (recall change of "surprise or knob" against the surprise alone, 1 % FPR, 95 % interval):
Sapienza +0.141 [+0.047, +0.234]; S-FFSD +0.103 [+0.069, +0.145]; BankSim +0.017 [+0.003, +0.030];
K-lines -0.104 [-0.182, -0.022]; ULB -0.182 [-0.263, -0.104]. The knob helps where a label knob
exists in the hidden state (anomalies announced by their context) and hurts where it does not.
