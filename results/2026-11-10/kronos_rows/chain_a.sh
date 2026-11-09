#!/bin/bash
# chunk A: K-lines (study rows with 10 draws, then the boundary rows)
cd /home/claude/work/KronosAnomalyDetection && source /home/claude/work/venv/bin/activate && export HF_HUB_OFFLINE=1 PYTHONPATH=.:scripts
python scripts/run_kronos_rows.py --dataset xu_livshits_1h --study > results/2026-11-10/kronos_rows/logs/xu_livshits_1h_study.log 2>&1
python scripts/run_kronos_rows.py --dataset xu_livshits_1h > results/2026-11-10/kronos_rows/logs/xu_livshits_1h.log 2>&1
echo CHAIN_A_DONE >> results/2026-11-10/kronos_rows/logs/xu_livshits_1h.log
