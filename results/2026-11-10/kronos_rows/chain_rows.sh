#!/bin/bash
# Kronos inference on the boundary rows of the given datasets (sequential, resumable)
cd /home/claude/work/KronosAnomalyDetection && source /home/claude/work/venv/bin/activate && export HF_HUB_OFFLINE=1 PYTHONPATH=.:scripts
for d in "$@"; do
  python scripts/run_kronos_rows.py --dataset $d >> results/2026-11-10/kronos_rows/logs/$d.log 2>&1
  echo "ROWS_DONE $d $(date)" >> results/2026-11-10/kronos_rows/logs/chain.log
done
