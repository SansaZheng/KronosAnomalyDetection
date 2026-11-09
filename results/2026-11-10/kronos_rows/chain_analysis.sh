#!/bin/bash
# boundary analysis, miss-rate protocol, explanation and detector study for one dataset whose rows are cached
cd /home/claude/work/KronosAnomalyDetection && source /home/claude/work/venv/bin/activate && export HF_HUB_OFFLINE=1 PYTHONPATH=.:scripts
d=$1; L=results/2026-11-10/kronos_rows/logs
python scripts/run_boundary_kronos_rows.py --dataset $d > $L/boundary_$d.log 2>&1 && \
python scripts/boundary_missrate.py --dataset $d --model kronos > $L/missrate_$d.log 2>&1 && \
python scripts/plot_missrate.py results/2026-11-10/knowledge_boundary_missrate_kronos_$d >> $L/missrate_$d.log 2>&1 && \
python scripts/explain_missrate.py $d --model kronos > $L/explain_$d.log 2>&1
python scripts/detector_study.py --dataset $d > $L/study_$d.log 2>&1
echo "ANALYSIS_DONE $d $(date)" >> $L/chain.log
