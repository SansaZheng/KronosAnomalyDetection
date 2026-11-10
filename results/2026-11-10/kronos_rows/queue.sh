#!/bin/bash
# Sequential, idempotent queue for the real-weight rerun (one process at a time: the container has 8 GB and 2 CPUs;
# two concurrent runs were OOM-killed on 2026-11-10).  Re-launch after the 2-hour background limit: finished steps
# are skipped, inference resumes from its partial checkpoint, the boundary analysis from its per-boundary pickle.
cd /home/claude/work/KronosAnomalyDetection && source /home/claude/work/venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONPATH=.:scripts MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2
R=results/2026-11-10; L=$R/kronos_rows/logs; mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/queue.log; }

rows() {                                   # Kronos inference on a dataset's rows
  local d=$1
  [ -f $R/kronos_rows/$d/test.npz ] && return 0
  log "rows $d start"; python scripts/run_kronos_rows.py --dataset $d >> $L/$d.log 2>&1; log "rows $d exit $?"
}
boundary() {
  local d=$1
  [ -f $R/knowledge_boundary_kronos_$d/results.json ] && return 0
  log "boundary $d start"; python scripts/run_boundary_kronos_rows.py --dataset $d >> $L/boundary_$d.log 2>&1; log "boundary $d exit $?"
}
missrate() {
  local d=$1
  [ -f $R/knowledge_boundary_missrate_kronos_$d/missrate.csv ] && return 0
  log "missrate $d start"; python scripts/boundary_missrate.py --dataset $d --model kronos > $L/missrate_$d.log 2>&1; log "missrate $d exit $?"
  python scripts/plot_missrate.py $R/knowledge_boundary_missrate_kronos_$d >> $L/missrate_$d.log 2>&1
}
explain() {
  local d=$1
  [ -f $R/knowledge_boundary_missrate_kronos_$d/missrate.csv ] || return 0
  log "explain $d start"; python scripts/explain_missrate.py $d --model kronos > $L/explain_$d.log 2>&1; log "explain $d exit $?"
}
study() {
  local d=$1
  [ -f $R/detector_study_$d/results.json ] && return 0
  log "study $d start"; python scripts/detector_study.py --dataset $d > $L/study_$d.log 2>&1; log "study $d exit $?"
}
analysis() { boundary $1; missrate $1; explain $1; study $1; }

# wait for anything already running from the earlier chains
while pgrep -f "scripts/(detector_study|run_boundary_kronos_rows|boundary_missrate|run_kronos_rows)\.py" > /dev/null; do sleep 30; done

analysis xu_livshits_1h
rows banksim; rows s_ffsd
analysis banksim
analysis s_ffsd
rows sapienza_15s;  analysis sapienza_15s
rows ulb_creditcard; analysis ulb_creditcard
python scripts/summarize_kronos_rerun.py > $L/summary.log 2>&1; log "summary exit $?"
log "QUEUE_DONE"
