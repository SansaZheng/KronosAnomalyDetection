#!/bin/bash
# Model-size study: Kronos-mini / -small / -base on identical row subsets, one process at a time (2 CPUs, 8 GB).
# Idempotent: finished steps are skipped, inference resumes from its partial checkpoint, the boundary
# analysis from its per-boundary pickle.  Re-launch after the 2-hour background limit.
cd /home/claude/work/KronosAnomalyDetection && source /home/claude/work/venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONPATH=.:scripts MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2
S=results/2026-11-14; ALL=999999,999999,999999
log() { echo "$(date '+%F %T') $*" >> $S/queue.log; }
icap() { case $1 in xu_livshits_1h) echo 600;; ulb_creditcard) echo 100;; *) echo 200;; esac; }

run() {                                    # run <model> <dataset>
  local m=$1 d=$2
  export KRONOS_MODEL=$m KRONOS_RESULTS=$S/$m
  local R=$S/$m L=$S/$m/logs; mkdir -p $L
  if [ ! -f $R/kronos_rows/$d/test.npz ]; then
    log "$m rows $d start"; python scripts/run_kronos_rows.py --dataset $d --rows-from $S/mini/kronos_rows/$d --caps $ALL >> $L/rows_$d.log 2>&1; log "$m rows $d exit $?"
  fi
  if [ $d = xu_livshits_1h ] && [ ! -f $R/kronos_rows/$d/study_test.npz ]; then
    log "$m study-rows $d start"; python scripts/run_kronos_rows.py --dataset $d --study --draws 5 --rows-from $S/mini/kronos_rows/$d --caps $ALL >> $L/rows_$d.log 2>&1; log "$m study-rows $d exit $?"
  fi
  if [ ! -f $R/knowledge_boundary_kronos_$d/results.json ]; then
    log "$m boundary $d start"; python scripts/run_boundary_kronos_rows.py --dataset $d --intervention-cap $(icap $d) >> $L/boundary_$d.log 2>&1; log "$m boundary $d exit $?"
  fi
  if [ ! -f $R/knowledge_boundary_missrate_kronos_$d/missrate.csv ]; then
    log "$m missrate $d start"; python scripts/boundary_missrate.py --dataset $d --model kronos > $L/missrate_$d.log 2>&1; log "$m missrate $d exit $?"
    python scripts/explain_missrate.py $d --model kronos > $L/explain_$d.log 2>&1; log "$m explain $d exit $?"
  fi
  if [ ! -f $R/detector_study_$d/results.json ]; then
    log "$m study $d start"; python scripts/detector_study.py --dataset $d > $L/study_$d.log 2>&1; log "$m study $d exit $?"
  fi
}

while pgrep -f "scripts/(detector_study|run_boundary_kronos_rows|boundary_missrate|run_kronos_rows)\.py" > /dev/null; do sleep 30; done
for m in mini small base; do
  for d in xu_livshits_1h sapienza_15s s_ffsd banksim ulb_creditcard; do run $m $d; done
done
log "QUEUE_DONE"
