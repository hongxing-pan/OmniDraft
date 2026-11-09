#!/bin/bash
# Runs the two follow-up experiments from OmniDraft_实验计划.md on 7B:
#   stage E: efficiency, 6 methods x 3 benchmarks x 30 questions x 3 repeats
#   stage S: sensitivity, 11 configs x WorldSense + DailyOmni, 30% kept
# One process per GPU; the efficiency tasks grab a free card from a slot pool,
# the sensitivity stage shards each dataset across all cards.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env_7b.sh"

LOGS=${LOGS:-/data/source_new02/olizhan/logs/senseff}
EFF_OUT=/data/source_new02/olizhan/results/omnidraft/efficiency_7b
SENS_OUT=/data/source_new02/olizhan/results/omnidraft/sensitivity
mkdir -p "$LOGS" "$EFF_OUT" "$SENS_OUT"
JOBS=${JOBS:-8}

stage_e () {
  # longest first so the tail is short
  local tasks=(
    "vme omniselect45" "vme omniselect30"
    "ws omniselect45" "ws omniselect30"
    "daily omniselect45" "daily omniselect30"
    "vme omnidraft45" "vme omnidraft30"
    "vme dycoke55" "vme full"
    "ws omnidraft45" "ws omnidraft30"
    "ws dycoke55" "ws full"
    "daily omnidraft45" "daily omnidraft30"
    "daily dycoke55" "daily full"
  )
  for t in "${tasks[@]}"; do
    set -- $t
    local ds=$1 m=$2
    (
      while :; do
        for i in $(seq 0 $((JOBS-1))); do
          if mkdir "$LOGS/.slot$i" 2>/dev/null; then
            printf '%s' "$ds/$m" > "$LOGS/.slot$i/owner"
            CUDA_VISIBLE_DEVICES=$i $PY "$HERE/run_efficiency.py" \
              --dataset "$ds" --method "$m" --model_path "$MODEL" --out "$EFF_OUT" \
              >> "$LOGS/eff__${ds}__${m}.log" 2>&1
            local rc=$?
            rm -rf "$LOGS/.slot$i"
            echo "[$(date +%H:%M:%S)] $ds/$m rc=$rc" >> "$LOGS/schedule.log"
            exit $rc
          fi
        done
        sleep 5
      done
    ) &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  done
  wait
}

stage_s () {
  for ds in ws daily; do
    echo "[$(date +%H:%M:%S)] sensitivity $ds start" >> "$LOGS/schedule.log"
    for s in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$s $PY "$HERE/run_sensitivity.py" \
        --dataset "$ds" --shard "$s" --nshards 8 \
        --model_path "$MODEL" --out "$SENS_OUT" \
        > "$LOGS/sens__${ds}__shard$s.log" 2>&1 &
    done
    wait
    echo "[$(date +%H:%M:%S)] sensitivity $ds done" >> "$LOGS/schedule.log"
  done
}

echo "[$(date +%H:%M:%S)] stage E start" | tee -a "$LOGS/schedule.log"
stage_e
echo "[$(date +%H:%M:%S)] stage E done" | tee -a "$LOGS/schedule.log"
echo "[$(date +%H:%M:%S)] stage S start" | tee -a "$LOGS/schedule.log"
stage_s
echo "ALL DONE $(date +%H:%M:%S)" | tee -a "$LOGS/schedule.log"
