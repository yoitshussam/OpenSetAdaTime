#!/bin/bash
# Disposable: process 2 of parallel sweep_new resume.
# DANCE and SPADA already done — run remaining 4 of 9.
# Pairs heavier methods (PPOT, UniJDOT) with lighter ones.

set -e
cd "$(dirname "$0")"

METHODS=(UniJDOT )

for m in "${METHODS[@]}"; do
  echo "==================== [p2] sweeping $m ===================="
  python main_sweep.py \
    --da_method "$m" \
    --source_dataset RealWorld_male \
    --target_dataset RealWorld_female \
    --backbone FNO \
    --exp_name sweep_new \
    --num_runs 3 \
    --num_sweeps 15 \
    --hp_search_strategy bayes \
    --metric_to_minimize H_score \
    --uniDA True
done
echo "==================== [p2] done ===================="
