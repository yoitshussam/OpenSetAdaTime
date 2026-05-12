#!/bin/bash
# Disposable: process 1 of parallel sweep_new resume.
# DANCE and SPADA already done — run remaining 5 of 9.
# Pairs heavier methods (TSFA, RAINCOAT, UniOT) with lighter ones.

set -e
cd "$(dirname "$0")"

METHODS=(UDA OSBP)

for m in "${METHODS[@]}"; do
  echo "==================== [p1] sweeping $m ===================="
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
echo "==================== [p1] done ===================="
