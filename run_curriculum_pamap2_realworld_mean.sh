#!/bin/bash
# v2 (mean-distance) curriculum: Pamap2 -> RealWorld, OSDA, hard, n=1..8.
# Sequential to avoid OOM. MLflow exp suffix is "_mean".

cd "$(dirname "$0")"

METHODS=( UDA OVANet DANCE PPOT RAINCOAT UniOT UniJDOT)
#  
for n in 1 2 3 4  ; do
  for m in "${METHODS[@]}"; do
    echo "==================== [n${n} fno_mean] $m ===================="
    python run_curriculum.py \
      --source_dataset Pamap2 \
      --target_dataset RealWorld \
      --scenario UniDA \
      --strategy hard \
      --n_unknown "$n" \
      --da_method "$m" \
      --backbone FNO \
      --num_runs 5 \
      --rank_variant fno_mean \
      || echo "[n${n} fno_mean] $m FAILED — continuing"
  done
  echo "==================== [n${n} fno_mean] done ===================="
done
echo "==================== ALL n done ===================="
