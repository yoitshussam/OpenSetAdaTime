#!/bin/bash
# v2 (mean-distance) curriculum: RealWorld -> Pamap2, OSDA, hard, n=1..4.
# Sequential to avoid the OOM cascade we hit with 4 parallel n-scripts.
# MLflow exp suffix is "_mean" so these don't clobber v1 results.

cd "$(dirname "$0")"

METHODS=(RAINCOAT  UDA OVANet DANCE PPOT UniOT UniJDOT)
# METHODS=(UniOT PPOT)

for n in  1 2 3 4 ; do
  for m in "${METHODS[@]}"; do
    echo "==================== [n${n} fno_mean] $m ===================="
    python run_curriculum.py \
      --source_dataset RealWorld \
      --target_dataset MHEALTH \
      --scenario UniDA  \
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
