#!/bin/bash
# v2 (mean-distance) curriculum: MHEALTH -> Pamap2, OSDA, hard, n=1..9.
# Sequential to avoid OOM. MLflow exp suffix is "_mean".

cd "$(dirname "$0")"

METHODS=(OSBP TSFA UDA OVANet DANCE PPOT RAINCOAT UniOT UniJDOT)

for n in 1 2 3 4 5 6 7 8 9 ; do
  for m in "${METHODS[@]}"; do
    echo "==================== [n${n} fno_mean] $m ===================="
    python run_curriculum.py \
      --source_dataset MHEALTH \
      --target_dataset Pamap2 \
      --scenario OSDA \
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
