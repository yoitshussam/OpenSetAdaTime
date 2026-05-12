#!/bin/bash
# Resume of run_curriculum_pamap2_mhealth_mean.sh after the bash session died
# mid-loop at n=3 (UniOT done, UniJDOT not started).
#
# Done already in mlruns/:
#   n=1, n=2: all 7 methods x 5 seeds
#   n=3:      6/7 methods done (missing UniJDOT)
# To do:
#   n=3: UniJDOT only
#   n=4..8: all 7 methods
#
# Wrap in nohup / tmux / screen so SIGHUP from closed SSH session doesn't kill it again:
#   nohup ./run_curriculum_pamap2_mhealth_mean_resume.sh > pm_mh2_resume.log 2>&1 &
# or
#   tmux new -d -s pmmh 'bash run_curriculum_pamap2_mhealth_mean_resume.sh 2>&1 | tee pm_mh2_resume.log'

cd "$(dirname "$0")"

METHODS=( UDA OVANet DANCE PPOT RAINCOAT UniOT UniJDOT )

run_one () {
  local n="$1" m="$2"
  echo "==================== [n${n} fno_mean] $m ===================="
  python run_curriculum.py \
    --source_dataset Pamap2 \
    --target_dataset MHEALTH \
    --scenario UniDA \
    --strategy hard \
    --n_unknown "$n" \
    --da_method "$m" \
    --backbone FNO \
    --num_runs 5 \
    --rank_variant fno_mean \
    || echo "[n${n} fno_mean] $m FAILED — continuing"
}

# Finish n=3 — only UniJDOT remains.
run_one 3 UniJDOT

# n=4..8 full sweep.
for n in 4 5 6 7 8; do
  for m in "${METHODS[@]}"; do
    run_one "$n" "$m"
  done
  echo "==================== [n${n} fno_mean] done ===================="
done

echo "==================== RESUME ALL n done ===================="
