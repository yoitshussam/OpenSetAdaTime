#!/bin/bash
# Queue A: MHEALTH->Pamap2 (n=1..9) then RealWorld->MHEALTH (n=1..8)
cd "$(dirname "$0")"
set -u
echo "[queue_a] start $(date)"
bash ./run_curriculum_mhealth_pamap2_mean.sh
echo "[queue_a] mhealth_pamap2 done $(date)"
bash ./run_curriculum_realworld_mhealth_mean.sh
echo "[queue_a] all done $(date)"
