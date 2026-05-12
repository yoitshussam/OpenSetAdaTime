#!/bin/bash
# Queue B: Pamap2->MHEALTH (n=1..8) then Pamap2->RealWorld (n=1..4) then MHEALTH->RealWorld (n=1..4)
cd "$(dirname "$0")"
set -u
echo "[queue_b] start $(date)"
bash ./run_curriculum_pamap2_mhealth_mean.sh
echo "[queue_b] pamap2_mhealth done $(date)"
bash ./run_curriculum_pamap2_realworld_mean.sh
echo "[queue_b] pamap2_realworld done $(date)"
bash ./run_curriculum_mhealth_realworld_mean.sh
echo "[queue_b] all done $(date)"
