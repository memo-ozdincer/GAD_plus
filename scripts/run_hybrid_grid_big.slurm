#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hybgrid_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hybgrid_%A_%a.err
#SBATCH --array=0-143
#
# Big hybrid HPO seed grid at noise = {0, 30} pm, fmax<0.05, n=287 test split.
#
# Grid (144 cells):
#   method        ∈ {hybrid_eckart, hybrid_damped_eckart}              (2)
#   switch-by-eig ∈ {true, false}                                       (2)
#   gad-dt        ∈ {0.002, 0.005, 0.010}                               (3)
#   trust-radius  ∈ {0.005, 0.01, 0.02, 0.05, 0.10, 0.20}               (6)
#   noise (pm)    ∈ {0, 30}                                             (2)
#
# Index decomposition (fastest -> slowest changing):
#   noise_i    = T %  2
#   tr_i       = (T /  2) % 6
#   dt_i       = (T / 12) % 3
#   switch_i   = (T / 36) % 2
#   method_i   = (T / 72) % 2

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

METHODS=(hybrid_eckart hybrid_damped_eckart)
SWITCHES=(true false)
DTS=(0.002 0.005 0.010)
TRS=(0.005 0.01 0.02 0.05 0.10 0.20)
NOISES=(0.0 0.03)
NOISE_PM=(0 30)

T=$SLURM_ARRAY_TASK_ID
NOISE_I=$(( T % 2 ))
TR_I=$(( (T / 2) % 6 ))
DT_I=$(( (T / 12) % 3 ))
SWITCH_I=$(( (T / 36) % 2 ))
METHOD_I=$(( (T / 72) % 2 ))

METHOD=${METHODS[$METHOD_I]}
SWITCH=${SWITCHES[$SWITCH_I]}
DT=${DTS[$DT_I]}
TR=${TRS[$TR_I]}
NOISE=${NOISES[$NOISE_I]}
NPM=${NOISE_PM[$NOISE_I]}

TAG="${METHOD}_sw${SWITCH}_dt${DT}_tr${TR}_${NPM}pm"
OUT=/lustre07/scratch/memoozd/gadplus/runs/hybrid_grid_big/${TAG}
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs

echo "task=$T  method=$METHOD  switch_eig=$SWITCH  dt=$DT  tr=$TR  noise=${NPM}pm  fmax<0.05"

python -u scripts/hybrid_gad_newton_runner.py \
    --method "$METHOD" \
    --switch-by-eig "$SWITCH" \
    --gad-dt "$DT" \
    --trust-radius "$TR" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 2000 \
    --split test \
    --force-threshold 0.05 \
    --start-from ts_noised \
    --output-dir "$OUT"
