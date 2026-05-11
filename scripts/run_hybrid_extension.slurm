#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hybridext_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hybridext_%A_%a.err
#SBATCH --array=0-5

# Extension experiments (2026-05-10):
#   0-3: hybrid_eckart_swtrue tr=0.05 at {30, 50, 150, 200} pm
#        (fills undamped Eckart eig-switch noise sweep; pairs with the
#         damped Eckart eig-switch sweep we already have IRC for)
#   4-5: hybrid (no-Eckart) sf=0.05 tr=0.05 at {10, 100} pm
#        (tests whether higher force_switch ever triggers Newton in the
#         Cartesian frame — the no-Eckart Newton-never-fires finding)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

METHODS=(  "hybrid_eckart" "hybrid_eckart" "hybrid_eckart" "hybrid_eckart" "hybrid" "hybrid" )
SWITCHES=( "true"          "true"          "true"          "true"          "false"  "false"  )
SF=(       "1e-2"          "1e-2"          "1e-2"          "1e-2"          "0.05"   "0.05"   )
NOISES=(   "0.03"          "0.05"          "0.15"          "0.20"          "0.01"   "0.10"   )

T=$SLURM_ARRAY_TASK_ID
METHOD=${METHODS[$T]}
SWITCH=${SWITCHES[$T]}
SF_VAL=${SF[$T]}
NOISE=${NOISES[$T]}
NOISE_PM=$(python -c "print(int(round($NOISE*1000)))")

TR=0.05
GAD_DT=5e-3

TAG="${METHOD}_sw${SWITCH}_dt${GAD_DT}_tr${TR}_sf${SF_VAL}"
OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/hybrid_extension/${TAG}_${NOISE_PM}pm
mkdir -p "$OUTDIR"

echo "task=$T method=$METHOD switch=$SWITCH sf=$SF_VAL noise=${NOISE_PM}pm tr=$TR"

python -u scripts/hybrid_gad_newton_runner.py \
    --method "$METHOD" \
    --switch-by-eig "$SWITCH" \
    --switch-force "$SF_VAL" \
    --gad-dt "$GAD_DT" \
    --trust-radius "$TR" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 1000 \
    --split test \
    --output-dir "$OUTDIR"
