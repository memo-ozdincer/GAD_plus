#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testgad_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testgad_%A_%a.err
#SBATCH --array=0-17

# GAD on T1x TEST split (287 samples). Three configs x 6 noise = 18 tasks.
#   gad_dt003_fmax : canonical (dt=0.003)
#   gad_dt005_fmax : the bigger-dt sweet spot (Euler stable, +2pp on train)
#   gad_adaptive_dt: eigenvalue-clamped adaptive step (the "fair" answer
#                    to Sella's QN trust-radius step)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

METHODS=(gad_dt003_fmax gad_dt005_fmax gad_adaptive_dt)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_set/${METHOD}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD noise=$NOISE split=test"

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 2000 \
    --split test \
    --output-dir "$OUTDIR"
