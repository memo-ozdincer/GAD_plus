#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testsellatraj_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testsellatraj_%A_%a.err
#SBATCH --array=0-5

# Per-step Sella trajectory logging on test split, libdef config.
# 6 noise levels × n=287 × max_steps=2000.
# Used for the dynamics figure (Sella's full descent curves on test).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_sella_trajlog/carteck_libdef
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID noise=$NOISE"

python -u scripts/sella_trajlog.py \
    --noise "$NOISE" --split test --n-samples 287 \
    --max-steps 2000 --fmax 0.01 \
    --cartesian --apply-eckart \
    --delta0 0.1 --gamma 0.4 \
    --output-dir "$OUTDIR"
