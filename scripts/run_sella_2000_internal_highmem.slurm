#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sella2kint_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sella2kint_%A_%a.err
#SBATCH --array=0-2

# Retry Sella internal at 2000 steps for 100/150/200pm with 48GB RAM
# (prior 16GB OOM'd).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.10 0.15 0.20)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}

mkdir -p /lustre07/scratch/memoozd/gadplus/runs/sella_2000

echo "=== Sella 2000-step internal (no eckart) | noise=$NOISE | 48GB RAM ==="

python -u scripts/sella_baseline.py \
    --fmax 0.01 \
    --noise "$NOISE" \
    --n-samples 300 \
    --max-steps 2000 \
    --split train \
    --internal \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/sella_2000
