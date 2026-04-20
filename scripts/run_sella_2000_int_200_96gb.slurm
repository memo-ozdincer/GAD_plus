#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sella2kint200_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sella2kint200_%A.err

# Sella internal (no Eckart) 2000 steps @ 200pm with 96GB RAM.
# Prior 48GB also OOM'd at 10h with exit 125.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p /lustre07/scratch/memoozd/gadplus/runs/sella_2000

python -u scripts/sella_baseline.py \
    --fmax 0.01 \
    --noise 0.20 \
    --n-samples 300 \
    --max-steps 2000 \
    --split train \
    --internal \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/sella_2000
