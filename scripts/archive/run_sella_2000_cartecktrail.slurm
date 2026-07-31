#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sella2kcetrail_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sella2kcetrail_%A.err

# Retry Sella cart+Eckart 2000 steps @ 200pm (prior 8h run timed out at 270/300).
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
    --cartesian \
    --apply-eckart \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/sella_2000
