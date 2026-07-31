#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gadfinish200_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gadfinish200_%A.err

# Finish the gad_dt003 200pm sweep (was 259/300 due to prior timeout).
# Reruns full 300 at 200pm.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u scripts/method_single.py \
    --method gad_dt003 \
    --noise 0.20 \
    --n-samples 300 \
    --n-steps 2000 \
    --force-criterion force_norm \
    --force-threshold 0.01 \
    --split train \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/round3
