#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sellaintretry_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sellaintretry_%A_%a.err
#SBATCH --array=0-1

# Retry Sella internal @ 150pm and 200pm with more memory.
# Original 60769799_1 and _2 OOM-died at 5h23m on C3H5NO2 samples.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.15 0.20)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}
OUT=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
mkdir -p "$OUT"

python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
    --n-samples 287 --max-steps 2000 --split test \
    --config-tag "internal_default" --output-dir "$OUT"
