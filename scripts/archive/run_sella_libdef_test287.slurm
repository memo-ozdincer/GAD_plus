#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sellalibtest_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sellalibtest_%A_%a.err
#SBATCH --array=0-5

# STAGE 1 for the Sella 5-tier split: re-run the headline Sella baseline
# (cartesian + Eckart, untuned library trust = delta0=0.1/gamma=0.4, Hess every
# step, 2000 steps) on the TEST split (n=287), saving coords_flat. The test-287
# Sella TS coords were not on disk; this regenerates them. IRC follows (stage 2).
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}
mkdir -p /lustre07/scratch/memoozd/gadplus/runs/sella_libdef_test287

python -u scripts/sella_baseline.py \
    --fmax 0.01 --noise "$NOISE" --n-samples 287 --split test \
    --max-steps 2000 --cartesian --apply-eckart \
    --delta0 0.1 --gamma 0.4 --diag-every 1 \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/sella_libdef_test287
