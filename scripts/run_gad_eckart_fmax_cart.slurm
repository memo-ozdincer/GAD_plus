#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gadfmaxcart_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gadfmaxcart_%A_%a.err
#SBATCH --array=0-5

# Cartesian-step rerun of the canonical gad_dt003_fmax (Eckart, fmax<0.01).
# Identical config to run_gad_eckart_fmax.slurm (same split default, 300
# samples, 2000 steps) so the only difference vs runs/gad_eckart_fmax is the
# projection.py change: GAD step now built in Cartesian coords instead of
# mass-weighted (commit 61119c2). New output dir keeps the old data intact.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}

mkdir -p /lustre07/scratch/memoozd/gadplus/runs/gad_eckart_fmax_cart

python -u scripts/method_single.py \
    --method gad_dt003_fmax \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 2000 \
    --output-dir /lustre07/scratch/memoozd/gadplus/runs/gad_eckart_fmax_cart
