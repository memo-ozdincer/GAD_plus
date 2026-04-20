#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sella2k_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sella2k_%A_%a.err
#SBATCH --array=0-11

# Sella at 2000 steps with coords saved.
# 2 configs (internal-no-eckart, cartesian-eckart) x 6 noise levels.
#
# Array layout: task_id = config_idx * 6 + noise_idx
#   config 0 = internal (Sella default; no Eckart, internal dof-removal)
#   config 1 = cartesian + eckart (previous best)
# Noise: 0=10pm 1=30pm 2=50pm 3=100pm 4=150pm 5=200pm

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)
CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
NOISE=${NOISES[$NOISE_IDX]}

case $CONFIG_IDX in
    0) COORD_FLAG="--internal";   ECKART_FLAG="";               LABEL="internal" ;;
    1) COORD_FLAG="--cartesian";  ECKART_FLAG="--apply-eckart"; LABEL="cart_eckart" ;;
esac

OUTPUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/sella_2000"
mkdir -p "$OUTPUT_DIR"

echo "=== Sella 2000-step | $LABEL | noise=$NOISE ==="

python -u scripts/sella_baseline.py \
    --fmax 0.01 \
    --noise "$NOISE" \
    --n-samples 300 \
    --max-steps 2000 \
    --split train \
    $COORD_FLAG \
    $ECKART_FLAG \
    --output-dir "$OUTPUT_DIR"
