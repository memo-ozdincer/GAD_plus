#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/multimode_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/multimode_%A_%a.err
#SBATCH --array=0-17

# Multi-mode GAD: 3 methods × 6 noise levels = 18 jobs
#
# multimode_all_neg  — flip all modes with λ < 0
# multimode_smooth   — sigmoid(-λ·50) weight per mode (differentiable)
# multimode_top2     — always flip v₁ and v₂

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate

OUTPUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/multimode"
mkdir -p "$OUTPUT_DIR"

METHODS=(multimode_all_neg multimode_smooth multimode_top2)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

N_NOISES=${#NOISES[@]}
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / N_NOISES))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % N_NOISES))

METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

echo "=== Multi-mode GAD: method=$METHOD noise=$NOISE steps=1000 ==="

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 1000 \
    --output-dir "$OUTPUT_DIR"

echo "=== Done ==="
