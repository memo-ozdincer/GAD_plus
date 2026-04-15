#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/round4_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/round4_%A_%a.err
#SBATCH --array=0-5

# Quick test: 2 methods × 3 noise levels = 6 jobs
# descent_then_gad_2: descent until n_neg<=2, then GAD permanently
# descent_then_gad_3: descent until n_neg<=3, then GAD permanently

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate

OUTPUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/round4"
mkdir -p "$OUTPUT_DIR"

METHODS=(descent_then_gad_2 descent_then_gad_3)
NOISES=(0.05 0.10 0.20)

N_NOISES=${#NOISES[@]}
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / N_NOISES))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % N_NOISES))

METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

echo "=== Round 4: method=$METHOD noise=$NOISE steps=1000 ==="

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 1000 \
    --output-dir "$OUTPUT_DIR"

echo "=== Done ==="
