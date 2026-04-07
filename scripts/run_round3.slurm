#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/round3_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/round3_%A_%a.err
#SBATCH --array=0-23

# Round 3: 4 methods × 6 noise levels = 24 jobs
#
# blend_plain_k10  — λ₂-blend WITHOUT preconditioning, k=10 (wide)
# blend_plain_k50  — λ₂-blend WITHOUT preconditioning, k=50
# blend_plain_k100 — λ₂-blend WITHOUT preconditioning, k=100 (sharp)
# gad_dt002        — dt=0.002, 3000 steps
#
# 6hr wall time. 300 samples each.

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate

OUTPUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/round3"
mkdir -p "$OUTPUT_DIR"

METHODS=(
    blend_plain_k10
    blend_plain_k50
    blend_plain_k100
    gad_dt002
)

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

N_METHODS=${#METHODS[@]}
N_NOISES=${#NOISES[@]}

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / N_NOISES))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % N_NOISES))

METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

# gad_dt002 needs 3000 steps to match displacement budget
if [ "$METHOD" == "gad_dt002" ]; then
    N_STEPS=3000
else
    N_STEPS=1000
fi

echo "=== Round 3: method=$METHOD noise=$NOISE steps=$N_STEPS ==="
echo "Array task: $SLURM_ARRAY_TASK_ID (method_idx=$METHOD_IDX, noise_idx=$NOISE_IDX)"

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps "$N_STEPS" \
    --output-dir "$OUTPUT_DIR"

echo "=== Done ==="
