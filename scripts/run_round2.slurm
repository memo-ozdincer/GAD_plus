#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/round2_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/round2_%A_%a.err
#SBATCH --array=0-47

# Round 2 experiments: 8 methods × 6 noise levels = 48 jobs
#
# A1: adaptive_mm       — corrected adaptive dt (dt_base=0.002, Multi-Mode GAD params)
# A2: adaptive_mm2      — even more conservative (dt_base=0.001)
# A3: gad_dt003         — smaller fixed dt=0.003, 2000 steps
# A4: gad_no_clamp      — no displacement capping (max_disp=999)
# A5: adaptive_floor    — adaptive dt with higher floor (dt_min=1e-3)
# B:  precond_descent   — precond descent when n_neg≥2, precond GAD when n_neg<2 (diagnostic)
# C1: blend_k50         — blended preconditioned GAD, sigmoid(50·λ₂)
# C2: blend_k100        — blended preconditioned GAD, sigmoid(100·λ₂)

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate

OUTPUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/round2"
mkdir -p "$OUTPUT_DIR"

METHODS=(
    adaptive_mm
    adaptive_mm2
    gad_dt003
    gad_no_clamp
    adaptive_floor
    precond_descent
    blend_k50
    blend_k100
)

NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

N_METHODS=${#METHODS[@]}
N_NOISES=${#NOISES[@]}

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / N_NOISES))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % N_NOISES))

METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

# gad_dt003 needs 2000 steps to match displacement budget
if [ "$METHOD" == "gad_dt003" ]; then
    N_STEPS=2000
else
    N_STEPS=1000
fi

echo "=== Round 2: method=$METHOD noise=$NOISE steps=$N_STEPS ==="
echo "Array task: $SLURM_ARRAY_TASK_ID (method_idx=$METHOD_IDX, noise_idx=$NOISE_IDX)"

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps "$N_STEPS" \
    --output-dir "$OUTPUT_DIR"

echo "=== Done ==="
