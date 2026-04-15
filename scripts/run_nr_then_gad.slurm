#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/nrtgad_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/nrtgad_%A_%a.err
#SBATCH --array=0-11

# NR→GAD one-way with tight displacement cap
# 2 methods × 6 noise levels = 12 jobs, all parallel

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate

OUTPUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/nr_then_gad"
mkdir -p "$OUTPUT_DIR"

METHODS=(nr_then_gad_cap01 nr_then_gad_cap005)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

N_NOISES=${#NOISES[@]}
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / N_NOISES))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % N_NOISES))

METHOD=${METHODS[$METHOD_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

echo "=== NR→GAD: method=$METHOD noise=$NOISE steps=1000 ==="

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 1000 \
    --output-dir "$OUTPUT_DIR"

echo "=== Done ==="
