#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/nrtgad_quick_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/nrtgad_quick_%A_%a.err
#SBATCH --array=0-1

# Quick test: 2 methods × 100pm only × 100 samples

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate

OUTPUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/nr_then_gad_quick"
mkdir -p "$OUTPUT_DIR"

METHODS=(nr_then_gad_cap01 nr_then_gad_cap005)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "=== QUICK: method=$METHOD noise=0.10 samples=100 steps=1000 ==="

python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise 0.10 \
    --n-samples 100 \
    --n-steps 1000 \
    --output-dir "$OUTPUT_DIR"

echo "=== Done ==="
