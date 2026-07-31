#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/round3x_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/round3x_%A_%a.err
#SBATCH --array=0-2

# gad_dt003 rerun at 100/150/200pm (the 3 that timed out in round2)

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate

OUTPUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/round3"
mkdir -p "$OUTPUT_DIR"

NOISES=(0.10 0.15 0.20)
NOISE=${NOISES[$SLURM_ARRAY_TASK_ID]}

echo "=== gad_dt003 rerun: noise=$NOISE steps=2000 ==="

python -u scripts/method_single.py \
    --method gad_dt003 \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 2000 \
    --output-dir "$OUTPUT_DIR"

echo "=== Done ==="
