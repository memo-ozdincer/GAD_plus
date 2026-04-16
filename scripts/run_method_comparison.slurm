#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/methodcmp_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/methodcmp_%A_%a.err
#SBATCH --array=0-5
#
# Method comparison: 7 GAD variants across 6 noise levels.
# 6 noise levels x 7 methods x 50 samples x 300 steps.
# Each array task runs one noise level x all 7 methods sequentially.
#
# Submit:  sbatch scripts/run_method_comparison.slurm

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/gadplus/tmp"
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/method_comparison"

# Noise levels (Angstrom): focus on medium-to-high where ping-pong should help
NOISE_LEVELS=(0.01 0.03 0.05 0.1 0.15 0.2)
NOISE=${NOISE_LEVELS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID, noise=$NOISE A"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/method_comparison.py" \
    --noise "$NOISE" \
    --n-samples 50 \
    --n-steps 300 \
    --split train \
    --output-dir "$SCRATCH/gadplus/runs/method_comparison"

echo "Done."
