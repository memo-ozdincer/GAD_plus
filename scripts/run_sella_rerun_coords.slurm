#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sellarerun_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sellarerun_%A_%a.err
#SBATCH --array=0-5
#
# Rerun best Sella config (cartesian+eckart, fmax=0.01, 1000 steps) with
# coords_flat + atomic_nums appended. Enables downstream sella_hip IRC
# validation on Sella-found TSs.

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

OUTPUT_DIR="$SCRATCH/gadplus/runs/sella_1000_coords"
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$OUTPUT_DIR"

NOISE_LEVELS=(0.01 0.03 0.05 0.1 0.15 0.2)
NOISE=${NOISE_LEVELS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | cart+eckart fmax=0.01 noise=$NOISE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/sella_baseline.py" \
    --fmax 0.01 \
    --noise "$NOISE" \
    --n-samples 300 \
    --max-steps 1000 \
    --split train \
    --cartesian \
    --apply-eckart \
    --output-dir "$OUTPUT_DIR"

echo "Done."
