#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/startgeom_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/startgeom_%A_%a.err
#SBATCH --array=0-4
#
# Phase 3: Starting geometry comparison (gad_projected, Level 2).
# 4 starting geometries as array job, 50 samples x 300 steps each.
#
# Submit:  sbatch scripts/run_starting_geom.slurm

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/starting_geom_300"

START_TYPES=(noised_ts reactant product midpoint geodesic_mid)
START=${START_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID, start=$START"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/starting_geom.py" \
    --start "$START" \
    --n-samples 300 \
    --n-steps 300 \
    --dt 0.01 \
    --k-track 0 \
    --noise 0.01 \
    --split train \
    --output-dir "$SCRATCH/gadplus/runs/starting_geom_300"

echo "Done."
