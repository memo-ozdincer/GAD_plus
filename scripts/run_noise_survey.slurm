#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/noise_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/noise_%A_%a.err
#SBATCH --array=0-8
#
# Phase 2: Noise robustness survey (gad_projected, Level 2).
# 9 noise levels as array job, 50 samples × 300 steps each.
#
# Submit:  sbatch scripts/run_noise_survey.slurm

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/noise_survey_300"

# Noise levels (Angstrom): 0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2
NOISE_LEVELS=(0.0 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2)
NOISE=${NOISE_LEVELS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID, noise=$NOISE A"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/noise_survey.py" \
    --noise "$NOISE" \
    --n-samples 300 \
    --n-steps 300 \
    --dt 0.01 \
    --k-track 0 \
    --split train \
    --output-dir "$SCRATCH/gadplus/runs/noise_survey_300"

echo "Done."
