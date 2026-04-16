#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irc3way_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irc3way_%A_%a.err
#SBATCH --array=0-8
#
# Three-way IRC comparison on the converged-TS set.
#
# Array layout (9 tasks total, 3 methods x 3 noise levels):
#   Task ID / 3 -> method index: 0=sella_baseline, 1=sella_hip, 2=rigorous
#   Task ID % 3 -> noise index:  0=0pm,            1=10pm,     2=50pm
#
# All outputs land in the same directory with method in the filename:
#   irc_validation_{method}_{noise}pm.parquet
#
# Submit:  sbatch scripts/run_irc_validate_three_way.slurm

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/gadplus/tmp"

OUTPUT_DIR="$SCRATCH/gadplus/runs/irc_validation_three_way"
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$OUTPUT_DIR"

METHODS=(sella_baseline sella_hip rigorous)
NOISE_PMS=(0 10 50)

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
NOISE_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

METHOD=${METHODS[$METHOD_IDX]}
NOISE_PM=${NOISE_PMS[$NOISE_IDX]}

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | method=$METHOD | noise_pm=$NOISE_PM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/irc_validate.py" \
    --method "$METHOD" \
    --noise-pm "$NOISE_PM" \
    --max-validate 30 \
    --irc-steps 500 \
    --rmsd-threshold 0.3 \
    --survey-dir "$SCRATCH/gadplus/runs/noise_survey_300" \
    --n-dataset-samples 300 \
    --output-dir "$OUTPUT_DIR"

echo "Done."
