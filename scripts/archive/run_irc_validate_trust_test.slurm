#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irc_trust_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irc_trust_%A_%a.err
#SBATCH --array=0-2
#
# Smoke test: trust original converged labels, skip refinement, do not apply the criterion before IRC.

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/irc_validation_trust_converged_test"

NOISE_PMS=(0 10 50)
NOISE_PM=${NOISE_PMS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID, noise_pm=$NOISE_PM"

python -u "$PROJECT_DIR/scripts/irc_validate.py" \
    --noise-pm "$NOISE_PM" \
    --max-validate 5 \
    --irc-steps 100 \
    --rmsd-threshold 0.3 \
    --survey-dir "$SCRATCH/gadplus/runs/noise_survey_300" \
    --n-dataset-samples 300 \
    --output-dir "$SCRATCH/gadplus/runs/irc_validation_trust_converged_test"

echo "Done."
