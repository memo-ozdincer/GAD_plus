#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irc_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irc_%A_%a.err
#SBATCH --array=0-2
#
# Phase 5: IRC validation on converged TS from Phase 2 (300 samples).
# Array: 3 noise levels (0pm, 10pm, 50pm), 30 converged TS each.
#
# Submit:  sbatch scripts/run_irc_validate.slurm

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/irc_validation_300"

NOISE_PMS=(0 10 50)
NOISE_PM=${NOISE_PMS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID, noise_pm=$NOISE_PM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/irc_validate.py" \
    --noise-pm "$NOISE_PM" \
    --max-validate 30 \
    --irc-steps 100 \
    --rmsd-threshold 0.3 \
    --ts-pick best_nneg1 \
    --ts-force-criterion fmax \
    --ts-force-threshold 0.01 \
    --refine-ts \
    --refine-steps 300 \
    --refine-dt 0.003 \
    --refine-force-criterion fmax \
    --refine-force-threshold 0.005 \
    --skip-if-ts-poor \
    --survey-dir "$SCRATCH/gadplus/runs/noise_survey_300" \
    --n-dataset-samples 300 \
    --output-dir "$SCRATCH/gadplus/runs/irc_validation_300"

echo "Done."
