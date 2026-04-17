#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircrigfull_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircrigfull_%A_%a.err
#SBATCH --array=0-5
#
# Full rigorous (Hratchian-Schlegel predictor-corrector) IRC sweep.
# 6 noise levels x 300 samples, on gad_dt003 converged TSs.
#
# Array tasks (noise -> survey_dir):
#   0: 10pm  -> round2
#   1: 30pm  -> round2
#   2: 50pm  -> round2
#   3: 100pm -> round3
#   4: 150pm -> round3
#   5: 200pm -> round3

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

OUTPUT_DIR="$SCRATCH/gadplus/runs/irc_rigorous_full"
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$OUTPUT_DIR"

NOISE_PMS=(10 30 50 100 150 200)
SURVEY_DIRS=(
    "$SCRATCH/gadplus/runs/round2"
    "$SCRATCH/gadplus/runs/round2"
    "$SCRATCH/gadplus/runs/round2"
    "$SCRATCH/gadplus/runs/round3"
    "$SCRATCH/gadplus/runs/round3"
    "$SCRATCH/gadplus/runs/round3"
)

NOISE_PM=${NOISE_PMS[$SLURM_ARRAY_TASK_ID]}
SURVEY_DIR=${SURVEY_DIRS[$SLURM_ARRAY_TASK_ID]}
N_SAMPLES=300

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | method=rigorous | noise_pm=$NOISE_PM | N=$N_SAMPLES | survey=$SURVEY_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/irc_validate.py" \
    --method rigorous \
    --source-method gad_dt003 \
    --noise-pm "$NOISE_PM" \
    --max-validate "$N_SAMPLES" \
    --irc-steps 500 \
    --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY_DIR" \
    --n-dataset-samples 300 \
    --output-dir "$OUTPUT_DIR" \
    --no-write-viewer-bundles

echo "Done."
