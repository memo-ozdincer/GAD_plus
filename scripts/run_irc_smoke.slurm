#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircsmoke_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircsmoke_%A_%a.err
#SBATCH --array=0-3
#
# IRC smoke test: sella_hip vs rigorous on gad_dt003 converged TSs.
#
# Array (4 tasks = 2 methods x 2 noise levels):
#   Task 0: sella_hip @ 10pm
#   Task 1: sella_hip @ 50pm
#   Task 2: rigorous  @ 10pm
#   Task 3: rigorous  @ 50pm
#
# Small N_SAMPLES for fast iteration. Edit N_SAMPLES below when scaling up.

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

OUTPUT_DIR="$SCRATCH/gadplus/runs/irc_sellahip_rigorous"
SURVEY_DIR="$SCRATCH/gadplus/runs/round2"   # gad_dt003 converged TSs at 10/30/50pm
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$OUTPUT_DIR"

METHODS=(sella_hip rigorous)
NOISE_PMS=(10 50)
N_SAMPLES=5

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
NOISE_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

METHOD=${METHODS[$METHOD_IDX]}
NOISE_PM=${NOISE_PMS[$NOISE_IDX]}

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | method=$METHOD | noise_pm=$NOISE_PM | N=$N_SAMPLES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/irc_validate.py" \
    --method "$METHOD" \
    --noise-pm "$NOISE_PM" \
    --max-validate "$N_SAMPLES" \
    --irc-steps 500 \
    --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY_DIR" \
    --n-dataset-samples 300 \
    --output-dir "$OUTPUT_DIR" \
    --no-write-viewer-bundles

echo "Done."
