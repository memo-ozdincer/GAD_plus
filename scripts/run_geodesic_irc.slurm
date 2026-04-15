#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/geod_irc_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/geod_irc_%A_%a.err
#SBATCH --array=0-3
#
# Geodesic midpoint starting geometry + IRC validation at scale.
#
# Array layout (4 jobs):
#   0: Geodesic midpoint, gad_small_dt, 300 samples, 1000 steps
#   1: IRC validation, noise=10pm, 30 converged TS
#   2: IRC validation, noise=50pm, 30 converged TS
#   3: IRC validation, noise=100pm, 30 converged TS
#
# Submit:  sbatch scripts/run_geodesic_irc.slurm

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs"

T=$SLURM_ARRAY_TASK_ID

echo "Job $SLURM_JOB_ID, task $T"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ $T -eq 0 ]; then
    # Geodesic midpoint starting geometry
    mkdir -p "$SCRATCH/gadplus/runs/geodesic_mid"
    python -u "$PROJECT_DIR/scripts/starting_geom.py" \
        --start geodesic_mid \
        --n-samples 300 \
        --n-steps 1000 \
        --dt 0.005 \
        --k-track 0 \
        --split train \
        --output-dir "$SCRATCH/gadplus/runs/geodesic_mid"
else
    # IRC validation
    NOISE_PMS=(10 50 100)
    NOISE_PM=${NOISE_PMS[$((T - 1))]}
    mkdir -p "$SCRATCH/gadplus/runs/irc_validation_300"
    python -u "$PROJECT_DIR/scripts/irc_validate.py" \
        --noise-pm "$NOISE_PM" \
        --max-validate 30 \
        --irc-steps 200 \
        --rmsd-threshold 0.5 \
        --ts-pick best_nneg1 \
        --ts-force-criterion fmax \
        --ts-force-threshold 0.01 \
        --refine-ts \
        --refine-steps 600 \
        --refine-dt 0.003 \
        --refine-force-criterion fmax \
        --refine-force-threshold 0.006 \
        --skip-if-ts-poor \
        --survey-dir "$SCRATCH/gadplus/runs/noise_survey_300" \
        --n-dataset-samples 300 \
        --output-dir "$SCRATCH/gadplus/runs/irc_validation_300"
fi

echo "Done."
