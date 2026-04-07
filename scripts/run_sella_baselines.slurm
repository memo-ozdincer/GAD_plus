#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sella_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sella_%A_%a.err
#SBATCH --array=0-23
#
# Sella TS baselines: 4 configs x 6 noise levels = 24 parallel jobs.
# 300 samples each. Matches GAD experiment setup exactly.
#
# Configs:
#   0 = sella_internal_fmax0.01     (match our force threshold, internal coords)
#   1 = sella_internal_fmax0.03     (Sella default, internal coords)
#   2 = sella_internal_fmax0.001    (tight, for retroactive n_neg check)
#   3 = sella_cartesian_fmax0.01    (Cartesian coords, our threshold)
#
# Noise: 0=10pm 1=30pm 2=50pm 3=100pm 4=150pm 5=200pm
# Array: task_id = config_idx * 6 + noise_idx
#
# Submit:  sbatch scripts/run_sella_baselines.slurm

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/sella_baselines"

# Config array: fmax, coordinate type
FMAX_VALS=(0.01 0.03 0.001 0.01)
COORD_FLAGS=("--internal" "--internal" "--internal" "--cartesian")
NOISE_LEVELS=(0.01 0.03 0.05 0.1 0.15 0.2)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
FMAX=${FMAX_VALS[$CONFIG_IDX]}
COORD=${COORD_FLAGS[$CONFIG_IDX]}
NOISE=${NOISE_LEVELS[$NOISE_IDX]}

echo "Job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID: fmax=$FMAX coords=$COORD noise=$NOISE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/sella_baseline.py" \
    --fmax "$FMAX" \
    --noise "$NOISE" \
    --n-samples 300 \
    --max-steps 200 \
    --split train \
    $COORD \
    --output-dir "$SCRATCH/gadplus/runs/sella_baselines"

echo "Done."
