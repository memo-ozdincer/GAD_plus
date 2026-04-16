#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sella1k_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sella1k_%A_%a.err
#SBATCH --array=0-23
#
# Sella TS baselines with max_steps=1000 (fair comparison with GAD).
# 4 configs x 6 noise levels = 24 parallel jobs. 300 samples each.
#
# Configs:
#   0 = internal, no eckart     (Sella default recommendation)
#   1 = internal, eckart        (Eckart-projected Hessian + internal coords)
#   2 = cartesian, no eckart    (raw HIP Hessian in Cartesian)
#   3 = cartesian, eckart       (Eckart-projected Hessian in Cartesian)
#
# All use fmax=0.01, diag_every_n=1, exact HIP Hessian.
#
# Noise: 0=10pm 1=30pm 2=50pm 3=100pm 4=150pm 5=200pm
# Array: task_id = config_idx * 6 + noise_idx
#
# Submit:  sbatch scripts/run_sella_1000.slurm

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
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/sella_1000"

NOISE_LEVELS=(0.01 0.03 0.05 0.1 0.15 0.2)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
NOISE=${NOISE_LEVELS[$NOISE_IDX]}

# Config flags
case $CONFIG_IDX in
    0) COORD_FLAG="--internal";   ECKART_FLAG="";               DESC="internal" ;;
    1) COORD_FLAG="--internal";   ECKART_FLAG="--apply-eckart";  DESC="internal_eckart" ;;
    2) COORD_FLAG="--cartesian";  ECKART_FLAG="";               DESC="cartesian" ;;
    3) COORD_FLAG="--cartesian";  ECKART_FLAG="--apply-eckart";  DESC="cartesian_eckart" ;;
esac

echo "Job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID: config=$DESC noise=$NOISE max_steps=1000"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/sella_baseline.py" \
    --fmax 0.01 \
    --noise "$NOISE" \
    --n-samples 300 \
    --max-steps 1000 \
    --split train \
    $COORD_FLAG \
    $ECKART_FLAG \
    --output-dir "$SCRATCH/gadplus/runs/sella_1000"

echo "Done."
