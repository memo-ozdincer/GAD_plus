#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_irc_v2_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_irc_v2_%A_%a.err
#SBATCH --array=0-11
#
# v2 IRC validation (IRC + post-IRC BFGS relaxation) on the GAD 15k and
# Sella 15k SCINE runs.
# Array layout: tid = method_idx * 6 + noise_idx
#   method_idx 0 = gad (main_scine_gad15k_60865063)
#   method_idx 1 = sella (main_scine_sella15k_60868140)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(10 30 50 100 150 200)
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 6))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 6))
NOISE_PM=${NOISES[$NOISE_IDX]}

case $METHOD_IDX in
    0) SRC_DIR="/lustre07/scratch/memoozd/gadplus/runs/main_scine_gad15k_60865063/noise${NOISE_PM}pm"
       TAG="gad" ;;
    1) SRC_DIR="/lustre07/scratch/memoozd/gadplus/runs/main_scine_sella15k_60868140/noise${NOISE_PM}pm"
       TAG="sella" ;;
esac

SUMMARY=$(ls "$SRC_DIR"/summary_*.parquet | head -1)
OUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/scine_irc_v2_${SLURM_ARRAY_JOB_ID}/${TAG}"
mkdir -p "$OUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID | tag=$TAG | noise=${NOISE_PM}pm"
echo "Source: $SUMMARY"

python -u scripts/scine_irc_validate_v2.py \
    --summary-parquet "$SUMMARY" \
    --traj-dir "$SRC_DIR" \
    --noise-pm "$NOISE_PM" \
    --method-tag "$TAG" \
    --output-dir "$OUT_DIR" \
    --max-irc-steps 500 \
    --max-relax-steps 500 \
    --relax-fmax 0.001 \
    --rmsd-threshold 0.3 \
    --n-workers "$SLURM_CPUS_PER_TASK"
