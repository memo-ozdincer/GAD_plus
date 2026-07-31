#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_irc_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_irc_%A_%a.err
#SBATCH --array=0-11
#
# IRC TOPO validation on SCINE/DFTB0 converged TSs.
# Array layout: task_id = method_idx * 6 + noise_idx
#   method 0 = SCINE GAD          (sub-dir main_scine_gad_60772085)
#   method 1 = SCINE Sella libdef (sub-dir main_scine_sella_60772086)
# Noise: 0=10pm 1=30pm 2=50pm 3=100pm 4=150pm 5=200pm

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
    0)  SRC_DIR="/lustre07/scratch/memoozd/gadplus/runs/main_scine_gad_60772085/noise${NOISE_PM}pm"
        TAG="gad" ;;
    1)  SRC_DIR="/lustre07/scratch/memoozd/gadplus/runs/main_scine_sella_60772086/noise${NOISE_PM}pm"
        TAG="sella" ;;
esac

SUMMARY=$(ls "$SRC_DIR"/summary_*.parquet | head -1)
OUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/scine_irc_${SLURM_ARRAY_JOB_ID}/${TAG}"
mkdir -p "$OUT_DIR"

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | method=$TAG | noise=${NOISE_PM}pm"
echo "Source summary: $SUMMARY"
echo "Output: $OUT_DIR"

python -u scripts/scine_irc_validate.py \
    --summary-parquet "$SUMMARY" \
    --traj-dir "$SRC_DIR" \
    --noise-pm "$NOISE_PM" \
    --method-tag "$TAG" \
    --output-dir "$OUT_DIR" \
    --max-irc-steps 500 \
    --rmsd-threshold 0.3 \
    --n-workers "$SLURM_CPUS_PER_TASK"
