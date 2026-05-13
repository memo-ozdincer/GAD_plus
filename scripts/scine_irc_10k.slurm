#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_irc10k_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_irc10k_%A_%a.err
#SBATCH --array=0-5
#
# IRC TOPO validation on the SCINE/DFTB0 GAD 10k-step converged TSs.
# Array task = noise index (0=10pm ... 5=200pm).

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
NOISE_PM=${NOISES[$SLURM_ARRAY_TASK_ID]}

SRC_DIR="/lustre07/scratch/memoozd/gadplus/runs/main_scine_gad10k_60858200/noise${NOISE_PM}pm"
SUMMARY=$(ls "$SRC_DIR"/summary_*.parquet | head -1)
OUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/scine_irc10k_${SLURM_ARRAY_JOB_ID}/gad"
mkdir -p "$OUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID | gad 10k | noise=${NOISE_PM}pm"
echo "Source: $SUMMARY"

python -u scripts/scine_irc_validate.py \
    --summary-parquet "$SUMMARY" \
    --traj-dir "$SRC_DIR" \
    --noise-pm "$NOISE_PM" \
    --method-tag "gad" \
    --output-dir "$OUT_DIR" \
    --max-irc-steps 500 \
    --rmsd-threshold 0.3 \
    --n-workers "$SLURM_CPUS_PER_TASK"
