#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_sella_irc15k_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_sella_irc15k_%A_%a.err
#SBATCH --array=0-5

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

SRC_DIR="/lustre07/scratch/memoozd/gadplus/runs/main_scine_sella15k_60868140/noise${NOISE_PM}pm"
SUMMARY=$(ls "$SRC_DIR"/summary_*.parquet | head -1)
OUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/scine_sella_irc15k_${SLURM_ARRAY_JOB_ID}/sella"
mkdir -p "$OUT_DIR"

python -u scripts/scine_irc_validate.py \
    --summary-parquet "$SUMMARY" \
    --traj-dir "$SRC_DIR" \
    --noise-pm "$NOISE_PM" \
    --method-tag "sella" \
    --output-dir "$OUT_DIR" \
    --max-irc-steps 500 \
    --rmsd-threshold 0.3 \
    --n-workers "$SLURM_CPUS_PER_TASK"
