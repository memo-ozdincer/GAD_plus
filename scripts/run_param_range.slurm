#!/bin/bash
# Launch parameter ranging: one SLURM job per (dt, k_track) pair.
# 15 parallel MIG jobs × 10 samples × 300 steps ≈ 3 min each.
#
# Usage: bash scripts/run_param_range.slurm

set -euo pipefail

PROJECT="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
LOG_DIR="$SCRATCH/gadplus/logs"
mkdir -p "$LOG_DIR" "$SCRATCH/gadplus/runs/param_range"

DT_VALUES="0.001 0.003 0.005 0.01 0.02"
K_VALUES="0 4 8"

echo "Submitting parameter range jobs..."

for DT in $DT_VALUES; do
    for K in $K_VALUES; do
        JOB_NAME="pr_dt${DT}_k${K}"
        JOB_ID=$(sbatch --parsable \
            --account=def-aspuru \
            --gpus=a100_2g.10gb:1 \
            --cpus-per-task=4 \
            --mem=16G \
            --time=1:00:00 \
            --job-name="$JOB_NAME" \
            --output="$LOG_DIR/${JOB_NAME}_%j.out" \
            --error="$LOG_DIR/${JOB_NAME}_%j.err" \
            --wrap="
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
source $PROJECT/.venv/bin/activate
export PYTHONPATH=$PROJECT/src:\$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export TMPDIR=$SCRATCH/gadplus/tmp
mkdir -p \$TMPDIR
python -u $PROJECT/scripts/param_range.py --dt $DT --k-track $K
")
        echo "  $JOB_NAME -> job $JOB_ID"
    done
done

echo ""
echo "15 jobs submitted. Monitor: squeue -u \$USER"
echo "Results: ls $SCRATCH/gadplus/runs/param_range/param_*.parquet"
