#!/bin/bash
# Launch pure GAD sweep: all starting geometries × all noise levels.
#
# Submits individual MIG jobs for each config. Total: 14 jobs.
# Each job: 50 samples × 300 steps × ~0.06s/step ≈ 15 min
#
# Usage:
#   bash scripts/run_pure_gad_sweep.sh [DT] [K_TRACK]
#   bash scripts/run_pure_gad_sweep.sh 0.005 0    # defaults

set -euo pipefail

DT="${1:-0.005}"
K_TRACK="${2:-0}"

PROJECT="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
OUTPUT_DIR="$SCRATCH/gadplus/runs/pure_gad_sweep"
LOG_DIR="$SCRATCH/gadplus/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Pure GAD sweep: dt=$DT, k_track=$K_TRACK"
echo "Output: $OUTPUT_DIR"
echo ""

submit_job() {
    local START="$1"
    local NOISE_PM="$2"
    local JOB_NAME="gad_${START}_n${NOISE_PM}"

    sbatch --parsable \
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
python -u $PROJECT/scripts/pure_gad_sweep.py \
    --start $START \
    --noise-pm $NOISE_PM \
    --dt $DT \
    --k-track $K_TRACK \
    --n-samples 50 \
    --n-steps 300 \
    --output-dir $OUTPUT_DIR
"
}

# 1. Noised TS: 11 noise levels (0 to 200pm in 20pm steps = 0.0 to 2.0 Angstrom)
for NOISE in 0 20 40 60 80 100 120 140 160 180 200; do
    JOB_ID=$(submit_job noised_ts "$NOISE")
    echo "  Submitted: noised_ts noise=${NOISE}pm (job $JOB_ID)"
done

# 2. Reactant (no noise)
JOB_ID=$(submit_job reactant 0)
echo "  Submitted: reactant (job $JOB_ID)"

# 3. Product (no noise)
JOB_ID=$(submit_job product 0)
echo "  Submitted: product (job $JOB_ID)"

# 4. Midpoint R→T (closest to geodesic interpolation we have)
JOB_ID=$(submit_job midpoint_rt 0)
echo "  Submitted: midpoint_rt (job $JOB_ID)"

echo ""
echo "Total: 14 jobs submitted."
echo "Expected completion: ~15 min (MIG jobs start in minutes)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Analyze: python scripts/analyze.py $OUTPUT_DIR"
