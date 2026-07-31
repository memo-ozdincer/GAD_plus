#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/flagship_viz_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/flagship_viz_%j.err
#
# CPU-only batch export of representative gad_small_dt viewer bundles.
#
# Submit:
#   sbatch scripts/run_flagship_visualizations_cpu.slurm

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/gadplus/tmp"

mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/method_cmp_300/flagship_visualizations"

python -u "$PROJECT_DIR/scripts/make_flagship_visualizations.py" \
    --traj-dir "$SCRATCH/gadplus/runs/method_cmp_300" \
    --method gad_small_dt \
    --noise-levels 10 50 100 150 200 \
    --picks fast slow failure \
    --output-dir "$SCRATCH/gadplus/runs/method_cmp_300/flagship_visualizations"

echo "Done."
