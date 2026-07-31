#!/bin/bash
#SBATCH --job-name=t1x_grid_aggregate
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_grid_aggregate_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_grid_aggregate_%j.err

# Run only after every matched raw and endpoint job succeeds.  This is a pure
# reader: it creates the denominator-explicit Markdown/JSON comparison table
# from the immutable raw and endpoint records.
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
export GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT_ROOT=/scratch/memoozd/gadplus/analysis/t1x-gxtb-matched-noise-grid
mkdir -p "$OUT_ROOT"
"$PYTHON" "$PROJECT_DIR/scripts/aggregate_t1x_gxtb_noise_grid.py" \
  "$PROJECT_DIR/experiments/t1x_gxtb_matched_noise_grid_manifest.json" \
  --output-dir "$OUT_ROOT"
echo "output_root=$OUT_ROOT"
