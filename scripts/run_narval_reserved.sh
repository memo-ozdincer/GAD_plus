#!/bin/bash
# Reserved-node workflow for overnight experimentation.
#
# 1. Reserve a full A100 node for 12 hours
# 2. Run experiments sequentially with srun
# 3. Claude Code or cron-like loop checks progress, fixes bugs, resubmits
#
# Usage:
#   salloc --account=def-aspuru --gpus=a100:1 --cpus-per-task=12 --mem=64G --time=12:00:00
#   bash scripts/run_narval_reserved.sh
#
# Or for the full 4-GPU node (run 4 experiments in parallel):
#   salloc --account=def-aspuru --gpus=a100:4 --cpus-per-task=48 --mem=250G --time=12:00:00

set -euo pipefail

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
OUTBASE="$SCRATCH/gadplus/runs/reserved_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTBASE"

source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export TMPDIR="$SCRATCH/gadplus/tmp"
mkdir -p "$TMPDIR"

echo "=== Reserved node experiments ==="
echo "Output: $OUTBASE"
echo ""

# Run experiments sequentially — each one takes the full GPU
METHODS=(pure_gad gad_tracked gad_projected gad_adaptive_dt nr_gad_flipflop)
NOISE_LEVELS=(0 5 10 15)

for method in "${METHODS[@]}"; do
    for noise in "${NOISE_LEVELS[@]}"; do
        tag="${method}_noise${noise}pm"
        echo "--- Running: $tag ---"

        srun --overlap python -m gadplus.orchestration.run \
            search="$method" \
            starting=noised_ts \
            starting.noise_levels_pm="[$noise]" \
            max_samples=50 \
            output_dir="$OUTBASE/$tag" \
            2>&1 | tee "$OUTBASE/${tag}.log"

        echo "--- Done: $tag ---"
        echo ""
    done
done

echo "=== All experiments complete ==="
echo "Analyze: python scripts/analyze.py $OUTBASE"
python scripts/analyze.py "$OUTBASE"
