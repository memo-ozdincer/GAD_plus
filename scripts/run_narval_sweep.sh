#!/bin/bash
# Submit a Hydra multirun sweep on Narval using MIG slices.
#
# This launches hundreds of independent jobs — one per (method, noise_level) combo.
# Each job gets a 2g.10gb MIG slice (10GB VRAM, ~$0.10/hr equivalent).
#
# Usage:
#   bash scripts/run_narval_sweep.sh
#
# Or from a reserved node:
#   salloc --account=rrg-aspuru --gpus=a100:1 --cpus-per-task=48 --mem=64G --time=12:00:00
#   # Then run individual experiments with srun:
#   srun python -m gadplus.orchestration.run search=gad_projected max_samples=100

set -euo pipefail

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

# Threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "=== Launching GAD_plus sweep on Narval ==="
echo "Each combo gets its own MIG slice job"
echo ""

python -m gadplus.orchestration.run --multirun \
    hydra/launcher=submitit_slurm \
    search=pure_gad,gad_tracked,gad_projected,gad_adaptive_dt,nr_gad_flipflop \
    starting.noise_levels_pm=0,1,3,5,10,15 \
    max_samples=300

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER"
echo "Results:  python scripts/analyze.py /lustre07/scratch/memoozd/gadplus/runs/"
