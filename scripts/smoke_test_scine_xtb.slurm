#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/%j.err
#
# Smoke test for SCINE and xTB calculator backends.
# CPU only — no GPU needed.
#
# Submit: sbatch scripts/smoke_test_scine_xtb.slurm

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export MPLBACKEND=Agg
export TMPDIR="$SCRATCH/gadplus/tmp"
mkdir -p "$TMPDIR" "$(dirname "$SCRATCH/gadplus/logs")"

echo "Job $SLURM_JOB_ID on $(hostname)"
python -V

# Phase 1: smoke-test the bare adapters.
python "$PROJECT_DIR/scripts/smoke_test_scine_xtb.py"

# Phase 2: tiny orchestration end-to-end through Hydra. 2 samples,
# noise=0, gad_projected for ~30 steps each — exercises the full plumbing.
RUN_BASE="$SCRATCH/gadplus/runs/smoke_${SLURM_JOB_ID}"
mkdir -p "$RUN_BASE"

for backend in scine xtb; do
    echo
    echo "=== orchestration end-to-end: $backend ==="
    python -m gadplus.orchestration.run \
        calculator="$backend" \
        search=gad_projected \
        starting=noised_ts \
        starting.noise_levels_pm="[0]" \
        starting.seeds_per_level=1 \
        search.n_steps=30 \
        split=test \
        max_samples=2 \
        output_dir="$RUN_BASE/$backend"
done

echo
echo "Smoke test artifacts under: $RUN_BASE"
ls -la "$RUN_BASE"/*/ 2>/dev/null | head -40
