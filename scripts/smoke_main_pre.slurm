#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_main_pre_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_main_pre_%j.err
#
# Pre-flight smoke for the main SCINE/xTB sweep. Runs all 4 combos at
# LOW noise (10 pm) on 10 samples each, with proper k_track=8 and the
# tuned xTB dt/delta0. Has to demonstrate non-trivial convergence on
# SCINE before we commit to the big array.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

BASE="/lustre07/scratch/memoozd/gadplus/runs/smoke_main_pre_${SLURM_JOB_ID}"
echo "Job $SLURM_JOB_ID on $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Output base: $BASE"

# 1) SCINE GAD (the main row). 10 samples, 1000 steps, 10pm noise, dt=0.005.
echo; echo "=== SCINE/DFTB0 + plain GAD (10pm, 1000 steps, dt=0.005) ==="
python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise 0.10 --n-samples 10 \
    --n-steps 1000 --dt 0.005 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$BASE/scine_gad"

# 2) SCINE Sella libdef.
echo; echo "=== SCINE/DFTB0 + Sella libdef (10pm, 1000 steps) ==="
python -u scripts/sella_smoke.py \
    --backend scine --method DFTB0 \
    --noise 0.10 --n-samples 10 \
    --max-steps 1000 --fmax 0.01 --apply-eckart \
    --delta0 0.1 --gamma 0.4 --diag-every 1 \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$BASE/scine_sella"

# 3) xTB GAD with tuned dt=0.0005. Smaller scope: 5 samples, 300 steps.
echo; echo "=== xTB/GFN2 + plain GAD (10pm, 300 steps, dt=0.0005) ==="
python -u scripts/gad_smoke.py \
    --backend xtb --method gfn2 \
    --noise 0.10 --n-samples 5 \
    --n-steps 300 --dt 0.0005 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$BASE/xtb_gad"

# 4) xTB Sella with tuned delta0=0.02.
echo; echo "=== xTB/GFN2 + Sella libdef tuned (10pm, 300 steps, delta0=0.02) ==="
python -u scripts/sella_smoke.py \
    --backend xtb --method gfn2 \
    --noise 0.10 --n-samples 5 \
    --max-steps 300 --fmax 0.01 --apply-eckart \
    --delta0 0.02 --gamma 0.4 --diag-every 1 \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$BASE/xtb_sella"

echo
echo "All four smokes finished. Inspect parquets in $BASE/*/summary_*.parquet"
