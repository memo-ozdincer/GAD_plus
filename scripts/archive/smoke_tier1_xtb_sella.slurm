#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_sella_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_sella_%j.err
#
# Tier-1 smoke: Sella cart+Eckart on xTB/GFN2 via dxtb.
# 5 test-split samples, noise=1.0 A (100 pm), 500 steps, 32 parallel workers.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_tier1_xtb_sella_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID on $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK}"

python -u scripts/sella_smoke.py \
    --backend xtb --method gfn2 \
    --noise 1.0 --n-samples 5 \
    --max-steps 500 --fmax 0.01 \
    --apply-eckart \
    --delta0 0.1 --gamma 0.4 --diag-every 1 \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
