#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_sella_easy_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_sella_easy_%j.err
#
# xTB Sella smoke at MILD conditions: 10 pm noise, 150 max steps, 5 samples.
# Previous 100 pm run got NaN final-eval (Sella drove geometries to where
# dxtb produces inf/NaN). 10 pm is a much gentler test.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_tier1_xtb_sella_easy_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID on $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK}"

python -u scripts/sella_smoke.py \
    --backend xtb --method gfn2 \
    --noise 0.1 --n-samples 5 \
    --max-steps 150 --fmax 0.01 \
    --apply-eckart \
    --delta0 0.1 --gamma 0.4 --diag-every 1 \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
