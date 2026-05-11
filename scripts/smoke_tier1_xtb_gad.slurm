#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_gad_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_gad_%j.err
#
# Tier-1 smoke: gad_projected on xTB/GFN2 via dxtb.
# 5 test-split samples, noise=1.0 A (100 pm), 500 steps, 32 parallel workers.
# dxtb Hessian is ~9.5 s/step on 10-atom mols; expected wall ~25 min for 5 samples
# at 32-way parallelism (each sample is a single-threaded process).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_tier1_xtb_gad_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID on $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK}"

python -u scripts/gad_smoke.py \
    --backend xtb --method gfn2 \
    --noise 1.0 --n-samples 5 \
    --n-steps 500 --dt 0.003 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
