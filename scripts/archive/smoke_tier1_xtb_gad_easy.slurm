#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_gad_easy_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_xtb_gad_easy_%j.err
#
# xTB GAD smoke at MILD conditions: 10 pm noise, 150 steps, 5 samples.
# Earlier 100 pm / 500 step run timed out (12500 s/sample under MP).
# This budget: ~150 * 12s/step = 1800 s/sample. 5 samples on 10 parallel
# workers means all finish in one wave (~30 min wall).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_tier1_xtb_gad_easy_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID on $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK}"

python -u scripts/gad_smoke.py \
    --backend xtb --method gfn2 \
    --noise 0.1 --n-samples 5 \
    --n-steps 150 --dt 0.003 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
