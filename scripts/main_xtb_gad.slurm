#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/main_xtb_gad_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/main_xtb_gad_%A_%a.err
#SBATCH --array=0-2
#
# xTB/GFN2 (via dxtb) + plain GAD. Tuned dt=0.0005 (10x smaller than the
# HIP/SCINE canonical dt=0.005) because xTB Hessian eigenvalues are ~5x
# stiffer; otherwise Euler integration blows up (see project memory
# project_scine_xtb_smoke_findings.md).
#
# Scope is reduced from HIP's full grid because dxtb costs ~10s/step on
# a 10-atom molecule:
#   - 3 noise levels (10, 100, 200 pm) instead of 6
#   - 50 samples instead of 287
#   - 1000 steps instead of 2000
# At ~10s/step that's ~10000 s/sample = 2.8 hr/sample. 32 workers means
# 2 waves of ~25 samples = ~6 hr per noise level.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES_PM=(10 100 200)
NOISES_A=(0.10 1.00 2.00)
NOISE_PM=${NOISES_PM[$SLURM_ARRAY_TASK_ID]}
NOISE_A=${NOISES_A[$SLURM_ARRAY_TASK_ID]}

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_xtb_gad_${SLURM_ARRAY_JOB_ID}/noise${NOISE_PM}pm"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) | noise=${NOISE_PM}pm"
echo "Out: $OUT"

python -u scripts/gad_smoke.py \
    --backend xtb --method gfn2 \
    --noise "$NOISE_A" --n-samples 50 \
    --n-steps 1000 --dt 0.0005 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
