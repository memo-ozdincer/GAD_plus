#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/main_xtb_sella_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/main_xtb_sella_%A_%a.err
#SBATCH --array=0-2
#
# xTB/GFN2 (via dxtb) + Sella libdef (Cartesian + Eckart, Hessian every
# step). Tuned delta0=0.02 (smaller than HIP libdef 0.1) because xTB
# Hessian eigenvalues are ~5x stiffer; with delta0=0.1 Sella drives
# geometries to where dxtb returns inf/NaN (see smoke findings).
#
# Same reduced scope as main_xtb_gad.slurm: 3 noise levels, 50 samples,
# max-steps 1000.

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

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_xtb_sella_${SLURM_ARRAY_JOB_ID}/noise${NOISE_PM}pm"
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) | noise=${NOISE_PM}pm"
echo "Out: $OUT"

python -u scripts/sella_smoke.py \
    --backend xtb --method gfn2 \
    --noise "$NOISE_A" --n-samples 50 \
    --max-steps 1000 --fmax 0.01 \
    --apply-eckart \
    --delta0 0.02 --gamma 0.4 --diag-every 1 \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
