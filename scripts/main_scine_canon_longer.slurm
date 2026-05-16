#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_canon_longer_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_canon_longer_%A_%a.err
#SBATCH --array=0-3
#
# Canon GAD (dt=0.007) with 3× step budget (45000 steps) at hard noise levels.
# Tests: is the 3-variant ensemble's lift mechanism-based or compute-based?
# If canon×3 ≈ canon, ensemble lift is mechanism-based (paper-worthy).
# If canon×3 ≈ ensemble, ensemble lift is just compute (less paper-worthy).
#
# 4 tasks: 30pm, 50pm, 100pm, 150pm. fmax<0.01 criterion.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES_PM=(30 50 100 150)
NOISES_A=(0.30 0.50 1.00 1.50)

NOISE_PM=${NOISES_PM[$SLURM_ARRAY_TASK_ID]}
NOISE_A=${NOISES_A[$SLURM_ARRAY_TASK_ID]}

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_canonlong_${SLURM_ARRAY_JOB_ID}/canon45k_noise${NOISE_PM}pm"
mkdir -p "$OUT"

echo "Task $SLURM_ARRAY_TASK_ID | canon dt=0.007 × 45000 steps | noise=${NOISE_PM}pm"

python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise "$NOISE_A" --n-samples 287 \
    --n-steps 45000 --dt 0.007 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
