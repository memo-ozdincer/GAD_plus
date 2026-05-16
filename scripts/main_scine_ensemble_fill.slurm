#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_ensemble_fill_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_ensemble_fill_%A_%a.err
#SBATCH --array=0-3
#
# Fill ensemble gaps: descent1@150pm, precond05@{10,30,150}pm.
# Each cell: 287 samples, 15k steps, dt=0.007, fmax<0.01.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

case $SLURM_ARRAY_TASK_ID in
    0) NOISE_A=1.50; NOISE_PM=150; CFG="descent1";   EXTRA="--descent-until-nneg 1" ;;
    1) NOISE_A=0.10; NOISE_PM=10;  CFG="precond05";  EXTRA="--use-preconditioning --eig-floor 0.05" ;;
    2) NOISE_A=0.30; NOISE_PM=30;  CFG="precond05";  EXTRA="--use-preconditioning --eig-floor 0.05" ;;
    3) NOISE_A=1.50; NOISE_PM=150; CFG="precond05";  EXTRA="--use-preconditioning --eig-floor 0.05" ;;
esac

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_ensfill_${SLURM_ARRAY_JOB_ID}/${CFG}_noise${NOISE_PM}pm"
mkdir -p "$OUT"

echo "Task $SLURM_ARRAY_TASK_ID | cfg=$CFG | noise=${NOISE_PM}pm | flags: $EXTRA"

python -u scripts/gad_hparam_smoke.py \
    --backend scine --method DFTB0 \
    --noise "$NOISE_A" --n-samples 287 \
    --n-steps 15000 --dt 0.007 \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    $EXTRA \
    --output-dir "$OUT"
