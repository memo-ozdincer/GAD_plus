#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_irc_ens_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_irc_ens_%A_%a.err
#SBATCH --array=0-6
#
# IRC TOPO validation on the descent1 and precond05 converged TSs.
# Answers: does ensemble coverage translate to TOPO (T1x-connected) lift?
#
# Tasks:
#   0 descent1 @ 10pm
#   1 descent1 @ 30pm
#   2 descent1 @ 50pm
#   3 descent1 @ 100pm
#   4 precond05 @ 50pm
#   5 precond05 @ 100pm
#   6 precond05 @ 200pm

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
    0) CFG="descent1";  NPM=10;  SRC="/lustre07/scratch/memoozd/gadplus/runs/main_scine_combos_61057565/descent1_noise10pm" ;;
    1) CFG="descent1";  NPM=30;  SRC="/lustre07/scratch/memoozd/gadplus/runs/main_scine_combos_61057565/descent1_noise30pm" ;;
    2) CFG="descent1";  NPM=50;  SRC="/lustre07/scratch/memoozd/gadplus/runs/main_scine_hparam_61051294/descent1_noise50pm" ;;
    3) CFG="descent1";  NPM=100; SRC="/lustre07/scratch/memoozd/gadplus/runs/main_scine_hparam_61051294/descent1_noise100pm" ;;
    4) CFG="precond05"; NPM=50;  SRC="/lustre07/scratch/memoozd/gadplus/runs/main_scine_hparam_61051294/precond05_noise50pm" ;;
    5) CFG="precond05"; NPM=100; SRC="/lustre07/scratch/memoozd/gadplus/runs/main_scine_hparam_61051294/precond05_noise100pm" ;;
    6) CFG="precond05"; NPM=200; SRC="/lustre07/scratch/memoozd/gadplus/runs/main_scine_hparam_61051294/precond05_noise200pm" ;;
esac

SUMMARY=$(ls "$SRC"/summary_*.parquet | head -1)
TAG="${CFG}"
OUT_DIR="/lustre07/scratch/memoozd/gadplus/runs/scine_irc_ens_${SLURM_ARRAY_JOB_ID}/${TAG}_${NPM}pm"
mkdir -p "$OUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID | cfg=$CFG | noise=${NPM}pm"
echo "Source: $SUMMARY"

python -u scripts/scine_irc_validate_v2.py \
    --summary-parquet "$SUMMARY" \
    --traj-dir "$SRC" \
    --noise-pm "$NPM" \
    --method-tag "$TAG" \
    --output-dir "$OUT_DIR" \
    --max-irc-steps 500 \
    --max-relax-steps 500 \
    --relax-fmax 0.001 \
    --rmsd-threshold 0.3 \
    --n-workers "$SLURM_CPUS_PER_TASK"
