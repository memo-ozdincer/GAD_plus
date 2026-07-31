#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_combos_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_combos_%A_%a.err
#SBATCH --array=0-9
#
# Composite hparam stacking tests on SCINE/DFTB0 GAD.
# Question: do the individually-winning interventions add?
# Each cell: 287 samples, 15000 steps, dt=0.007, fmax<0.01.
#
# Array tasks 0-9:
#  0 descent1 @ 10pm     (fill picture below 50pm)
#  1 descent1 @ 30pm     (fill picture below 50pm)
#  2 descent1+ktrack16 @ 30pm
#  3 descent1+ktrack16 @ 50pm
#  4 descent1+ktrack16 @ 100pm
#  5 descent1+precond05 @ 50pm
#  6 descent1+precond05 @ 100pm
#  7 descent1+precond05 @ 150pm
#  8 descent1+ktrack16+precond05 @ 100pm
#  9 descent1+ktrack16+precond05 @ 150pm

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
    0) NOISE_A=0.10; NOISE_PM=10;  CFG="descent1";             EXTRA="--descent-until-nneg 1" ;;
    1) NOISE_A=0.30; NOISE_PM=30;  CFG="descent1";             EXTRA="--descent-until-nneg 1" ;;
    2) NOISE_A=0.30; NOISE_PM=30;  CFG="descent1_ktrack16";    EXTRA="--descent-until-nneg 1 --k-track 16" ;;
    3) NOISE_A=0.50; NOISE_PM=50;  CFG="descent1_ktrack16";    EXTRA="--descent-until-nneg 1 --k-track 16" ;;
    4) NOISE_A=1.00; NOISE_PM=100; CFG="descent1_ktrack16";    EXTRA="--descent-until-nneg 1 --k-track 16" ;;
    5) NOISE_A=0.50; NOISE_PM=50;  CFG="descent1_precond05";   EXTRA="--descent-until-nneg 1 --use-preconditioning --eig-floor 0.05" ;;
    6) NOISE_A=1.00; NOISE_PM=100; CFG="descent1_precond05";   EXTRA="--descent-until-nneg 1 --use-preconditioning --eig-floor 0.05" ;;
    7) NOISE_A=1.50; NOISE_PM=150; CFG="descent1_precond05";   EXTRA="--descent-until-nneg 1 --use-preconditioning --eig-floor 0.05" ;;
    8) NOISE_A=1.00; NOISE_PM=100; CFG="descent1_kt16_pc05";   EXTRA="--descent-until-nneg 1 --k-track 16 --use-preconditioning --eig-floor 0.05" ;;
    9) NOISE_A=1.50; NOISE_PM=150; CFG="descent1_kt16_pc05";   EXTRA="--descent-until-nneg 1 --k-track 16 --use-preconditioning --eig-floor 0.05" ;;
esac

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_combos_${SLURM_ARRAY_JOB_ID}/${CFG}_noise${NOISE_PM}pm"
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
