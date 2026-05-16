#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_hparam_smoke_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_hparam_smoke_%A_%a.err
#SBATCH --array=0-20
#
# SCINE/DFTB0 GAD hparam smoke at the difficult noise levels {50, 100, 200}pm.
# 7 configurations × 3 noise levels = 21 tasks.
# Each task: 287 samples, 15000 steps, dt=0.007 (canonical headline recipe).
#
# Array layout: tid = cfg_idx*3 + noise_idx
#   cfg_idx 0: no_eckart  (use_projection=False — control, n_neg still projected)
#   cfg_idx 1: ktrack16   (k_track=16 — more headroom for mode tracking)
#   cfg_idx 2: descent1   (descent_until_nneg=1 — pure descent to a 1-saddle first)
#   cfg_idx 3: descent3   (descent_until_nneg=3 — gentler descent gate)
#   cfg_idx 4: purify     (purify_hessian=True — TR mode numerical purification)
#   cfg_idx 5: mm_allneg  (multimode=all_neg — lift all negative modes simultaneously)
#   cfg_idx 6: precond05  (use_preconditioning, eig_floor=0.05)
#   noise_idx 0: 50pm, 1: 100pm, 2: 200pm

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES_PM=(50 100 200)
NOISES_A=(0.50 1.00 2.00)
CFG_NAMES=(no_eckart ktrack16 descent1 descent3 purify mm_allneg precond05)

CFG_IDX=$((SLURM_ARRAY_TASK_ID / 3))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 3))
CFG=${CFG_NAMES[$CFG_IDX]}
NOISE_PM=${NOISES_PM[$NOISE_IDX]}
NOISE_A=${NOISES_A[$NOISE_IDX]}

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_hparam_${SLURM_ARRAY_JOB_ID}/${CFG}_noise${NOISE_PM}pm"
mkdir -p "$OUT"

echo "Task $SLURM_ARRAY_TASK_ID | cfg=$CFG | noise=${NOISE_PM}pm"
echo "Out: $OUT"

# Default flags for canonical headline recipe
BASE_FLAGS=(--backend scine --method DFTB0 --noise "$NOISE_A"
            --n-samples 287 --n-steps 15000 --dt 0.007
            --force-threshold 0.01 --force-criterion fmax --split test
            --n-workers "$SLURM_CPUS_PER_TASK" --output-dir "$OUT")

case $CFG in
    no_eckart)
        python -u scripts/gad_hparam_smoke.py "${BASE_FLAGS[@]}" --no-projection ;;
    ktrack16)
        python -u scripts/gad_hparam_smoke.py "${BASE_FLAGS[@]}" --k-track 16 ;;
    descent1)
        python -u scripts/gad_hparam_smoke.py "${BASE_FLAGS[@]}" --descent-until-nneg 1 ;;
    descent3)
        python -u scripts/gad_hparam_smoke.py "${BASE_FLAGS[@]}" --descent-until-nneg 3 ;;
    purify)
        python -u scripts/gad_hparam_smoke.py "${BASE_FLAGS[@]}" --purify-hessian ;;
    mm_allneg)
        python -u scripts/gad_hparam_smoke.py "${BASE_FLAGS[@]}" --multimode all_neg ;;
    precond05)
        python -u scripts/gad_hparam_smoke.py "${BASE_FLAGS[@]}" --use-preconditioning --eig-floor 0.05 ;;
    *)
        echo "Unknown config: $CFG" >&2; exit 1 ;;
esac
