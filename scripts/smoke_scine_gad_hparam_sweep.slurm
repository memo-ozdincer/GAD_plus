#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_hp_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/smoke_scine_hp_%A_%a.err
#SBATCH --array=0-11
#
# SCINE/DFTB0 GAD hyperparameter smoke sweep.
# Same 20 T1x test samples at 10pm noise across all configs.
# Goal: find a setting that clears the fmax≈0.01 plateau.
#
# Array indices map to configs (see CONFIGS below).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# label dt n_steps extra_flags (single-space-separated for `read -r`)
declare -a CONFIGS=(
    "baseline_dt005_2k 0.005 2000"
    "dt010_2k 0.010 2000"
    "dt020_2k 0.020 2000"
    "dt005_5k 0.005 5000"
    "dt005_10k 0.005 10000"
    "adapt_dtmax020 0.010 2000 --use-adaptive-dt --dt-max 0.020"
    "adapt_dtmax050 0.010 2000 --use-adaptive-dt --dt-max 0.050"
    "precond_dt050 0.050 2000 --use-preconditioning --eig-floor 0.01"
    "precond_dt100 0.100 2000 --use-preconditioning --eig-floor 0.01"
    "precond_dt200 0.200 2000 --use-preconditioning --eig-floor 0.01"
    "precond_dt100_floor005 0.100 2000 --use-preconditioning --eig-floor 0.005"
    "precond_dt100_short 0.100 500 --use-preconditioning --eig-floor 0.01"
)

IDX=$SLURM_ARRAY_TASK_ID
read -r LABEL DT NSTEPS EXTRA <<< "${CONFIGS[$IDX]}"

OUT="/lustre07/scratch/memoozd/gadplus/runs/smoke_scine_hp_${SLURM_ARRAY_JOB_ID}/${LABEL}"
mkdir -p "$OUT"
echo "Task $IDX: $LABEL"
echo "  dt=$DT n_steps=$NSTEPS extra=$EXTRA"
echo "  Out: $OUT"

# 20 samples — the first 20 T1x test molecules
python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise 0.10 --n-samples 20 \
    --n-steps "$NSTEPS" --dt "$DT" \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT" \
    $EXTRA

echo "DONE: $LABEL"
