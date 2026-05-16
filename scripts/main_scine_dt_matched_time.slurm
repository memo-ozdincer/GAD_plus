#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_dt_matched_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_dt_matched_%A_%a.err
#SBATCH --array=0-11
#
# SCINE/DFTB0 GAD dt sweep at MATCHED physical time (dt * n_steps = 105).
# Decouples the "smaller dt = better stability" hypothesis from the
# step-budget confound that ruined the first dt sweep.
#
# Hard noise levels {50, 100, 150, 200}pm, fmax criterion only.
# dt    n_steps   t_physical
# 0.003 35000      105
# 0.005 21000      105
# 0.007 15000      105   (canonical headline — for control)
#
# 3 dt × 4 noise = 12 tasks.
# Array layout: tid = dt_idx*4 + noise_idx

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DTS=(0.003 0.005 0.007)
N_STEPS_ARR=(35000 21000 15000)
NOISES_PM=(50 100 150 200)
NOISES_A=(0.50 1.00 1.50 2.00)

DT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % 4))
DT=${DTS[$DT_IDX]}
N_STEPS=${N_STEPS_ARR[$DT_IDX]}
NOISE_PM=${NOISES_PM[$NOISE_IDX]}
NOISE_A=${NOISES_A[$NOISE_IDX]}

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_dtmatched_${SLURM_ARRAY_JOB_ID}/dt${DT}_n${N_STEPS}_noise${NOISE_PM}pm"
mkdir -p "$OUT"

echo "Task $SLURM_ARRAY_TASK_ID | dt=$DT | n_steps=$N_STEPS | noise=${NOISE_PM}pm"
echo "Out: $OUT"

python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise "$NOISE_A" --n-samples 287 \
    --n-steps "$N_STEPS" --dt "$DT" \
    --force-threshold 0.01 --force-criterion fmax \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
