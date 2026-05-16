#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/scine_dt_sweep_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/scine_dt_sweep_%A_%a.err
#SBATCH --array=0-59
#
# SCINE/DFTB0 GAD dt sweep across noise and force criterion. 15k steps, 287
# samples per task. The aim is to recover GAD's lead over Sella at high noise
# by reducing the step size (mode tracking breaks on shallow saddles at the
# current dt=0.007).
#
# Array layout: tid = dt_idx*12 + noise_idx*2 + crit_idx
#   5 dt values × 6 noise levels × 2 force criteria = 60 tasks
#   dt        ∈ {0.001, 0.002, 0.003, 0.005, 0.007}
#   noise_pm  ∈ {10, 30, 50, 100, 150, 200}
#   criterion ∈ {fmax, force_norm}

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DTS=(0.001 0.002 0.003 0.005 0.007)
NOISES_PM=(10 30 50 100 150 200)
NOISES_A=(0.10 0.30 0.50 1.00 1.50 2.00)
CRITERIA=(fmax force_norm)

DT_IDX=$((SLURM_ARRAY_TASK_ID / 12))
REM=$((SLURM_ARRAY_TASK_ID % 12))
NOISE_IDX=$((REM / 2))
CRIT_IDX=$((REM % 2))

DT=${DTS[$DT_IDX]}
NOISE_PM=${NOISES_PM[$NOISE_IDX]}
NOISE_A=${NOISES_A[$NOISE_IDX]}
CRIT=${CRITERIA[$CRIT_IDX]}

OUT="/lustre07/scratch/memoozd/gadplus/runs/main_scine_dtsweep_${SLURM_ARRAY_JOB_ID}/dt${DT}_noise${NOISE_PM}pm_${CRIT}"
mkdir -p "$OUT"

echo "Task $SLURM_ARRAY_TASK_ID | dt=$DT | noise=${NOISE_PM}pm | criterion=$CRIT"
echo "Out: $OUT"

python -u scripts/gad_smoke.py \
    --backend scine --method DFTB0 \
    --noise "$NOISE_A" --n-samples 287 \
    --n-steps 15000 --dt "$DT" \
    --force-threshold 0.01 --force-criterion "$CRIT" \
    --split test \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUT"
