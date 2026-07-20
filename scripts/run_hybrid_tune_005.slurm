#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hybtune005_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hybtune005_%A_%a.err
#SBATCH --array=0-19
#
# Hybrid tuning grid at noise = {0, 30} pm with fmax<0.05.
# method = hybrid_damped_eckart (canonical), dt = 5e-3, n_steps = 2000, n=287 test split.
# Grid:
#   switch-by-eig ∈ {true (swEIG), false (swFORCE)}
#   trust-radius  ∈ {0.005, 0.01, 0.02, 0.05, 0.10}
#   noise (pm)    ∈ {0, 30}
#   => 2 × 5 × 2 = 20 tasks

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SWITCH_OPTS=(true false)
TR_OPTS=(0.005 0.01 0.02 0.05 0.10)
NOISE_OPTS=(0.0 0.03)              # 0 pm, 30 pm in Å
NOISE_PM_OPTS=(0 30)

T=$SLURM_ARRAY_TASK_ID

# Decode index: outer = switch, then tr, then noise
SWITCH_I=$(( T / 10 ))            # 0..1
REM=$(( T % 10 ))
TR_I=$(( REM / 2 ))                # 0..4
NOISE_I=$(( REM % 2 ))             # 0..1

SWITCH=${SWITCH_OPTS[$SWITCH_I]}
TR=${TR_OPTS[$TR_I]}
NOISE=${NOISE_OPTS[$NOISE_I]}
NOISE_PM=${NOISE_PM_OPTS[$NOISE_I]}

TAG="hybrid_damped_eckart_sw${SWITCH}_tr${TR}_fmax0p05_${NOISE_PM}pm"
OUT=/lustre07/scratch/memoozd/gadplus/runs/hybrid_tune_fmax05/${TAG}
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs

echo "task=$T  method=hybrid_damped_eckart  switch_eig=$SWITCH  tr=$TR  noise=${NOISE_PM}pm  fmax<0.05"

python -u scripts/hybrid_gad_newton_runner.py \
    --method hybrid_damped_eckart \
    --switch-by-eig "$SWITCH" \
    --gad-dt 5e-3 \
    --trust-radius "$TR" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 2000 \
    --split test \
    --force-threshold 0.05 \
    --start-from ts_noised \
    --output-dir "$OUT"
