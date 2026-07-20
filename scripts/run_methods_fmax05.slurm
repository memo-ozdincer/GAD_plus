#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/meth005_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/meth005_%A_%a.err
#SBATCH --array=0-9
#
# All 5 canonical methods at noise = {0, 30} pm with fmax<0.05.
# n=287 test split, 2000-step budget, fmax convergence criterion = 0.05.
#
# Tasks:
#   0,1: gad_dt003_fmax        @ 0pm, 30pm
#   2,3: gad_dt003_no_eckart   @ 0pm, 30pm
#   4,5: sella_cart_eckart     @ 0pm, 30pm
#   6,7: sella_cart_no_eckart  @ 0pm, 30pm
#   8,9: sella_internal        @ 0pm, 30pm

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NOISE_OPTS=(0.0 0.03)
NOISE_PM_OPTS=(0 30)

T=$SLURM_ARRAY_TASK_ID
METHOD_I=$(( T / 2 ))
NOISE_I=$(( T % 2 ))
NOISE=${NOISE_OPTS[$NOISE_I]}
NOISE_PM=${NOISE_PM_OPTS[$NOISE_I]}

OUT=/lustre07/scratch/memoozd/gadplus/runs/methods_fmax05
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs

echo "task=$T  method_i=$METHOD_I  noise=${NOISE_PM}pm  fmax<0.05"

case "$METHOD_I" in
  0)
    python -u scripts/method_single.py \
        --method gad_dt003_fmax \
        --start-from ts_noised --noise "$NOISE" \
        --n-samples 287 --n-steps 2000 \
        --force-criterion fmax --force-threshold 0.05 \
        --split test \
        --output-dir "$OUT"
    ;;
  1)
    python -u scripts/method_single.py \
        --method gad_dt003_no_eckart \
        --start-from ts_noised --noise "$NOISE" \
        --n-samples 287 --n-steps 2000 \
        --force-criterion fmax --force-threshold 0.05 \
        --split test \
        --output-dir "$OUT"
    ;;
  2)
    python -u scripts/sella_baseline.py \
        --start-from ts_noised --noise "$NOISE" \
        --cartesian --apply-eckart \
        --fmax 0.05 \
        --n-samples 287 --max-steps 2000 \
        --split test \
        --config-tag fmax05 \
        --output-dir "$OUT"
    ;;
  3)
    python -u scripts/sella_baseline.py \
        --start-from ts_noised --noise "$NOISE" \
        --cartesian \
        --fmax 0.05 \
        --n-samples 287 --max-steps 2000 \
        --split test \
        --config-tag fmax05 \
        --output-dir "$OUT"
    ;;
  4)
    python -u scripts/sella_baseline.py \
        --start-from ts_noised --noise "$NOISE" \
        --internal \
        --fmax 0.05 \
        --n-samples 287 --max-steps 2000 \
        --split test \
        --config-tag fmax05 \
        --output-dir "$OUT"
    ;;
esac
