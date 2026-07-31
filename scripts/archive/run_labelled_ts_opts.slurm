#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/labelled_ts_opt_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/labelled_ts_opt_%A_%a.err
#SBATCH --array=0-3
#
# Canonical 5-method comparison at the labelled TS (test split, noise=0):
#   [0] gad_dt003_no_eckart      method_single.py
#   [1] sella_cart_eckart        sella_baseline.py --cartesian --apply-eckart
#   [2] sella_cart_no_eckart     sella_baseline.py --cartesian
#   [3] sella_internal           sella_baseline.py --internal (Sella default)
#
# Job [-1] (gad_dt003_fmax) was already run in run_labelled_ts_0pm.slurm.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=/lustre07/scratch/memoozd/gadplus/runs/labelled_ts_0pm
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs

case "$SLURM_ARRAY_TASK_ID" in
  0)
    echo "=== gad_dt003_no_eckart from labelled TS ==="
    python -u scripts/method_single.py \
        --method gad_dt003_no_eckart \
        --start-from ts_noised \
        --noise 0.0 \
        --n-samples 287 --n-steps 2000 \
        --force-criterion fmax --force-threshold 0.01 \
        --split test \
        --output-dir "$OUT"
    ;;
  1)
    echo "=== Sella cart+Eckart from labelled TS ==="
    python -u scripts/sella_baseline.py \
        --start-from ts_noised \
        --noise 0.0 \
        --cartesian --apply-eckart \
        --fmax 0.01 \
        --n-samples 287 --max-steps 2000 \
        --split test \
        --config-tag labelled_ts \
        --output-dir "$OUT"
    ;;
  2)
    echo "=== Sella cart no-Eckart from labelled TS ==="
    python -u scripts/sella_baseline.py \
        --start-from ts_noised \
        --noise 0.0 \
        --cartesian \
        --fmax 0.01 \
        --n-samples 287 --max-steps 2000 \
        --split test \
        --config-tag labelled_ts \
        --output-dir "$OUT"
    ;;
  3)
    echo "=== Sella internal (default) from labelled TS ==="
    python -u scripts/sella_baseline.py \
        --start-from ts_noised \
        --noise 0.0 \
        --internal \
        --fmax 0.01 \
        --n-samples 287 --max-steps 2000 \
        --split test \
        --config-tag labelled_ts \
        --output-dir "$OUT"
    ;;
esac
