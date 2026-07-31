#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/sellafill_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/sellafill_%A_%a.err
#SBATCH --array=0-2

# Fill the missing Sella cells noticed during 2026-05-11 report build.
#
# Cells:
#   0: sella libdef Hess-freq d=3 at 200 pm  (timed out / never ran first time)
#   1: Sella internal (lib default) at 150 pm
#   2: Sella internal (lib default) at 200 pm

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

T=$SLURM_ARRAY_TASK_ID

case $T in
  0)
    NOISE=0.20; NPM=200
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_hessfreq/sella_carteck_libdef_d3
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
        --n-samples 287 --max-steps 2000 --split test \
        --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
        --diag-every 3 --config-tag "carteck_libdef_d3" \
        --output-dir "$OUT"
    ;;
  1)
    NOISE=0.15; NPM=150
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
        --n-samples 287 --max-steps 2000 --split test \
        --config-tag "internal_default" --output-dir "$OUT"
    ;;
  2)
    NOISE=0.20; NPM=200
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
        --n-samples 287 --max-steps 2000 --split test \
        --config-tag "internal_default" --output-dir "$OUT"
    ;;
esac
