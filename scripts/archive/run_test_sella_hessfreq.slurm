#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testhessfreq_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testhessfreq_%A_%a.err
#SBATCH --array=0-23

# Sella Hessian-injection frequency sweep on T1x test (n=287).
#
# We already have:
#   diag_every=1  (every step) -> runs/test_set/sella_carteck_libdef
#   no_hessian    (never)      -> runs/test_set/sella_carteck_nohess (timed out)
#
# This sweep fills in the middle: diag_every ∈ {3, 5, 10, 25}.
# Upstream Sella nsteps_per_diag default: 3 (every 3 steps).
#
# 4 frequencies × 6 noise = 24 tasks. Uses delta0=0.10, gamma=0.40.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

FREQS=(3 5 10 25)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

F_IDX=$((SLURM_ARRAY_TASK_ID / 6))
N_IDX=$((SLURM_ARRAY_TASK_ID % 6))
FREQ=${FREQS[$F_IDX]}
NOISE=${NOISES[$N_IDX]}
NPM=$(awk -v n=$NOISE 'BEGIN { printf "%d", n*1000 }')

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_hessfreq/sella_carteck_libdef_d${FREQ}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID freq=$FREQ noise=${NPM}pm out=$OUTDIR"

python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
    --n-samples 287 --max-steps 2000 --split test \
    --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
    --diag-every "$FREQ" \
    --config-tag "carteck_libdef_d${FREQ}" \
    --output-dir "$OUTDIR"
