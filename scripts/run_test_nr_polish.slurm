#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testnrpolish_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testnrpolish_%A_%a.err
#SBATCH --array=0-11

# NR-GAD polish: spectral-partitioned Newton-Raphson when n_neg=1, GAD otherwise.
# Question: does NR break the GAD plateau at fmax≈0.01?
#
# 2 configs (loose threshold = 0.01 / strict threshold = 1e-4)
# × 6 noise (10/30/50/100/150/200pm)
# = 12 cells. n=287, 5000 step budget.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CFGS=(nr_gad_polish_dt007_loose nr_gad_polish_dt007_strict)
NOISES=(0.01 0.03 0.05 0.10 0.15 0.20)

CFG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
N_IDX=$((SLURM_ARRAY_TASK_ID % 6))
METHOD=${CFGS[$CFG_IDX]}
NOISE=${NOISES[$N_IDX]}

OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_nrpolish/${METHOD}
mkdir -p "$OUTDIR"

echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD noise=$NOISE"

# Reduce n=80 (paper-acceptable subset) so the NR polish phase can fit in 18h.
# Each sample takes ~5-10 min in NR phase due to Hessian-eigendecomp cost.
python -u scripts/method_single.py \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples 80 \
    --n-steps 3000 \
    --split test \
    --output-dir "$OUTDIR"
