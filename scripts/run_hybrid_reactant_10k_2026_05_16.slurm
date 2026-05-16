#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hybrid_react10k_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hybrid_react10k_%A_%a.err
#SBATCH --array=0-1

# Follow-up to FINDING_HYBRID_REACTANT_2026_05_16.md:
# Test whether the hybrid's 0% conv from reactant @ 2000 steps is budget-limited.
# Predicted fix: 5000+ step budget should let GAD reach n_neg=1 manifold,
# then Newton can fire.
#
# Cells:
#   0  Hybrid damped Eckart eig-switch tr=0.05 from reactant, 10k steps
#   1  Hybrid undamped Eckart eig-switch tr=0.05 from reactant, 10k steps

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

T=$SLURM_ARRAY_TASK_ID

case $T in
  0)
    OUT=/lustre07/scratch/memoozd/gadplus/runs/start_reactant_hybrid_10k/damped_dt5e-3_tr0.05
    mkdir -p "$OUT"
    python -u scripts/hybrid_gad_newton_runner.py \
        --method hybrid_damped_eckart --switch-by-eig true \
        --gad-dt 5e-3 --trust-radius 0.05 \
        --noise 0.0 --n-samples 287 --n-steps 10000 \
        --split test --force-threshold 0.01 \
        --start-from reactant \
        --output-dir "$OUT"
    ;;
  1)
    OUT=/lustre07/scratch/memoozd/gadplus/runs/start_reactant_hybrid_10k/undamped_dt5e-3_tr0.05
    mkdir -p "$OUT"
    python -u scripts/hybrid_gad_newton_runner.py \
        --method hybrid_eckart --switch-by-eig true \
        --gad-dt 5e-3 --trust-radius 0.05 \
        --noise 0.0 --n-samples 287 --n-steps 10000 \
        --split test --force-threshold 0.01 \
        --start-from reactant \
        --output-dir "$OUT"
    ;;
esac

echo "TASK $T DONE"
