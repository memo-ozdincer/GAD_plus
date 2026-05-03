#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testmid_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testmid_%A_%a.err
#SBATCH --array=0-3

# Single-ended TS search from R+P linear midpoint, no TS noise (0pm).
# Tests the paper's claim that GAD can climb from a non-TS starting point.
#
# 4 methods: GAD dt=0.003, GAD dt=0.007, Sella libdef, Sella internal default.
# All at noise=0pm starting from midpoint(reactant, product).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

T=$SLURM_ARRAY_TASK_ID
case $T in
    0)
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_midpoint/gad_dt003_fmax
        mkdir -p "$OUTDIR"
        python -u scripts/method_single.py --method gad_dt003_fmax \
            --noise 0.0 --n-samples 287 --n-steps 10000 --split test \
            --start-from midpoint --output-dir "$OUTDIR"
        ;;
    1)
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_midpoint/gad_dt007_fmax
        mkdir -p "$OUTDIR"
        python -u scripts/method_single.py --method gad_dt007_fmax \
            --noise 0.0 --n-samples 287 --n-steps 10000 --split test \
            --start-from midpoint --output-dir "$OUTDIR"
        ;;
    2)
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_midpoint/sella_carteck_libdef
        mkdir -p "$OUTDIR"
        python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.0 \
            --n-samples 287 --max-steps 10000 --split test \
            --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
            --config-tag carteck_libdef --start-from midpoint \
            --output-dir "$OUTDIR"
        ;;
    3)
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_midpoint/sella_internal_default
        mkdir -p "$OUTDIR"
        python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.0 \
            --n-samples 287 --max-steps 10000 --split test \
            --delta0 0.048 --gamma 0.0 \
            --config-tag internal_default --start-from midpoint \
            --output-dir "$OUTDIR"
        ;;
esac
