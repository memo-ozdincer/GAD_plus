#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testreact_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testreact_%A_%a.err
#SBATCH --array=0-3

# Single-ended TS search: start from REACTANT (not noised TS), run each
# method to (hopefully) climb to the saddle. Tests the paper's claim
# that GAD can be initialized far from the TS, including at minima.
#
# 4 methods × 1 setting = 4 array tasks. Step budget: 10000 (this is the
# hardest starting condition, may need a lot).

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

T=$SLURM_ARRAY_TASK_ID
case $T in
    0)
        # GAD canonical (dt=0.003) from reactant
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_reactant/gad_dt003_fmax
        mkdir -p "$OUTDIR"
        python -u scripts/method_single.py --method gad_dt003_fmax \
            --noise 0.0 --n-samples 287 --n-steps 10000 --split test \
            --start-from reactant --output-dir "$OUTDIR"
        ;;
    1)
        # GAD dt=0.005 from reactant (the sweet-spot)
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_reactant/gad_dt005_fmax
        mkdir -p "$OUTDIR"
        python -u scripts/method_single.py --method gad_dt005_fmax \
            --noise 0.0 --n-samples 287 --n-steps 10000 --split test \
            --start-from reactant --output-dir "$OUTDIR"
        ;;
    2)
        # Sella cart+Eckart libdef (best Sella) from reactant
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_reactant/sella_carteck_libdef
        mkdir -p "$OUTDIR"
        python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.0 \
            --n-samples 287 --max-steps 10000 --split test \
            --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
            --config-tag carteck_libdef --start-from reactant \
            --output-dir "$OUTDIR"
        ;;
    3)
        # Sella internal default from reactant
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_reactant/sella_internal_default
        mkdir -p "$OUTDIR"
        python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.0 \
            --n-samples 287 --max-steps 10000 --split test \
            --delta0 0.048 --gamma 0.0 \
            --config-tag internal_default --start-from reactant \
            --output-dir "$OUTDIR"
        ;;
esac
