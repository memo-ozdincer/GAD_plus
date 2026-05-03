#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testhuge_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testhuge_%A_%a.err
#SBATCH --array=0-15

# Huge-noise probe — small n=50 sample to see what happens at 300/500/1000/2000pm.
# 4 noise levels × 4 methods (GAD dt=0.003, GAD dt=0.007, Sella libdef, Sella libdef-extended-steps)
# = 16 cells. Just a sniff test — not a full sweep. Looks at:
#   - Sella's failure mode at huge noise (does it remain "wrong saddle"?)
#   - GAD's failure mode at huge noise (does the plateau hold?)
#   - HIP Hessian quality at huge displacements
#   - Sella-with-5000-steps test (does early stopping limit Sella?)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NOISES=(0.30 0.50 1.00 2.00)
METHODS=(gad_dt003 gad_dt007 sella_libdef_2k sella_libdef_5k)

M_IDX=$((SLURM_ARRAY_TASK_ID / 4))
N_IDX=$((SLURM_ARRAY_TASK_ID % 4))
NOISE=${NOISES[$N_IDX]}
NPM=$(awk -v n=$NOISE 'BEGIN { printf "%d", n*1000 }')
METHOD=${METHODS[$M_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID noise=${NPM}pm method=$METHOD"

case $METHOD in
    gad_dt003)
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_huge/gad_dt003_fmax
        mkdir -p "$OUTDIR"
        python -u scripts/method_single.py --method gad_dt003_fmax \
            --noise "$NOISE" --n-samples 50 --n-steps 5000 --split test \
            --output-dir "$OUTDIR"
        ;;
    gad_dt007)
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_huge/gad_dt007_fmax
        mkdir -p "$OUTDIR"
        python -u scripts/method_single.py --method gad_dt007_fmax \
            --noise "$NOISE" --n-samples 50 --n-steps 5000 --split test \
            --output-dir "$OUTDIR"
        ;;
    sella_libdef_2k)
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_huge/sella_libdef_2k
        mkdir -p "$OUTDIR"
        python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
            --n-samples 50 --max-steps 2000 --split test \
            --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
            --config-tag carteck_libdef_2k \
            --output-dir "$OUTDIR"
        ;;
    sella_libdef_5k)
        # Tests "is Sella early-stopping?": same Sella but 5000-step budget
        OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/test_huge/sella_libdef_5k
        mkdir -p "$OUTDIR"
        python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
            --n-samples 50 --max-steps 5000 --split test \
            --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
            --config-tag carteck_libdef_5k \
            --output-dir "$OUTDIR"
        ;;
esac
