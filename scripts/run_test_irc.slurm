#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/testirc_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/testirc_%A_%a.err
#SBATCH --array=0-35

# IRC validation (sella_hip, all endpoints) on every test-set TS pool.
# 6 method-groups × 6 noise = 36 tasks. Explicit case to avoid bash
# array+colon-split parsing bugs.
#
# Group decoding: g = task / 6, n_idx = task % 6
#   g=0: gad_dt003_fmax (test_dtgrid 5k)
#   g=1: gad_dt005_fmax (test_dtgrid 5k)
#   g=2: gad_dt007_fmax (test_dtgrid 5k)
#   g=3: sella_carteck_default
#   g=4: sella_carteck_libdef
#   g=5: sella_internal_default

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

TASK=$SLURM_ARRAY_TASK_ID
G=$(( TASK / 6 ))
N_IDX=$(( TASK % 6 ))

case $N_IDX in
    0) NOISE=10 ;;
    1) NOISE=30 ;;
    2) NOISE=50 ;;
    3) NOISE=100 ;;
    4) NOISE=150 ;;
    5) NOISE=200 ;;
esac

case $G in
    0)
        SOURCE=gad_dt003_fmax
        SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_dtgrid/gad_dt003_fmax
        OUT=/lustre07/scratch/memoozd/gadplus/runs/test_irc/gad_dt003_fmax
        COORDS=traj
        ;;
    1)
        SOURCE=gad_dt005_fmax
        SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_dtgrid/gad_dt005_fmax
        OUT=/lustre07/scratch/memoozd/gadplus/runs/test_irc/gad_dt005_fmax
        COORDS=traj
        ;;
    2)
        SOURCE=gad_dt007_fmax
        SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_dtgrid/gad_dt007_fmax
        OUT=/lustre07/scratch/memoozd/gadplus/runs/test_irc/gad_dt007_fmax
        COORDS=traj
        ;;
    3)
        SOURCE=sella_cartesian_eckart_fmax0p01_carteck_default
        SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_carteck_default
        OUT=/lustre07/scratch/memoozd/gadplus/runs/test_irc/sella_carteck_default
        COORDS=summary
        ;;
    4)
        SOURCE=sella_cartesian_eckart_fmax0p01_carteck_libdef
        SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_carteck_libdef
        OUT=/lustre07/scratch/memoozd/gadplus/runs/test_irc/sella_carteck_libdef
        COORDS=summary
        ;;
    5)
        SOURCE=sella_internal_fmax0p01_internal_default
        SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
        OUT=/lustre07/scratch/memoozd/gadplus/runs/test_irc/sella_internal_default
        COORDS=summary
        ;;
esac

mkdir -p "$OUT"
echo "task=$TASK g=$G source=$SOURCE survey=$SURVEY coords=$COORDS noise=${NOISE}pm"

python -u scripts/irc_validate.py \
    --method sella_hip --source-method "$SOURCE" \
    --coords-source "$COORDS" --all-endpoints --split test \
    --noise-pm "$NOISE" --max-validate 287 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY" --n-dataset-samples 287 \
    --output-dir "$OUT" --no-write-viewer-bundles
