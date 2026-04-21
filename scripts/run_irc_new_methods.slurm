#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircnew_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircnew_%A_%a.err
#SBATCH --array=0-22
#
# sella_hip IRC (all endpoints) on the four new TS sets.
# Array layout (23 tasks):
#   0-5   GAD no-Eckart @ 10/30/50/100/150/200 pm  (coords from traj)
#   6-11  Sella cart+Eckart 2000 @ 10/30/50/100/150/200 pm (coords from summary)
#   12-17 Sella cart no-Eckart 2000 @ 10/30/50/100/150/200 pm
#   18-22 Sella internal 2000 @ 10/30/50/100/150 pm (200 still running, skip)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

NOISES=(10 30 50 100 150 200)
SCRATCH="/lustre07/scratch/memoozd"
TASK=$SLURM_ARRAY_TASK_ID

if [ $TASK -lt 6 ]; then
    NOISE_IDX=$TASK
    SOURCE_METHOD="gad_dt003_no_eckart"
    SURVEY_DIR="$SCRATCH/gadplus/runs/gad_no_eckart"
    COORDS_SOURCE="traj"
    OUT_DIR="$SCRATCH/gadplus/runs/irc_gad_no_eckart"
elif [ $TASK -lt 12 ]; then
    NOISE_IDX=$((TASK - 6))
    SOURCE_METHOD="sella_cartesian_eckart_fmax0p01"
    SURVEY_DIR="$SCRATCH/gadplus/runs/sella_2000"
    COORDS_SOURCE="summary"
    OUT_DIR="$SCRATCH/gadplus/runs/irc_sella_carte_2000"
elif [ $TASK -lt 18 ]; then
    NOISE_IDX=$((TASK - 12))
    SOURCE_METHOD="sella_cartesian_fmax0p01"
    SURVEY_DIR="$SCRATCH/gadplus/runs/sella_2000"
    COORDS_SOURCE="summary"
    OUT_DIR="$SCRATCH/gadplus/runs/irc_sella_cart_2000"
else
    NOISE_IDX=$((TASK - 18))
    SOURCE_METHOD="sella_internal_fmax0p01"
    SURVEY_DIR="$SCRATCH/gadplus/runs/sella_2000"
    COORDS_SOURCE="summary"
    OUT_DIR="$SCRATCH/gadplus/runs/irc_sella_int_2000"
fi

NOISE=${NOISES[$NOISE_IDX]}
mkdir -p "$OUT_DIR"

echo "=== IRC | method=sella_hip | source=$SOURCE_METHOD | noise=${NOISE}pm | coords=$COORDS_SOURCE ==="

python -u scripts/irc_validate.py \
    --method sella_hip \
    --source-method "$SOURCE_METHOD" \
    --coords-source "$COORDS_SOURCE" \
    --all-endpoints \
    --noise-pm "$NOISE" \
    --max-validate 300 \
    --irc-steps 500 \
    --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY_DIR" \
    --n-dataset-samples 300 \
    --output-dir "$OUT_DIR" \
    --no-write-viewer-bundles

echo "Done."
