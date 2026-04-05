#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/targeted_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/targeted_%A_%a.err
#SBATCH --array=0-41
#
# Targeted tests: damped NR-GAD, geodesic midpoint, high step count, randomized samples.
#
# Array layout (42 jobs):
#   0-17:  Damped NR-GAD (3 damping levels × 6 noise levels)
#   18-23: gad_small_dt at high noise with 2000 steps (6 noise levels)
#   24-29: gad_small_dt randomized samples (offset=300, 6 noise levels)
#   30-35: gad_projected randomized samples (offset=300, 6 noise levels)
#   36-41: gad_small_dt with 3000 steps at high noise (6 noise levels)
#
# Submit:  sbatch scripts/run_targeted_tests.slurm

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SCRATCH="/lustre07/scratch/memoozd"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/gadplus/tmp"
mkdir -p "$TMPDIR" "$SCRATCH/gadplus/logs" "$SCRATCH/gadplus/runs/targeted"

NOISE_LEVELS=(0.01 0.03 0.05 0.1 0.15 0.2)
T=$SLURM_ARRAY_TASK_ID

if [ $T -lt 18 ]; then
    # 0-17: Damped NR-GAD (3 variants × 6 noise)
    METHODS=(nr_gad_damped_01 nr_gad_damped_02 nr_gad_damped_03)
    METHOD_IDX=$((T / 6))
    NOISE_IDX=$((T % 6))
    METHOD=${METHODS[$METHOD_IDX]}
    NOISE=${NOISE_LEVELS[$NOISE_IDX]}
    N_STEPS=1000
    N_SAMPLES=300
    OFFSET=0

elif [ $T -lt 24 ]; then
    # 18-23: gad_small_dt 2000 steps
    NOISE_IDX=$((T - 18))
    METHOD="gad_small_dt"
    NOISE=${NOISE_LEVELS[$NOISE_IDX]}
    N_STEPS=2000
    N_SAMPLES=300
    OFFSET=0

elif [ $T -lt 30 ]; then
    # 24-29: gad_small_dt randomized (offset=300)
    NOISE_IDX=$((T - 24))
    METHOD="gad_small_dt"
    NOISE=${NOISE_LEVELS[$NOISE_IDX]}
    N_STEPS=1000
    N_SAMPLES=300
    OFFSET=300

elif [ $T -lt 36 ]; then
    # 30-35: gad_projected randomized (offset=300)
    NOISE_IDX=$((T - 30))
    METHOD="gad_projected"
    NOISE=${NOISE_LEVELS[$NOISE_IDX]}
    N_STEPS=1000
    N_SAMPLES=300
    OFFSET=300

else
    # 36-41: gad_small_dt 3000 steps
    NOISE_IDX=$((T - 36))
    METHOD="gad_small_dt"
    NOISE=${NOISE_LEVELS[$NOISE_IDX]}
    N_STEPS=3000
    N_SAMPLES=300
    OFFSET=0
fi

echo "Job $SLURM_JOB_ID, task $T: method=$METHOD noise=$NOISE steps=$N_STEPS samples=$N_SAMPLES offset=$OFFSET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u "$PROJECT_DIR/scripts/method_single.py" \
    --method "$METHOD" \
    --noise "$NOISE" \
    --n-samples "$N_SAMPLES" \
    --n-steps "$N_STEPS" \
    --random-offset "$OFFSET" \
    --split train \
    --output-dir "$SCRATCH/gadplus/runs/targeted"

echo "Done."
