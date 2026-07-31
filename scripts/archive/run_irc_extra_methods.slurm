#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircextra_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircextra_%A_%a.err
#SBATCH --array=0-22
#
# Parallel IRC sweep on auxiliary GAD methods that have summary+traj
# data but were never IRC-validated. 23 tasks total.
#
# Layout:
#   0-3   gad_dt002 @ 10/30/50/100 pm (round3)
#   4-9   blend_plain_k50 @ 10/30/50/100/150/200 pm (round3)
#   10-15 gad_small_dt @ 10/30/50/100/150/200 pm (precond_gad)
#   16-21 gad_projected @ 10/30/50/100/150/200 pm (method_cmp_300, Round 1 GAD)
#   22    gad_dt003 @ 200pm refill (force_norm) — replaces partial n=259 IRC

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

SCRATCH="/lustre07/scratch/memoozd"
TASK=$SLURM_ARRAY_TASK_ID

if [ $TASK -lt 4 ]; then
    NOISES=(10 30 50 100); NOISE=${NOISES[$TASK]}
    SOURCE="gad_dt002"; SURVEY_DIR="$SCRATCH/gadplus/runs/round3"
    OUT="$SCRATCH/gadplus/runs/irc_gad_dt002"
elif [ $TASK -lt 10 ]; then
    NOISES=(10 30 50 100 150 200); NOISE=${NOISES[$((TASK-4))]}
    SOURCE="blend_plain_k50"; SURVEY_DIR="$SCRATCH/gadplus/runs/round3"
    OUT="$SCRATCH/gadplus/runs/irc_blend_plain_k50"
elif [ $TASK -lt 16 ]; then
    NOISES=(10 30 50 100 150 200); NOISE=${NOISES[$((TASK-10))]}
    SOURCE="gad_small_dt"; SURVEY_DIR="$SCRATCH/gadplus/runs/precond_gad"
    OUT="$SCRATCH/gadplus/runs/irc_gad_small_dt"
elif [ $TASK -lt 22 ]; then
    NOISES=(10 30 50 100 150 200); NOISE=${NOISES[$((TASK-16))]}
    SOURCE="gad_projected"; SURVEY_DIR="$SCRATCH/gadplus/runs/method_cmp_300"
    OUT="$SCRATCH/gadplus/runs/irc_gad_projected_round1"
else
    NOISE=200
    SOURCE="gad_dt003"; SURVEY_DIR="$SCRATCH/gadplus/runs/round3"
    OUT="$SCRATCH/gadplus/runs/irc_gad_dt003_200_refill"
fi

mkdir -p "$OUT"
echo "=== task=$TASK | source=$SOURCE | noise=${NOISE}pm | survey=$SURVEY_DIR ==="

python -u scripts/irc_validate.py \
    --method sella_hip --source-method "$SOURCE" \
    --coords-source traj --all-endpoints \
    --noise-pm "$NOISE" --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY_DIR" --n-dataset-samples 300 \
    --output-dir "$OUT" --no-write-viewer-bundles
