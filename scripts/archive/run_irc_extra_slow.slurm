#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircextrasl_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircextrasl_%A_%a.err
#SBATCH --array=0-5
#
# Retry of slow tasks from 59967832 (Round 1 GAD TSs IRC ~30x slower per sample).
# 16h budget, 100-sample cap to ensure we get useful data even if not full 300.
#
# Layout:
#   0  gad_dt002 @ 10pm  (round3)
#   1  gad_dt002 @ 30pm  (round3)
#   2  gad_projected @ 30pm  (method_cmp_300)
#   3  gad_projected @ 50pm  (method_cmp_300)
#   4  gad_projected @ 100pm (method_cmp_300)
#   5  gad_projected @ 150pm (method_cmp_300)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

SCRATCH="/lustre07/scratch/memoozd"
TASK=$SLURM_ARRAY_TASK_ID

case $TASK in
    0) NOISE=10;  SOURCE="gad_dt002";     SURVEY="$SCRATCH/gadplus/runs/round3"; OUT="$SCRATCH/gadplus/runs/irc_gad_dt002" ;;
    1) NOISE=30;  SOURCE="gad_dt002";     SURVEY="$SCRATCH/gadplus/runs/round3"; OUT="$SCRATCH/gadplus/runs/irc_gad_dt002" ;;
    2) NOISE=30;  SOURCE="gad_projected"; SURVEY="$SCRATCH/gadplus/runs/method_cmp_300"; OUT="$SCRATCH/gadplus/runs/irc_gad_projected_round1" ;;
    3) NOISE=50;  SOURCE="gad_projected"; SURVEY="$SCRATCH/gadplus/runs/method_cmp_300"; OUT="$SCRATCH/gadplus/runs/irc_gad_projected_round1" ;;
    4) NOISE=100; SOURCE="gad_projected"; SURVEY="$SCRATCH/gadplus/runs/method_cmp_300"; OUT="$SCRATCH/gadplus/runs/irc_gad_projected_round1" ;;
    5) NOISE=150; SOURCE="gad_projected"; SURVEY="$SCRATCH/gadplus/runs/method_cmp_300"; OUT="$SCRATCH/gadplus/runs/irc_gad_projected_round1" ;;
esac

mkdir -p "$OUT"
echo "=== task=$TASK | source=$SOURCE | noise=${NOISE}pm | budget=16h cap=100 ==="

python -u scripts/irc_validate.py \
    --method sella_hip --source-method "$SOURCE" \
    --coords-source traj --all-endpoints \
    --noise-pm "$NOISE" --max-validate 100 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY" --n-dataset-samples 300 \
    --output-dir "$OUT" --no-write-viewer-bundles
