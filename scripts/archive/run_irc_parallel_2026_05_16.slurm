#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircpar_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircpar_%A_%a.err
#SBATCH --array=0-11
#
# 3 IRC cells x 4 sample-id partitions each = 12 parallel MIG tasks.
# Wall budget 2h per task — each handles at most ~75 samples (most cells have
# <140 converged TS so partitions actually carry ~5-35 each).
#
# Cells:
#   cell A: Sella d=3 @ 200pm           (n=287 pooled, ~67 converged)
#   cell B: Sella internal d=1 @ 200pm  (n=287 pooled, ~40 converged)
#   cell C: Sella libdef midpoint @ 0pm (n=287, ~134 converged)
#
# Task mapping (T = SLURM_ARRAY_TASK_ID):
#   T 0-3:  cell A, partitions [0,72), [72,144), [144,216), [216,287)
#   T 4-7:  cell B, partitions [0,72), [72,144), [144,216), [216,287)
#   T 8-11: cell C, partitions [0,72), [72,144), [144,216), [216,287)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

T=$SLURM_ARRAY_TASK_ID
CELL=$((T / 4))
PART=$((T % 4))
RANGES=(0 72 144 216 287)
S=${RANGES[$PART]}
E=${RANGES[$((PART + 1))]}

case $CELL in
  0)
    SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_hessfreq/sella_carteck_libdef_d3
    SRC="sella_cartesian_eckart_fmax0p01_carteck_libdef_d3"
    NPM=200
    OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_libdef_d3_2026_05_16/p${PART}
    ;;
  1)
    SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
    SRC="sella_internal_fmax0p01_internal_default"
    NPM=200
    OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_internal_2026_05_16/200pm/p${PART}
    ;;
  2)
    SURVEY=/lustre07/scratch/memoozd/gadplus/runs/start_midpoint/sella_carteck_libdef
    SRC="sella_cartesian_eckart_fmax0p01_carteck_libdef_midpoint"
    NPM=0
    OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_midpoint_2026_05_16/p${PART}
    ;;
esac

mkdir -p "$OUT"
echo "T=$T  cell=$CELL  part=$PART  samples [$S,$E)  noise=${NPM}pm"

python -u scripts/irc_validate.py \
    --method sella_hip \
    --source-method "$SRC" \
    --coords-source summary --all-endpoints \
    --noise-pm "$NPM" --max-validate 300 \
    --sample-start "$S" --sample-end "$E" \
    --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY" \
    --n-dataset-samples 300 --split test \
    --output-dir "$OUT" \
    --no-write-viewer-bundles

echo "DONE"
