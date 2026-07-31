#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irc150_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irc150_%A_%a.err
#SBATCH --array=0-3
#
# Sella internal d=1 @ 150pm IRC validation, 4-way partitioned.
# Submit AFTER job 61166201 (Sella internal 150pm refill) lands AND after
# build_pooled_summaries_2026_05_16.py (or manual concat) has produced a
# canonical summary parquet at the path below.
#
# Partitions: [0,72), [72,144), [144,216), [216,287)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

T=$SLURM_ARRAY_TASK_ID
RANGES=(0 72 144 216 287)
S=${RANGES[$T]}
E=${RANGES[$((T + 1))]}

SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
SRC="sella_internal_fmax0p01_internal_default"

# Stage pooled summary if it exists (built by build_pooled_summaries)
if [ -f "$SURVEY/pooled_summary_150pm.parquet" ] \
   && [ ! -f "$SURVEY/summary_${SRC}_150pm.parquet" ]; then
    cp "$SURVEY/pooled_summary_150pm.parquet" "$SURVEY/summary_${SRC}_150pm.parquet"
fi

if [ ! -f "$SURVEY/summary_${SRC}_150pm.parquet" ]; then
    echo "ERROR: $SURVEY/summary_${SRC}_150pm.parquet not found. Run build_pooled_summaries first."
    exit 1
fi

OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_internal_2026_05_16/150pm/p${T}
mkdir -p "$OUT"

echo "T=$T samples [$S,$E) IRC validation of Sella internal d=1 @ 150pm"

python -u scripts/irc_validate.py \
    --method sella_hip \
    --source-method "$SRC" \
    --coords-source summary --all-endpoints \
    --noise-pm 150 --max-validate 300 \
    --sample-start "$S" --sample-end "$E" \
    --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY" \
    --n-dataset-samples 300 --split test \
    --output-dir "$OUT" \
    --no-write-viewer-bundles

echo "DONE"
