#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/ircsellad3_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/ircsellad3_%A_%a.err
#SBATCH --array=0-5

# IRC validation for Sella libdef Hess-freq d=3 across all 6 noise levels.
# Task 5 (200pm) depends on the upstream 60769799_0 finishing first;
# we'll just rerun the array if cell 5's survey is missing.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

NOISES_PM=(10 30 50 100 150 200)
NPM=${NOISES_PM[$SLURM_ARRAY_TASK_ID]}

SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_hessfreq/sella_carteck_libdef_d3
OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_libdef_d3
mkdir -p "$OUT"

if [ ! -f "$SURVEY/summary_sella_cartesian_eckart_fmax0p01_carteck_libdef_d3_${NPM}pm.parquet" ]; then
    echo "Survey for ${NPM}pm not found yet; skipping (likely the 200 pm cell not done)"
    exit 0
fi

python -u scripts/irc_validate.py \
    --method sella_hip \
    --source-method "sella_cartesian_eckart_fmax0p01_carteck_libdef_d3" \
    --coords-source summary --all-endpoints \
    --noise-pm "$NPM" --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY" \
    --n-dataset-samples 300 --split test \
    --output-dir "$OUT" \
    --no-write-viewer-bundles
