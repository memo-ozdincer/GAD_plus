#!/bin/bash
# Narval template for g-xTB projected GAD.  For Trillium use
# scripts/trillium/run_gxtb_gad.slurm.
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gxtb_gad_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gxtb_gad_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="${PROJECT_DIR:-/lustre06/project/6033559/memoozd/GAD_plus}"
export GADPLUS_SCRATCH="${GADPLUS_SCRATCH:-/lustre07/scratch/memoozd/gadplus}"
cd "$PROJECT_DIR"
source scripts/lib/gxtb_runtime.sh
OUT="$GADPLUS_SCRATCH/runs/gxtb_gad_${SLURM_JOB_ID}"

"$PYTHON" -u scripts/gad_smoke.py \
  --backend gxtb --method gxTB --noise "${NOISE:-0.10}" \
  --sample-indices "${SAMPLE_IDS:-2}" \
  --n-steps "${N_STEPS:-100}" --dt "${DT:-0.007}" \
  --force-threshold 0.01 --force-criterion fmax \
  --n-workers 1 --output-dir "$OUT"
