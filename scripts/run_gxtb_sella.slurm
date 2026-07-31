#!/bin/bash
# Narval template for g-xTB Cartesian+Eckart Sella.  For Trillium use
# scripts/trillium/run_gxtb_sella.slurm.
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gxtb_sella_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gxtb_sella_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="${PROJECT_DIR:-/lustre06/project/6033559/memoozd/GAD_plus}"
export GADPLUS_SCRATCH="${GADPLUS_SCRATCH:-/lustre07/scratch/memoozd/gadplus}"
cd "$PROJECT_DIR"
source scripts/lib/gxtb_runtime.sh
OUT="$GADPLUS_SCRATCH/runs/gxtb_sella_${SLURM_JOB_ID}"

"$PYTHON" -u scripts/sella_smoke.py \
  --backend gxtb --method gxTB --noise "${NOISE:-0.10}" \
  --sample-indices "${SAMPLE_IDS:-2}" \
  --max-steps "${N_STEPS:-100}" --fmax 0.01 \
  --apply-eckart --delta0 "${DELTA0:-0.1}" --gamma "${GAMMA:-0.4}" \
  --diag-every 1 --n-workers 1 --output-dir "$OUT"
