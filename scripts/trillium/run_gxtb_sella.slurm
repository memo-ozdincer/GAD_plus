#!/bin/bash
# Trillium CPU template: Cartesian+Eckart Sella, full g-xTB Hessian every step.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_sella_%A_%a.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_sella_%A_%a.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

PROJECT_DIR="/scratch/memoozd/GAD/GAD_plus"
export GADPLUS_SCRATCH="/scratch/memoozd/gadplus"
export GADPLUS_T1X_H5="/scratch/memoozd/GAD/data/transition1x.h5"
export GADPLUS_GXTB_EXE="/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb"
export GADPLUS_GXTB_PARALLEL=96
cd "$PROJECT_DIR"
source scripts/lib/gxtb_runtime.sh

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  IFS=, read -r -a SAMPLE_ID_ARRAY <<< "${SAMPLE_IDS:?Set SAMPLE_IDS for an array job}"
  SAMPLE_IDS="${SAMPLE_ID_ARRAY[$SLURM_ARRAY_TASK_ID]:?Array task has no matching sample ID}"
fi

OUT="$GADPLUS_SCRATCH/runs/gxtb_sella_${SLURM_JOB_ID}"
"$PYTHON" -u scripts/sella_smoke.py \
  --backend gxtb --method gxTB --noise "${NOISE:-0.10}" \
  --start-geometry "${START_GEOMETRY:-noised_ts}" \
  --sample-indices "${SAMPLE_IDS:-2}" \
  --max-steps "${N_STEPS:-100}" --fmax 0.01 \
  --apply-eckart --delta0 "${DELTA0:-0.1}" --gamma "${GAMMA:-0.4}" \
  --diag-every 1 --n-workers 1 --output-dir "$OUT"
