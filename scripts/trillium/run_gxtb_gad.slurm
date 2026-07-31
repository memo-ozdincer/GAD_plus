#!/bin/bash
# Trillium CPU template: projected GAD with g-xTB full Hessians.
# Start with one trajectory and 32 xtb parallel workers; tune DT only after
# recording per-Hessian timing on representative labelled T1x samples.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_gad_%A_%a.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_gad_%A_%a.err

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

OUT="$GADPLUS_SCRATCH/runs/gxtb_gad_${SLURM_JOB_ID}"
"$PYTHON" -u scripts/gad_smoke.py \
  --backend gxtb --method gxTB --noise "${NOISE:-0.10}" \
  --start-geometry "${START_GEOMETRY:-noised_ts}" \
  --sample-indices "${SAMPLE_IDS:-2}" \
  --n-steps "${N_STEPS:-100}" --dt "${DT:-0.003}" \
  --force-threshold 0.01 --force-criterion fmax \
  --n-workers 1 --output-dir "$OUT"
