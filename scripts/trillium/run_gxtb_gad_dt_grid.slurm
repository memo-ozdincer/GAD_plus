#!/bin/bash
# Fixed-seed g-xTB GAD timestep calibration.  One array task = one timestep.
# Default grid is intentionally conservative; override DT_VALUES only after
# recording why a different range is needed.
#SBATCH --account=def-aspuru
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_gad_dt_%A_%a.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_gad_dt_%A_%a.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5

read -r -a DT_GRID <<< "${DT_VALUES:-0.001 0.003 0.005 0.007}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}
if (( TASK_ID >= ${#DT_GRID[@]} )); then
    echo "Task $TASK_ID has no corresponding timestep in DT_VALUES." >&2
    exit 2
fi

PROJECT_DIR="/scratch/memoozd/GAD/GAD_plus"
export GADPLUS_SCRATCH="/scratch/memoozd/gadplus"
export GADPLUS_T1X_H5="/scratch/memoozd/GAD/data/transition1x.h5"
export GADPLUS_GXTB_EXE="/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb"
export GADPLUS_GXTB_PARALLEL=96
cd "$PROJECT_DIR"
source scripts/lib/gxtb_runtime.sh

DT=${DT_GRID[$TASK_ID]}
OUT="$GADPLUS_SCRATCH/runs/gxtb_gad_dt_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_dt${DT}"
"$PYTHON" -u scripts/gad_smoke.py \
  --backend gxtb --method gxTB --noise "${NOISE:-0.10}" \
  --sample-indices "${SAMPLE_IDS:-2}" --seed "${SEED:-42}" \
  --n-steps "${N_STEPS:-20}" --dt "$DT" \
  --force-threshold 0.01 --force-criterion fmax \
  --n-workers 1 --output-dir "$OUT"
