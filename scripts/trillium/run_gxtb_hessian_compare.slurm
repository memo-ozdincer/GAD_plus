#!/bin/bash
# Time and compare g-xTB full Hessians against HIP or dxTB on labelled T1x TSs.
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/gxtb_hessian_compare_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/gxtb_hessian_compare_%j.err

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

OUT="$GADPLUS_SCRATCH/runs/gxtb_hessian_compare_${SLURM_JOB_ID}"
"$PYTHON" -u scripts/compare_gxtb_dxtb_hessians.py \
  --gxtb-executable "$GADPLUS_GXTB_EXE" \
  --reference "${REFERENCE:-hip}" --device "${DEVICE:-cpu}" \
  --checkpoint "${HIP_CHECKPOINT:-/scratch/memoozd/GAD/models/hip_v2.ckpt}" \
  --sample-ids ${SAMPLE_IDS:-0 1 2} --output-dir "$OUT"
