#!/bin/bash
# Matched g-xTB versus dxtb Hessian comparison on labelled T1x TSs.
# Reports raw Cartesian, projected Cartesian, and reduced MW/Eckart spectra.
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/gxtb_dxtb_hessian_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/gxtb_dxtb_hessian_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
OUT="/lustre07/scratch/memoozd/gadplus/runs/gxtb_dxtb_hessian_${SLURM_JOB_ID}"
cd "$PROJECT_DIR"
source /lustre07/scratch/memoozd/gadplus/envs/gadplus-project-20260716/bin/activate
PYTHON=/lustre07/scratch/memoozd/gadplus/envs/gadplus-project-20260716/bin/python
export PYTHONPATH="$PROJECT_DIR/src:/lustre06/project/6033559/memoozd/hip:/lustre06/project/6033559/memoozd/transition1x:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GADPLUS_GXTB_EXE="${GADPLUS_GXTB_EXE:-$PROJECT_DIR/g-xtb/xtb-6.7.1/bin/xtb}"

"$PYTHON" -u scripts/compare_gxtb_dxtb_hessians.py \
  --gxtb-executable "$GADPLUS_GXTB_EXE" \
  --reference hip \
  --device cuda \
  --sample-ids ${SAMPLE_IDS:-0 1 2} \
  --output-dir "$OUT"
