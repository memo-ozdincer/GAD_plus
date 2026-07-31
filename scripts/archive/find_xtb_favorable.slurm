#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/find_xtb_favorable_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/find_xtb_favorable_%j.err
#
# Compute xTB-GFN1 fmax at the HIP-TS geometry for all 287 test samples.
# Lowest-fmax samples are where xTB's nearest saddle is closest to HIP's;
# we'll run the xTB panel on those.

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u scripts/find_xtb_favorable_samples.py
