#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/buildpdf_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/buildpdf_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u scripts/build_pdf_2026_05_11.py
