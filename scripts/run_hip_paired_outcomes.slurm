#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --job-name=hip-paired-outcomes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hip_paired_outcomes_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hip_paired_outcomes_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11
PROJECT_DIR=/lustre06/project/6033559/memoozd/GAD_plus
OUT=/lustre07/scratch/memoozd/gadplus/runs/hip-paired-outcomes-${SLURM_JOB_ID}
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source .venv/bin/activate
python -u scripts/analyze_hip_paired_outcomes.py \
  --gad-dir /lustre07/scratch/memoozd/gadplus/runs/test_dtgrid/gad_dt007_fmax \
  --sella-dir /lustre07/scratch/memoozd/gadplus/runs/test_sella_trajlog/carteck_libdef \
  --noise-pm 150 \
  --output-dir "$OUT"
