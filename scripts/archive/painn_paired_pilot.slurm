#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=painn-gad-sella
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/painn_gad_sella_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/painn_gad_sella_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
CANDIDATE_DIR="${1:?pass the PaiNN candidate directory}"
VALIDATION_SUMMARY="${2:?pass the completed PaiNN validation summary.json}"
OUT="/lustre07/scratch/memoozd/gadplus/runs/painn-gad-sella-${SLURM_JOB_ID}"
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/painn_paired_pilot.py \
  --candidate-dir "$CANDIDATE_DIR" \
  --validation-summary "$VALIDATION_SUMMARY" \
  --output-dir "$OUT" \
  --noise-pm 50 100 \
  --seeds 3 \
  --max-steps 300 \
  --gad-dt 0.005 \
  --sella-fmax 0.01
