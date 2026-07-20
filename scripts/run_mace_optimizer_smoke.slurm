#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/mace_optimizer_%A.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/mace_optimizer_%A.err

# Small, serial-on-GPU MACE-OFF23 GAD/Sella comparison.  One worker is
# intentional: each worker owns a full MACE model and a GPU context.
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.6

PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

: "${SEARCH:?SEARCH must be gad or sella}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

MODEL="${MODEL:-small}"
SAMPLE_IDS="${SAMPLE_IDS:-2,4,5,7,8}"
NOISE="${NOISE:-0.01}"
mkdir -p "$OUTPUT_DIR"

if [[ "$SEARCH" == "gad" ]]; then
  python -u scripts/gad_smoke.py \
    --backend mace --method "$MODEL" \
    --sample-indices "$SAMPLE_IDS" --noise "$NOISE" \
    --n-steps "${N_STEPS:-1000}" --dt "${DT:-0.003}" \
    --force-threshold 0.01 --force-criterion fmax \
    --n-workers 1 --output-dir "$OUTPUT_DIR"
elif [[ "$SEARCH" == "sella" ]]; then
  python -u scripts/sella_smoke.py \
    --backend mace --method "$MODEL" \
    --sample-indices "$SAMPLE_IDS" --noise "$NOISE" \
    --max-steps "${N_STEPS:-1000}" --fmax 0.01 \
    --apply-eckart --delta0 0.1 --gamma 0.4 --diag-every 1 \
    --n-workers 1 --output-dir "$OUTPUT_DIR"
else
  echo "Unknown SEARCH=$SEARCH; expected gad or sella" >&2
  exit 2
fi
