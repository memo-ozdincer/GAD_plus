#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/horm_optimizer_%A.out

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

cd /lustre06/project/6033559/memoozd/GAD_plus
source /lustre07/scratch/memoozd/venvs/horm/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="/lustre06/project/6033559/memoozd/GAD_plus/src:/lustre06/project/6033559/memoozd/transition1x:/lustre06/project/6033559/memoozd/Transition1x:${PYTHONPATH:-}"

: "${SEARCH:?SEARCH must be gad or sella}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
mkdir -p "$OUTPUT_DIR"
N_WORKERS="${N_WORKERS:-1}"

if [[ "$SEARCH" == "gad" ]]; then
  python -u scripts/gad_smoke.py --backend horm --method leftnet \
    --sample-indices "${SAMPLE_IDS:-2}" --noise "${NOISE:-0.0}" \
    --n-steps "${N_STEPS:-1000}" --dt "${DT:-0.005}" \
    --descent-until-nneg "${DESCENT_UNTIL_NNEG:-0}" \
    --blend-sharpness "${BLEND_SHARPNESS:-0.0}" \
    --force-threshold 0.01 --force-criterion fmax --n-workers "$N_WORKERS" --output-dir "$OUTPUT_DIR"
elif [[ "$SEARCH" == "sella" ]]; then
  python -u scripts/sella_smoke.py --backend horm --method leftnet \
    --sample-indices "${SAMPLE_IDS:-2}" --noise "${NOISE:-0.0}" \
    --max-steps "${N_STEPS:-1000}" --fmax 0.01 --apply-eckart \
    --delta0 0.1 --gamma 0.4 --diag-every 1 --n-workers "$N_WORKERS" --output-dir "$OUTPUT_DIR"
else
  echo "Unknown SEARCH=$SEARCH" >&2
  exit 2
fi
