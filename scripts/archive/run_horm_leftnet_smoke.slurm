#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/horm_leftnet_%A.out

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6

cd /lustre06/project/6033559/memoozd/GAD_plus
source /lustre07/scratch/memoozd/venvs/horm/bin/activate
export PYTHONPATH="/lustre06/project/6033559/memoozd/GAD_plus/src:/lustre06/project/6033559/memoozd/transition1x:/lustre06/project/6033559/memoozd/Transition1x:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -u scripts/horm_leftnet_smoke.py \
  --checkpoint "${CHECKPOINT:-/lustre07/scratch/memoozd/models/horm/left.ckpt}" \
  --sample-id "${SAMPLE_ID:-2}" \
  --epsilon "${EPSILON:-0.001}"
