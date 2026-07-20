#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=horm-leftnet-fd
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/horm_leftnet_fd_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/horm_leftnet_fd_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
cd "$PROJECT_DIR"
source /lustre07/scratch/memoozd/venvs/horm/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:/lustre06/project/6033559/memoozd/transition1x:/lustre06/project/6033559/memoozd/Transition1x:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1

for checkpoint in /lustre07/scratch/memoozd/models/horm/left.ckpt /lustre07/scratch/memoozd/models/horm/left_orig.ckpt; do
  echo "CHECKPOINT=$checkpoint"
  python -u scripts/horm_leftnet_smoke.py --checkpoint "$checkpoint" --sample-id 2 \
    --epsilons 0.01 0.003 0.001 0.0003
done
