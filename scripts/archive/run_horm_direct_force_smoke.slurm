#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --job-name=horm-leftdf-smoke
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/horm_leftdf_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/horm_leftdf_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source /lustre07/scratch/memoozd/venvs/horm/bin/activate
export PYTHONPATH="/lustre06/project/6033559/memoozd/GAD_plus/src:/lustre06/project/6033559/memoozd/transition1x:/lustre06/project/6033559/memoozd/Transition1x:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -u scripts/horm_leftnet_smoke.py \
  --checkpoint /lustre07/scratch/memoozd/models/horm/left-df.ckpt \
  --direct-forces \
  --sample-id 2 \
  --epsilons 0.01 0.003 0.001
