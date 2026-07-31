#!/bin/bash
#SBATCH --account=def-aspuru
#SBATCH --job-name=horm-native-shard
#SBATCH --array=0-3
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/horm_native_shard_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/horm_native_shard_%A_%a.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
START=$((SLURM_ARRAY_TASK_ID * 25))
OUT="/lustre07/scratch/memoozd/gadplus/runs/horm-native-shards-${SLURM_ARRAY_JOB_ID}/shard-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source /lustre07/scratch/memoozd/venvs/horm/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:/lustre06/project/6033559/memoozd/transition1x:/lustre06/project/6033559/memoozd/Transition1x:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/build_painn_native_saddles.py \
  --backend horm \
  --n-samples 1 \
  --start-sample "$START" \
  --max-examined 25 \
  --endpoint-fmax 0.005 \
  --neb-fmax 0.01 \
  --endpoint-max-steps 500 \
  --neb-max-steps 500 \
  --output-dir "$OUT"
