#!/bin/bash
#SBATCH --job-name=hip_cs2
#SBATCH --account=def-aspuru
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-35
#SBATCH --output=/scratch/memoozd/gadplus/logs/hip_cs2_%A_%a.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/hip_cs2_%A_%a.err

# H100 handoff launcher: three high-noise cells times twelve immutable shards.
# Change only the GPU resource spelling/account if required at the destination.
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
PYG_OVERLAY=/scratch/memoozd/gadplus/envs/hip-pyg27-overlay/lib/python3.11/site-packages
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/hip-cs2-${SLURM_ARRAY_JOB_ID}}
H5=${GADPLUS_T1X_H5:-/scratch/memoozd/GAD/data/transition1x.h5}
CHECKPOINT=${GADPLUS_HIP_CHECKPOINT:-/scratch/memoozd/GAD/models/hip_v2.ckpt}
NOISES=(0.10 0.15 0.20)
N_SHARDS=12
NOISE_INDEX=$((SLURM_ARRAY_TASK_ID / N_SHARDS))
SHARD_ID=$((SLURM_ARRAY_TASK_ID % N_SHARDS))
NOISE=${NOISES[$NOISE_INDEX]}

mkdir -p "$RUN_ROOT" /scratch/memoozd/gadplus/logs /scratch/memoozd/gadplus/tmp
cd "$WORK_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PYG_OVERLAY:$WORK_ROOT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONUNBUFFERED=1 WANDB_DISABLED=true
export TMPDIR=/scratch/memoozd/gadplus/tmp

python scripts/hip_cs2_benchmark.py worker \
  --output-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
  --noise "$NOISE" --seed 42 --n-samples 287 --max-steps 5000 \
  --step-fraction 0.01 --spectral-temperature 0.01 \
  --n-shards "$N_SHARDS" --shard-id "$SHARD_ID" --device cuda
