#!/bin/bash
#SBATCH --job-name=hip_h100_keeper
#SBATCH --account=def-aspuru
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/hip_h100_keeper_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/hip_h100_keeper_%j.err

# Persistent interactive H100 reservation. Once RUNNING, SSH to the node shown
# by `squeue -j JOB_ID`, run HIP work there, and release it with `scancel JOB_ID`.
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

mkdir -p /scratch/memoozd/gadplus/logs /scratch/memoozd/gadplus/tmp
cd /scratch/memoozd/GAD/GAD_plus
exec .venv/bin/python scripts/keep_gpu_allocation.py \
  --state-file "/scratch/memoozd/gadplus/allocations/hip_h100_${SLURM_JOB_ID}.json"
