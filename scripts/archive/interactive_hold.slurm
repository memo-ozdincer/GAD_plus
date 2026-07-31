#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hold_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hold_%j.err
#
# Holds an A100 node for 3 hours for interactive use.
# Attach with:  srun --jobid=<JOBID> --pty bash
#
# Submit:  sbatch scripts/interactive_hold.slurm

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "Sleeping for 3 hours..."

sleep 10800
