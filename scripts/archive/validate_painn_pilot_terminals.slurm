#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=painn-terminal-irc
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/painn_terminal_irc_%j.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/painn_terminal_irc_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
SUMMARY="${1:?pass paired pilot summary.parquet}"
CANDIDATES="${2:?pass PaiNN candidate directory}"
OUT="/lustre07/scratch/memoozd/gadplus/runs/painn-terminal-irc-${SLURM_JOB_ID}"
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -u scripts/validate_painn_pilot_terminals.py \
  --summary "$SUMMARY" --candidate-dir "$CANDIDATES" --output-dir "$OUT"
