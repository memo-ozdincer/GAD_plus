#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=painn-gad-dt
#SBATCH --array=0-4
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/painn_gad_dt_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/painn_gad_dt_%A_%a.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
PROJECT_DIR="/lustre06/project/6033559/memoozd/GAD_plus"
CANDIDATES="/lustre07/scratch/memoozd/gadplus/runs/painn-native-tight-65762463"
VALIDATION="$CANDIDATES/../painn-native-irc-65764388/summary.json"
DTS=(0.00025 0.0005 0.001 0.002 0.005)
DT="${DTS[$SLURM_ARRAY_TASK_ID]}"
OUT="/lustre07/scratch/memoozd/gadplus/runs/painn-gad-dt-${SLURM_ARRAY_JOB_ID}/dt-${DT}"
mkdir -p "$OUT" /lustre07/scratch/memoozd/gadplus/logs
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/painn_paired_pilot.py \
  --candidate-dir "$CANDIDATES" \
  --validation-summary "$VALIDATION" \
  --output-dir "$OUT" \
  --methods gad \
  --noise-pm 50 100 \
  --seeds 3 \
  --max-steps 300 \
  --gad-dt "$DT"
