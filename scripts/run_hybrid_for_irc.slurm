#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hybridcoords_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hybridcoords_%A_%a.err
#SBATCH --array=0-9

# Hybrid sweep WITH coords logged for IRC validation.
# 10 cells:
#   0-5: damped_eckart_swtrue tr=0.05 × {10,30,50,100,150,200} pm  (canonical config, full noise sweep)
#   6:   no-Eckart hybrid force-switch tr=0.005 @ 10pm  (best 10pm, ties plain GAD)
#   7:   no-Eckart hybrid force-switch tr=0.005 @ 100pm
#   8:   no-Eckart hybrid force-switch tr=0.10  @ 10pm
#   9:   no-Eckart hybrid force-switch tr=0.10  @ 100pm

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# (method, switch, trust_radius, noise) per task index
METHODS=(  "hybrid_damped_eckart" "hybrid_damped_eckart" "hybrid_damped_eckart" "hybrid_damped_eckart" "hybrid_damped_eckart" "hybrid_damped_eckart" "hybrid" "hybrid" "hybrid" "hybrid" )
SWITCHES=( "true" "true" "true" "true" "true" "true" "false" "false" "false" "false" )
TRUSTS=(   "0.05" "0.05" "0.05" "0.05" "0.05" "0.05" "0.005" "0.005" "0.10" "0.10" )
NOISES=(   "0.01" "0.03" "0.05" "0.10" "0.15" "0.20" "0.01"  "0.10"  "0.01" "0.10" )

T=$SLURM_ARRAY_TASK_ID
METHOD=${METHODS[$T]}
SWITCH=${SWITCHES[$T]}
TR=${TRUSTS[$T]}
NOISE=${NOISES[$T]}
NOISE_PM=$(python -c "print(int(round($NOISE*1000)))")

TAG="${METHOD}_sw${SWITCH}_dt5e-3_tr${TR}"
OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/hybrid_for_irc/${TAG}_${NOISE_PM}pm
mkdir -p "$OUTDIR"

echo "task=$T method=$METHOD switch=$SWITCH noise=${NOISE_PM}pm trust=$TR"

python -u scripts/hybrid_gad_newton_runner.py \
    --method "$METHOD" \
    --switch-by-eig "$SWITCH" \
    --gad-dt 5e-3 \
    --trust-radius "$TR" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 1000 \
    --split test \
    --output-dir "$OUTDIR"
