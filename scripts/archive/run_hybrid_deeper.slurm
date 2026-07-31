#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/hybriddeep_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/hybriddeep_%A_%a.err
#SBATCH --array=0-9

# Deeper hybrid sweep with Newton actually firing, to disambiguate
# Eckart / damping / switch / tr confounds.
# Fixed: gad_dt=5e-3, tr=0.05, n_samples=287 test, n_steps=1000.
#
# Cells (method, switch_kind, switch_value, noise):
#   0: hybrid (no-Eckart),       force, 1e-2,  10pm
#   1: hybrid (no-Eckart),       force, 1e-2, 100pm
#   2: hybrid_eckart_swfalse,    force, 1e-2,  10pm
#   3: hybrid_eckart_swfalse,    force, 1e-2, 100pm
#   4: hybrid_damped_eckart_swfalse, force, 1e-2,  10pm
#   5: hybrid_damped_eckart_swfalse, force, 1e-2, 100pm
#   6: hybrid_eckart_swtrue,     eig,   ----,  10pm    (undamped Eckart with eig-switch)
#   7: hybrid_eckart_swtrue,     eig,   ----, 100pm
#   8: hybrid (no-Eckart),       force, 1e-3,  10pm    (tr-effect control vs existing tr=0.005)
#   9: hybrid (no-Eckart),       force, 1e-3, 100pm

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

METHODS=( "hybrid" "hybrid" "hybrid_eckart" "hybrid_eckart" "hybrid_damped_eckart" "hybrid_damped_eckart" "hybrid_eckart" "hybrid_eckart" "hybrid" "hybrid" )
SWITCHES=( "false" "false" "false" "false" "false" "false" "true" "true" "false" "false" )
SF=(       "1e-2"  "1e-2"  "1e-2"  "1e-2"  "1e-2"  "1e-2"  "1e-2" "1e-2" "1e-3"  "1e-3" )
NOISES=(   "0.01"  "0.10"  "0.01"  "0.10"  "0.01"  "0.10"  "0.01" "0.10" "0.01"  "0.10" )

T=$SLURM_ARRAY_TASK_ID
METHOD=${METHODS[$T]}
SWITCH=${SWITCHES[$T]}
SF_VAL=${SF[$T]}
NOISE=${NOISES[$T]}
NOISE_PM=$(python -c "print(int(round($NOISE*1000)))")

TR=0.05
GAD_DT=5e-3

# OUTDIR includes sf so cells don't clash. method_tag (inside summary
# filename) does NOT include sf -- but that's OK because each OUTDIR has
# only one cell.
TAG="${METHOD}_sw${SWITCH}_dt${GAD_DT}_tr${TR}_sf${SF_VAL}"
OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/hybrid_deeper/${TAG}_${NOISE_PM}pm
mkdir -p "$OUTDIR"

echo "task=$T method=$METHOD switch=$SWITCH sf=$SF_VAL noise=${NOISE_PM}pm tr=$TR"

python -u scripts/hybrid_gad_newton_runner.py \
    --method "$METHOD" \
    --switch-by-eig "$SWITCH" \
    --switch-force "$SF_VAL" \
    --gad-dt "$GAD_DT" \
    --trust-radius "$TR" \
    --noise "$NOISE" \
    --n-samples 287 \
    --n-steps 1000 \
    --split test \
    --output-dir "$OUTDIR"
