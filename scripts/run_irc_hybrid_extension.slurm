#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irchybridext_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irchybridext_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

METHODS=(  "hybrid_eckart" "hybrid_eckart" "hybrid_eckart" "hybrid_eckart" "hybrid" "hybrid" )
SWITCHES=( "true"          "true"          "true"          "true"          "false"  "false"  )
SF=(       "1e-2"          "1e-2"          "1e-2"          "1e-2"          "0.05"   "0.05"   )
NOISE_PM=( "30"            "50"            "150"           "200"           "10"     "100"    )

T=$SLURM_ARRAY_TASK_ID
METHOD=${METHODS[$T]}
SWITCH=${SWITCHES[$T]}
SF_VAL=${SF[$T]}
NPM=${NOISE_PM[$T]}

TR=0.05
GAD_DT=5e-3

if [ "$METHOD" = "hybrid" ]; then
    SOURCE_METHOD="${METHOD}_dt0.005_tr${TR}"
else
    if [ "$SWITCH" = "true" ]; then SW_TAG="swEIG"; else SW_TAG="swFORCE"; fi
    SOURCE_METHOD="${METHOD}_${SW_TAG}_dt0.005_tr${TR}"
fi

TAG="${METHOD}_sw${SWITCH}_dt${GAD_DT}_tr${TR}_sf${SF_VAL}"
SURVEY_DIR=/lustre07/scratch/memoozd/gadplus/runs/hybrid_extension/${TAG}_${NPM}pm
OUTDIR=/lustre07/scratch/memoozd/gadplus/runs/irc_hybrid_extension/${TAG}_${NPM}pm
mkdir -p "$OUTDIR"

echo "task=$T survey_dir=$SURVEY_DIR source_method=$SOURCE_METHOD noise=${NPM}pm"

python -u scripts/irc_validate.py \
    --method sella_hip --source-method "$SOURCE_METHOD" \
    --coords-source summary --all-endpoints \
    --noise-pm "$NPM" --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
    --survey-dir "$SURVEY_DIR" \
    --n-dataset-samples 300 --split test \
    --output-dir "$OUTDIR" \
    --no-write-viewer-bundles
