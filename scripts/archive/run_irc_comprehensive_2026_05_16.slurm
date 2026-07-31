#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/irccompr_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/irccompr_%A_%a.err
#SBATCH --array=0-3

# IRC TOPO validation for the new cells from wave 1 (job 61087603 / 61087774 / 61088001).
# Pre-written; submit after `squeue -u memoozd` shows the wave 1 cells gone
# and the canonical pooled-summary parquets exist (run build_pooled_summaries
# first to merge main + safety-net pieces).
#
# Cells:
#   0  Sella d=3 @ 200pm                — closes d=1 vs d=3 chemistry verdict at 200pm
#   1  Sella internal d=1 @ 150pm       — replaces partial / sample-biased IRC TOPO
#   2  Sella internal d=1 @ 200pm       — replaces partial / sample-biased IRC TOPO
#   3  Sella libdef from midpoint @ 0pm — answers "is the 53% midpoint conv real-basin?"

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

T=$SLURM_ARRAY_TASK_ID

case $T in
  0)
    # d=3 @ 200pm — pool main + safety-net first
    SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_hessfreq/sella_carteck_libdef_d3
    OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_libdef_d3_2026_05_16
    POOLED=$SURVEY/pooled_summary_200pm.parquet
    if [ ! -f "$POOLED" ]; then
        echo "Pooled summary missing; run build_pooled_summaries_2026_05_16.py first"
        exit 1
    fi
    # The IRC script expects the survey-dir glob to find one summary file.
    # Stage by copying the pooled file to the standard survey filename.
    cp "$POOLED" "$SURVEY/summary_sella_cartesian_eckart_fmax0p01_carteck_libdef_d3_200pm.parquet"
    mkdir -p "$OUT"
    python -u scripts/irc_validate.py \
        --method sella_hip \
        --source-method "sella_cartesian_eckart_fmax0p01_carteck_libdef_d3" \
        --coords-source summary --all-endpoints \
        --noise-pm 200 --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
        --survey-dir "$SURVEY" \
        --n-dataset-samples 300 --split test \
        --output-dir "$OUT" \
        --no-write-viewer-bundles
    ;;
  1|2)
    # Sella internal @ 150 / 200 pm
    NPM=$([ $T -eq 1 ] && echo 150 || echo 200)
    SURVEY=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
    OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_internal_2026_05_16/${NPM}pm
    if [ $T -eq 2 ]; then
        # 200pm uses pooled main+safety-net
        POOLED=$SURVEY/pooled_summary_200pm.parquet
        if [ ! -f "$POOLED" ]; then
            echo "Pooled summary 200pm missing; run build_pooled_summaries_2026_05_16.py first"
            exit 1
        fi
        cp "$POOLED" "$SURVEY/summary_sella_internal_fmax0p01_internal_default_200pm.parquet"
    fi
    mkdir -p "$OUT"
    python -u scripts/irc_validate.py \
        --method sella_hip \
        --source-method "sella_internal_fmax0p01_internal_default" \
        --coords-source summary --all-endpoints \
        --noise-pm "$NPM" --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
        --survey-dir "$SURVEY" \
        --n-dataset-samples 300 --split test \
        --output-dir "$OUT" \
        --no-write-viewer-bundles
    ;;
  3)
    # Midpoint @ 0pm
    SURVEY=/lustre07/scratch/memoozd/gadplus/runs/start_midpoint/sella_carteck_libdef
    OUT=/lustre07/scratch/memoozd/gadplus/runs/irc_sella_midpoint_2026_05_16
    mkdir -p "$OUT"
    python -u scripts/irc_validate.py \
        --method sella_hip \
        --source-method "sella_cartesian_eckart_fmax0p01_carteck_libdef_midpoint" \
        --coords-source summary --all-endpoints \
        --noise-pm 0 --max-validate 300 --irc-steps 500 --rmsd-threshold 0.3 \
        --survey-dir "$SURVEY" \
        --n-dataset-samples 300 --split test \
        --output-dir "$OUT" \
        --no-write-viewer-bundles
    ;;
esac

echo "TASK $T DONE"
