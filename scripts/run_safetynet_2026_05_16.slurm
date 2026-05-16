#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/safety_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/safety_%A_%a.err
#SBATCH --array=0-5

# Safety-net: partition the two at-risk cells (R2 + R3-b) across 3 sub-cells
# each so each sub-cell processes ~96 samples and finishes in <8h. The original
# 12h cells (61087603_2 and _4) may complete in time; if they don't, the
# log-fallback in integrate_comprehensive_2026_05_16.py uses both sources.
#
# Tasks:
#   0  R2-part1: Sella d=3 @ 200pm, samples [0, 96)
#   1  R2-part2: Sella d=3 @ 200pm, samples [96, 192)
#   2  R2-part3: Sella d=3 @ 200pm, samples [192, 287)
#   3  R3b-part1: Sella internal @ 200pm, samples [0, 96)
#   4  R3b-part2: Sella internal @ 200pm, samples [96, 192)
#   5  R3b-part3: Sella internal @ 200pm, samples [192, 287)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

T=$SLURM_ARRAY_TASK_ID

case $T in
  0) START=0;   END=96;  CFG=d3;       NOISE=0.20 ;;
  1) START=96;  END=192; CFG=d3;       NOISE=0.20 ;;
  2) START=192; END=287; CFG=d3;       NOISE=0.20 ;;
  3) START=0;   END=96;  CFG=internal; NOISE=0.20 ;;
  4) START=96;  END=192; CFG=internal; NOISE=0.20 ;;
  5) START=192; END=287; CFG=internal; NOISE=0.20 ;;
esac

if [ "$CFG" = "d3" ]; then
    OUT=/lustre07/scratch/memoozd/gadplus/runs/safetynet/sella_carteck_libdef_d3
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
        --n-samples 287 --max-steps 2000 --split test \
        --sample-start "$START" --sample-end "$END" \
        --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
        --diag-every 3 --config-tag "carteck_libdef_d3" \
        --output-dir "$OUT"
elif [ "$CFG" = "internal" ]; then
    OUT=/lustre07/scratch/memoozd/gadplus/runs/safetynet/sella_internal_default
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise "$NOISE" \
        --n-samples 287 --max-steps 2000 --split test \
        --sample-start "$START" --sample-end "$END" \
        --config-tag "internal_default" --output-dir "$OUT"
fi

echo "TASK $T DONE"
