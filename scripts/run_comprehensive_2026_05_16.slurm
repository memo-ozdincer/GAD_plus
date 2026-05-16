#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --exclude=ng20304
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre07/scratch/memoozd/gadplus/logs/compr_%A_%a.out
#SBATCH --error=/lustre07/scratch/memoozd/gadplus/logs/compr_%A_%a.err
#SBATCH --array=0-8

# Comprehensive wave to close open questions from BENCHMARK_REPORT_2026-05-16.
#
# Cells:
#   0 R1-a: Hybrid damped Eckart eig tr=0.05 from REACTANT @ 0pm
#   1 R1-b: Hybrid undamped Eckart eig tr=0.05 from REACTANT @ 0pm
#   2 R2:   Sella cart+Eckart untuned d=3 @ 200pm  (fills the missing d=3 cell)
#   3 R3-a: Sella internal tuned d=1 @ 150pm  (replaces partial 222/287)
#   4 R3-b: Sella internal tuned d=1 @ 200pm  (replaces partial 196/287)
#   5 R4-a: GAD dt=0.005 × 10000 steps @ 50pm  (test fmax<0.001 reachability)
#   6 R4-b: Sella cart+Eckart untuned d=1 × 10000 steps @ 50pm
#   7 R4-c: Hybrid damped Eckart eig tr=0.05 × 10000 steps @ 50pm
#   8 R1-c: Sella cart+Eckart untuned d=1 from MIDPOINT @ 0pm (companion to reactant)

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 cuda/12.6
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
export WANDB_DISABLED=true PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

T=$SLURM_ARRAY_TASK_ID

case $T in
  0)
    # R1-a: Hybrid damped from reactant @ 0pm
    OUT=/lustre07/scratch/memoozd/gadplus/runs/start_reactant_hybrid/damped_dt5e-3_tr0.05
    mkdir -p "$OUT"
    python -u scripts/hybrid_gad_newton_runner.py \
        --method hybrid_damped_eckart --switch-by-eig true \
        --gad-dt 5e-3 --trust-radius 0.05 \
        --noise 0.0 --n-samples 287 --n-steps 2000 \
        --split test --force-threshold 0.01 \
        --start-from reactant \
        --output-dir "$OUT"
    ;;
  1)
    # R1-b: Hybrid undamped from reactant @ 0pm
    OUT=/lustre07/scratch/memoozd/gadplus/runs/start_reactant_hybrid/undamped_dt5e-3_tr0.05
    mkdir -p "$OUT"
    python -u scripts/hybrid_gad_newton_runner.py \
        --method hybrid_eckart --switch-by-eig true \
        --gad-dt 5e-3 --trust-radius 0.05 \
        --noise 0.0 --n-samples 287 --n-steps 2000 \
        --split test --force-threshold 0.01 \
        --start-from reactant \
        --output-dir "$OUT"
    ;;
  2)
    # R2: Sella cart+Eckart untuned d=3 @ 200pm
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_hessfreq/sella_carteck_libdef_d3
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.20 \
        --n-samples 287 --max-steps 2000 --split test \
        --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
        --diag-every 3 --config-tag "carteck_libdef_d3" \
        --output-dir "$OUT"
    ;;
  3)
    # R3-a: Sella internal tuned d=1 @ 150pm
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.15 \
        --n-samples 287 --max-steps 2000 --split test \
        --config-tag "internal_default" --output-dir "$OUT"
    ;;
  4)
    # R3-b: Sella internal tuned d=1 @ 200pm
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_set/sella_internal_default
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.20 \
        --n-samples 287 --max-steps 2000 --split test \
        --config-tag "internal_default" --output-dir "$OUT"
    ;;
  5)
    # R4-a: GAD dt=0.005 × 10000 steps @ 50pm (uses HIP via method_single.py)
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_longbudget/gad_dt005_10k
    mkdir -p "$OUT"
    python -u scripts/method_single.py \
        --method gad_dt005_fmax \
        --noise 0.05 --n-samples 287 \
        --n-steps 10000 --split test \
        --output-dir "$OUT"
    ;;
  6)
    # R4-b: Sella cart+Eckart untuned d=1 with max-steps=10000 @ 50pm and fmax=0.001 target
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_longbudget/sella_carteck_libdef_10k
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.001 --noise 0.05 \
        --n-samples 287 --max-steps 10000 --split test \
        --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
        --diag-every 1 --config-tag "carteck_libdef_10k" \
        --output-dir "$OUT"
    ;;
  7)
    # R4-c: Hybrid damped @ 50pm × 10000 steps with fmax=0.001
    OUT=/lustre07/scratch/memoozd/gadplus/runs/test_longbudget/hybrid_damped_10k
    mkdir -p "$OUT"
    python -u scripts/hybrid_gad_newton_runner.py \
        --method hybrid_damped_eckart --switch-by-eig true \
        --gad-dt 5e-3 --trust-radius 0.05 \
        --noise 0.05 --n-samples 287 --n-steps 10000 \
        --split test --force-threshold 0.001 \
        --output-dir "$OUT"
    ;;
  8)
    # R1-c: Sella from MIDPOINT @ 0pm (cross-family companion to reactant probe)
    OUT=/lustre07/scratch/memoozd/gadplus/runs/start_midpoint/sella_carteck_libdef
    mkdir -p "$OUT"
    python -u scripts/sella_baseline.py --fmax 0.01 --noise 0.0 \
        --n-samples 287 --max-steps 2000 --split test \
        --cartesian --apply-eckart --delta0 0.1 --gamma 0.4 \
        --diag-every 1 --config-tag "carteck_libdef_midpoint" \
        --start-from midpoint \
        --output-dir "$OUT"
    ;;
esac

echo "TASK $T DONE"
