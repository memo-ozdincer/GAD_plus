#!/bin/bash
#SBATCH --job-name=t1x_pure_gad_shell
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/gadplus/logs/t1x_pure_gad_shell_%j.out
#SBATCH --error=/scratch/memoozd/gadplus/logs/t1x_pure_gad_shell_%j.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.11.5
PROJECT_DIR=/scratch/memoozd/GAD/GAD_plus
SOURCE_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-ablation-1950894/eta001_tau001
OUT_ROOT=/scratch/memoozd/gadplus/runs/t1x-intrinsic-gxtb-pure-gad-shell-${SLURM_JOB_ID}
export GADPLUS_SCRATCH=/scratch/memoozd/gadplus
export GADPLUS_T1X_H5=/scratch/memoozd/GAD/data/transition1x.h5
export GADPLUS_GXTB_EXE=/scratch/memoozd/GAD/third_party/g-xtb/xtb-6.7.1/bin/xtb
export GADPLUS_GXTB_PARALLEL=32 GADPLUS_ENV="$PROJECT_DIR/.venv"
export GADPLUS_GXTB_WORK_ROOT=/scratch/memoozd/gadplus/tmp/gxtb_pure_gad_shell_${SLURM_JOB_ID}
source "$PROJECT_DIR/scripts/lib/gxtb_runtime.sh"
export PYTHON PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
for offset in 0 6 12 18; do
  srun --ntasks=6 --ntasks-per-node=6 --cpus-per-task=32 --cpu-bind=cores \
    bash -c 'task=$((SLURM_PROCID + '"$offset"')); "$PYTHON" '"$PROJECT_DIR"'/scripts/t1x_intrinsic_gxtb_minimum_shell.py --source-summary '"$SOURCE_ROOT"'/summary.json --source-root '"$SOURCE_ROOT"' --output-root '"$OUT_ROOT"' --task-id "$task" --gxtb-executable "$GADPLUS_GXTB_EXE" --parallel "$GADPLUS_GXTB_PARALLEL" --gate-variant gad --direction softest --kappas 0.025,0.075,0.15 --max-steps 300'
done
echo "run_root=$OUT_ROOT"
