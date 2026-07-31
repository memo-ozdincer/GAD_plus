#!/usr/bin/env bash
# Replay immutable per-trajectory records into W&B, one process at a time.
#
# A single stream is intentional: the W&B service spawns helper processes, so
# concurrent uploaders can exhaust a login-node process quota. Deterministic
# run IDs make this safely resumable after interruption.
set -uo pipefail

cd /scratch/memoozd/GAD/GAD_plus
set -a
source /scratch/memoozd/GAD/secrets/wandb.env
set +a
export WANDB_SILENT=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src

replay_regular() {
  local noise=$1 budget=$2 job_id=$3
  local noise_value=${noise/p/.}
  .venv/bin/python scripts/export_regular_gad_trace_campaign.py \
    "/scratch/memoozd/gadplus/runs/t1x-gxtb-grid-regular_gad-${noise}-${job_id}" \
    --noise "$noise_value" --budget "$budget" --group t1x-gxtb-matched-noise-grid --mode online
}

replay_sella() {
  local noise=$1 job_id=$2
  .venv/bin/python scripts/export_sella_trace_campaign.py \
    "/scratch/memoozd/gadplus/runs/t1x-gxtb-grid-sella-${noise}-${job_id}" \
    --group t1x-gxtb-matched-noise-grid --mode online
}

replay_competitive() {
  local family=$1 noise=$2 job_id=$3 job_type=$4
  .venv/bin/python scripts/export_wandb_campaign.py \
    "/scratch/memoozd/gadplus/runs/t1x-gxtb-grid-${family}-${noise}-${job_id}" \
    --group t1x-gxtb-matched-noise-grid --job-type "$job_type" --mode online
}

replay_regular 0p10 300 1984934 || echo "regular 0p10 replay reported failures"
replay_regular 0p20 300 1984938 || echo "regular 0p20 replay reported failures"
replay_regular 1p00 2000 1984943 || echo "regular 1p00 replay reported failures"
replay_sella 0p10 1984937 || echo "Sella 0p10 replay reported failures"
replay_sella 0p20 1984941 || echo "Sella 0p20 replay reported failures"
replay_sella 1p00 1984946 || echo "Sella 1p00 replay reported failures"
replay_competitive competitive 0p10 1984935 competitive-gad || echo "competitive 0p10 replay reported failures"
replay_competitive competitive 0p20 1984939 competitive-gad || echo "competitive 0p20 replay reported failures"
replay_competitive competitive 1p00 1984944 competitive-gad || echo "competitive 1p00 replay reported failures"
replay_competitive competitive_subspace 0p10 1984936 competitive-subspace-gad || echo "subspace 0p10 replay reported failures"
replay_competitive competitive_subspace 0p20 1984940 competitive-subspace-gad || echo "subspace 0p20 replay reported failures"
replay_competitive competitive_subspace 1p00 1984945 competitive-subspace-gad || echo "subspace 1p00 replay reported failures"
