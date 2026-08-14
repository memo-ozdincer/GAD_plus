#!/bin/bash
# Run the frozen HIP CS2 campaign from an SSH shell on a reserved H100 node.
set -euo pipefail

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
PYG_OVERLAY=/scratch/memoozd/gadplus/envs/hip-pyg27-overlay/lib/python3.11/site-packages
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/hip-cs2-h100-production-20260809}
H5=${GADPLUS_T1X_H5:-/scratch/memoozd/GAD/data/transition1x.h5}
CHECKPOINT=${GADPLUS_HIP_CHECKPOINT:-/scratch/memoozd/GAD/models/hip_v2.ckpt}
KEEPER_JOB_ID=${KEEPER_JOB_ID:?set KEEPER_JOB_ID to the active persistent reservation}
KEEPER_STATE=/scratch/memoozd/gadplus/allocations/hip_h100_${KEEPER_JOB_ID}.json
# Four disjoint workers keep the H100 occupied while respecting the keeper's
# eight allocated CPU cores. Each holds one 233 MB checkpoint copy; per-sample
# JSON remains the restart boundary. The array template retains 12 shards for
# independent scheduled jobs.
N_SHARDS=${N_SHARDS:-4}
RUN_IRC_TOPO=${RUN_IRC_TOPO:-1}

campaign_pgid=$(ps -o pgid= -p "$$" | tr -d ' ')
if [[ "$campaign_pgid" != "$$" ]]; then
  echo "launch this campaign with setsid so its allocation watchdog is isolated" >&2
  exit 2
fi
if [[ $(/opt/slurm/bin/squeue -h -j "$KEEPER_JOB_ID" -o '%T') != RUNNING ]]; then
  echo "keeper job $KEEPER_JOB_ID is not RUNNING" >&2
  exit 2
fi
watch_keeper() {
  while [[ $(/opt/slurm/bin/squeue -h -j "$KEEPER_JOB_ID" -o '%T') == RUNNING ]]; do
    sleep 30
  done
  echo "keeper job $KEEPER_JOB_ID ended; terminating campaign process group" >&2
  kill -TERM -- "-$campaign_pgid"
}
watch_keeper &
watchdog_pid=$!
cleanup_watchdog() {
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
}
trap cleanup_watchdog EXIT

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6
cd "$WORK_ROOT"
source .venv/bin/activate
keeper_output=$(.venv/bin/python - "$KEEPER_STATE" <<'PY'
import json
import socket
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"keeper state is missing: {path}")
state = json.loads(path.read_text())
expected_host = state.get("hostname", "").split(".")[0]
actual_host = socket.gethostname().split(".")[0]
if expected_host != actual_host:
    raise SystemExit(f"keeper host mismatch: allocation={expected_host}, shell={actual_host}")
gpu = state.get("cuda_visible_devices") or state.get("slurm_job_gpus")
if not gpu:
    raise SystemExit(f"keeper state has no assigned GPU: {state}")
cpus = state.get("cpu_affinity")
if not cpus:
    raise SystemExit(f"keeper state has no CPU affinity: {state}")
print(expected_host)
print(gpu)
print(",".join(str(cpu) for cpu in cpus))
PY
)
readarray -t keeper_values <<< "$keeper_output"
export CUDA_VISIBLE_DEVICES=${keeper_values[1]}
taskset --cpu-list --pid "${keeper_values[2]}" "$$" >/dev/null
export PYTHONPATH="$PYG_OVERLAY:$WORK_ROOT/src"
export OMP_NUM_THREADS=2 GADPLUS_WORKER_THREADS=2
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true
export TMPDIR=/scratch/memoozd/gadplus/tmp
mkdir -p "$RUN_ROOT" "$TMPDIR" /scratch/memoozd/gadplus/logs

printf '%s  %s\n' \
  154d658f9c5d0b082a9c4893f3978038494d2499794a5ac647448fe397f2d1cb "$CHECKPOINT" \
  6a20f8a3f49c50d462270d10d4c44ca102e788072e2096a91d70b5a0f598b629 "$H5" \
  | sha256sum --check --strict

python - "$RUN_ROOT" "$KEEPER_JOB_ID" <<'PY'
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
keeper_job_id = sys.argv[2]
work = Path.cwd()
files = [
    "scripts/hip_cs2_benchmark.py",
    "scripts/hip_cs2_irc_topo.py",
    "scripts/summarize_hip_cs2_results.py",
    "scripts/run_hip_cs2_on_reserved_h100.sh",
    "src/gadplus/calculator/ase_adapter.py",
    "src/gadplus/calculator/hip.py",
    "src/gadplus/core/convergence.py",
    "src/gadplus/data/transition1x.py",
    "src/gadplus/search/intrinsic_gad.py",
    "src/gadplus/search/irc_sella_hip.py",
    "src/gadplus/search/irc_validate.py",
]
files.extend(
    str(path.relative_to(work))
    for path in sorted((work / "src/gadplus/projection").rglob("*.py"))
)
manifest = {
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip(),
    "files_sha256": {
        name: hashlib.sha256((work / name).read_bytes()).hexdigest()
        for name in sorted(set(files))
    },
}
manifest_path = root / "code_manifest.json"
if manifest_path.exists():
    if json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit(
            f"code manifest mismatch in {manifest_path}; use a new output root"
        )
else:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
allocation_root = root / "allocations"
allocation_root.mkdir(exist_ok=True)
(allocation_root / f"keeper_{keeper_job_id}.json").write_text(
    json.dumps(
        {
            "keeper_job_id": keeper_job_id,
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

python - <<'PY'
import os
import torch
import torch_cluster
import torch_geometric
import torch_scatter
from torch_scatter import segment_coo

assert torch.cuda.is_available(), "reserved-node campaign requires a visible CUDA GPU"
assert torch_geometric.__version__ == "2.7.0"
assert torch_cluster.__version__ == "1.6.3+pt27cu126"
assert torch_scatter.__version__ == "2.1.2+pt27cu126"
device = torch.device("cuda")
segment = segment_coo(
    torch.tensor([1.0, 2.0, 3.0], device=device),
    torch.tensor([0, 0, 1], device=device),
)
assert torch.equal(segment.cpu(), torch.tensor([3.0, 3.0]))
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
print(f"cuda_visible_devices={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
print(
    f"torch-geometric={torch_geometric.__version__} "
    f"torch-cluster={torch_cluster.__version__} "
    f"torch-scatter={torch_scatter.__version__}",
    flush=True,
)
PY

SMOKE_ROOT="$RUN_ROOT/preflight_four_worker"
smoke_pids=()
for shard_id in $(seq 0 $((N_SHARDS - 1))); do
  python scripts/hip_cs2_benchmark.py worker \
    --output-root "$SMOKE_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
    --noise 0.15 --seed 42 --n-samples 287 --sample-ids 0,1,2,3 --max-steps 2 \
    --step-fraction 0.01 --spectral-temperature 0.01 \
    --n-shards "$N_SHARDS" --shard-id "$shard_id" --device cuda --resume &
  smoke_pids+=("$!")
done
smoke_status=0
for smoke_pid in "${smoke_pids[@]}"; do
  wait "$smoke_pid" || smoke_status=1
done
(( smoke_status == 0 )) || exit 1
python scripts/hip_cs2_benchmark.py aggregate \
  --output-root "$SMOKE_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
  --noise 0.15 --seed 42 --n-samples 287 --sample-ids 0,1,2,3 --max-steps 2 \
  --step-fraction 0.01 --spectral-temperature 0.01 \
  --n-shards "$N_SHARDS" --expected-shards "$N_SHARDS" --device cuda
jq -e 'select(.planned == 4 and .calculator_valid == 4 and .errors == 0)' \
  "$SMOKE_ROOT/noise_0.15A/summary.json" >/dev/null

for noise in 0.10 0.15 0.20; do
  worker_pids=()
  for shard_id in $(seq 0 $((N_SHARDS - 1))); do
    python scripts/hip_cs2_benchmark.py worker \
      --output-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
      --noise "$noise" --seed 42 --n-samples 287 --max-steps 5000 \
      --step-fraction 0.01 --spectral-temperature 0.01 \
      --n-shards "$N_SHARDS" --shard-id "$shard_id" --device cuda --resume &
    worker_pids+=("$!")
  done
  worker_status=0
  for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid" || worker_status=1
  done
  (( worker_status == 0 )) || exit 1
  python scripts/hip_cs2_benchmark.py aggregate \
    --output-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
    --noise "$noise" --seed 42 --n-samples 287 --max-steps 5000 \
    --step-fraction 0.01 --spectral-temperature 0.01 \
    --n-shards "$N_SHARDS" --expected-shards "$N_SHARDS" --device cuda
done

if [[ "$RUN_IRC_TOPO" == 1 ]]; then
  for noise in 0.10 0.15 0.20; do
    worker_pids=()
    for shard_id in $(seq 0 $((N_SHARDS - 1))); do
      python scripts/hip_cs2_irc_topo.py worker \
        --search-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
        --noise "$noise" --n-samples 287 --split test --irc-steps 500 \
        --rmsd-threshold 0.3 --n-shards "$N_SHARDS" --shard-id "$shard_id" \
        --device cuda --resume &
      worker_pids+=("$!")
    done
    worker_status=0
    for worker_pid in "${worker_pids[@]}"; do
      wait "$worker_pid" || worker_status=1
    done
    (( worker_status == 0 )) || exit 1
    python scripts/hip_cs2_irc_topo.py aggregate \
      --search-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
      --noise "$noise" --n-samples 287 --split test --irc-steps 500 \
      --rmsd-threshold 0.3 --n-shards "$N_SHARDS" \
      --expected-shards "$N_SHARDS" --device cuda
  done
  python scripts/summarize_hip_cs2_results.py "$RUN_ROOT"
else
  python scripts/summarize_hip_cs2_results.py "$RUN_ROOT" --allow-missing-irc
fi
