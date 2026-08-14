#!/bin/bash
# Resume only missing HIP search/IRC sample files on a persistent H100 keeper.
set -euo pipefail

WORK_ROOT=/scratch/memoozd/GAD/GAD_plus
PYG_OVERLAY=/scratch/memoozd/gadplus/envs/hip-pyg27-overlay/lib/python3.11/site-packages
RUN_ROOT=${RUN_ROOT:-/scratch/memoozd/gadplus/runs/hip-cs2-h100-production-20260809-v2}
H5=${GADPLUS_T1X_H5:-/scratch/memoozd/GAD/data/transition1x.h5}
CHECKPOINT=${GADPLUS_HIP_CHECKPOINT:-/scratch/memoozd/GAD/models/hip_v2.ckpt}
KEEPER_JOB_ID=${KEEPER_JOB_ID:?set KEEPER_JOB_ID to the active persistent reservation}
KEEPER_STATE=/scratch/memoozd/gadplus/allocations/hip_h100_${KEEPER_JOB_ID}.json
N_SAMPLES=287
CONCURRENCY=4

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6
cd "$WORK_ROOT"
source .venv/bin/activate

keeper_output=$(.venv/bin/python - "$KEEPER_STATE" <<'PY'
import json, socket, sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text())
expected = state["hostname"].split(".")[0]
actual = socket.gethostname().split(".")[0]
if expected != actual:
    raise SystemExit(f"keeper host mismatch: allocation={expected}, shell={actual}")
gpu = state.get("cuda_visible_devices") or state.get("slurm_job_gpus")
cpus = state.get("cpu_affinity")
if not gpu or not cpus:
    raise SystemExit(f"incomplete keeper state: {state}")
print(gpu)
print(",".join(map(str, cpus)))
PY
)
readarray -t keeper_values <<< "$keeper_output"
export CUDA_VISIBLE_DEVICES=${keeper_values[0]}
taskset --cpu-list --pid "${keeper_values[1]}" "$$" >/dev/null
export PYTHONPATH="$PYG_OVERLAY:$WORK_ROOT/src"
export OMP_NUM_THREADS=2 GADPLUS_WORKER_THREADS=2
export PYTHONUNBUFFERED=1 WANDB_DISABLED=true
export TMPDIR=/scratch/memoozd/gadplus/tmp
mkdir -p "$TMPDIR" "$RUN_ROOT/allocations" /scratch/memoozd/gadplus/logs

printf '%s  %s\n' \
  154d658f9c5d0b082a9c4893f3978038494d2499794a5ac647448fe397f2d1cb "$CHECKPOINT" \
  6a20f8a3f49c50d462270d10d4c44ca102e788072e2096a91d70b5a0f598b629 "$H5" \
  | sha256sum --check --strict

python - "$RUN_ROOT" "$KEEPER_JOB_ID" <<'PY'
import hashlib, json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
root, job = Path(sys.argv[1]), sys.argv[2]
work = Path.cwd()
manifest = json.loads((root / "code_manifest.json").read_text())
bad = []
for name, expected in manifest["files_sha256"].items():
    observed = hashlib.sha256((work / name).read_bytes()).hexdigest()
    if observed != expected:
        bad.append((name, expected, observed))
if bad:
    raise SystemExit(f"frozen scientific code manifest mismatch: {bad}")
payload = {
    "keeper_job_id": job,
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "launched_at_utc": datetime.now(timezone.utc).isoformat(),
    "orchestrator": "resume_hip_missing_on_reserved_h100.sh",
}
path = root / "allocations" / f"keeper_{job}.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

python - <<'PY'
import torch
from torch_scatter import segment_coo
assert torch.cuda.is_available()
value = segment_coo(
    torch.tensor([1.0, 2.0, 3.0], device="cuda"),
    torch.tensor([0, 0, 1], device="cuda"),
)
assert torch.equal(value.cpu(), torch.tensor([3.0, 3.0]))
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
PY

missing_ids() {
  python - "$1" "$N_SAMPLES" <<'PY'
import sys
from pathlib import Path
root, count = Path(sys.argv[1]), int(sys.argv[2])
for sample_id in range(count):
    if not (root / f"sample_{sample_id:03d}.json").is_file():
        print(sample_id)
PY
}

run_batched() {
  local status=0
  local -a pids=()
  for sample_id in "${sample_ids[@]}"; do
    "$@" "$sample_id" &
    pids+=("$!")
    if (( ${#pids[@]} == CONCURRENCY )); then
      for pid in "${pids[@]}"; do wait "$pid" || status=1; done
      pids=()
      (( status == 0 )) || return 1
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}

run_search_sample() {
  local sample_id=$1
  python scripts/hip_cs2_benchmark.py worker \
    --output-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
    --noise 0.20 --seed 42 --n-samples "$N_SAMPLES" --max-steps 5000 \
    --step-fraction 0.01 --spectral-temperature 0.01 \
    --n-shards "$N_SAMPLES" --shard-id "$sample_id" --device cuda --resume
}

mapfile -t sample_ids < <(missing_ids "$RUN_ROOT/noise_0.20A/samples")
echo "search noise=0.20 missing=${#sample_ids[@]}" flush=true
run_batched run_search_sample

python - "$RUN_ROOT/noise_0.20A/shards" "$N_SAMPLES" <<'PY'
import json, os, sys
from pathlib import Path
root, count = Path(sys.argv[1]), int(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
for sample_id in range(count):
    path = root / f"shard_{sample_id:03d}.json"
    temporary = root / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps({"shard_id": sample_id, "sample_ids": [sample_id]}) + "\n")
    os.replace(temporary, path)
PY
python scripts/hip_cs2_benchmark.py aggregate \
  --output-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
  --noise 0.20 --seed 42 --n-samples "$N_SAMPLES" --max-steps 5000 \
  --step-fraction 0.01 --spectral-temperature 0.01 \
  --n-shards "$N_SAMPLES" --expected-shards "$N_SAMPLES" --device cuda

run_irc_sample() {
  local sample_id=$1
  python scripts/hip_cs2_irc_topo.py worker \
    --search-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
    --noise "$irc_noise" --n-samples "$N_SAMPLES" --split test --irc-steps 500 \
    --rmsd-threshold 0.3 --n-shards "$N_SAMPLES" --shard-id "$sample_id" \
    --device cuda --resume
}

for irc_noise in 0.10 0.15 0.20; do
  irc_root="$RUN_ROOT/irc_topo/noise_${irc_noise}A"
  mapfile -t sample_ids < <(missing_ids "$irc_root/samples")
  echo "IRC noise=$irc_noise missing=${#sample_ids[@]}" flush=true
  run_batched run_irc_sample
  python - "$irc_root/shards" "$N_SAMPLES" <<'PY'
import json, os, sys
from pathlib import Path
root, count = Path(sys.argv[1]), int(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
for sample_id in range(count):
    path = root / f"shard_{sample_id:03d}.json"
    temporary = root / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps({"shard_id": sample_id, "sample_ids": [sample_id]}) + "\n")
    os.replace(temporary, path)
PY
  python scripts/hip_cs2_irc_topo.py aggregate \
    --search-root "$RUN_ROOT" --h5 "$H5" --checkpoint "$CHECKPOINT" \
    --noise "$irc_noise" --n-samples "$N_SAMPLES" --split test --irc-steps 500 \
    --rmsd-threshold 0.3 --n-shards "$N_SAMPLES" \
    --expected-shards "$N_SAMPLES" --device cuda
done

python scripts/summarize_hip_cs2_results.py "$RUN_ROOT"
