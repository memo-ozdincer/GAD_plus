"""Keep a Slurm GPU allocation alive until the job is explicitly cancelled."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=Path, required=True)
    args = parser.parse_args()
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    args.state_file.write_text(
        json.dumps(
            {
                "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "hostname": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS", ""),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while not stop:
        time.sleep(3600)


if __name__ == "__main__":
    main()
