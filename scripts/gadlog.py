#!/usr/bin/env python
"""Minimal JSON experiment-record helper; no database or service required."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REQUIRED = {"id", "status", "question", "decision_rule", "next_action"}
VALID_STATUS = {
    "planned", "submitted", "running", "passed", "failed", "invalidated", "completed",
    "implementation_failed",
}


def _root(value: str | None) -> Path:
    return Path(value or Path(__file__).resolve().parents[1]).resolve()


def _path(root: Path, experiment_id: str) -> Path:
    return root / "experiments" / f"{experiment_id}.json"


def _read(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _write(path: Path, record: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def validate(root: Path, experiment_id: str) -> int:
    path = _path(root, experiment_id)
    if not path.is_file():
        print(f"missing record: {path}", file=sys.stderr)
        return 2
    try:
        record = _read(path)
    except json.JSONDecodeError as exc:
        print(f"invalid JSON: {exc}", file=sys.stderr)
        return 2
    missing = sorted(REQUIRED - record.keys())
    if missing:
        print(f"missing required keys: {', '.join(missing)}", file=sys.stderr)
        return 2
    if record["id"] != experiment_id:
        print("record id does not match filename", file=sys.stderr)
        return 2
    if record["status"] not in VALID_STATUS:
        print(f"invalid status: {record['status']}", file=sys.stderr)
        return 2
    artifacts = record.get("artifacts", {})
    scratch_dir = artifacts.get("scratch_run_dir")
    if scratch_dir and not str(scratch_dir).startswith("/lustre07/scratch/"):
        print("scratch_run_dir must be under /lustre07/scratch", file=sys.stderr)
        return 2
    print(f"valid: {experiment_id}")
    return 0


def list_records(root: Path) -> int:
    for path in sorted((root / "experiments").glob("*.json")):
        try:
            record = _read(path)
            print(f"{record.get('id', path.stem):40s} {record.get('status', 'invalid')}")
        except json.JSONDecodeError:
            print(f"{path.stem:40s} invalid-json")
    return 0


def attach_job(root: Path, experiment_id: str, job_id: int) -> int:
    if validate(root, experiment_id):
        return 2
    path = _path(root, experiment_id)
    record = _read(path)
    resources = record.setdefault("resources", {})
    jobs = resources.setdefault("slurm_job_ids", [])
    if job_id not in jobs:
        jobs.append(job_id)
    _write(path, record)
    print(f"attached {job_id} to {experiment_id}")
    return 0


def job_status(root: Path, experiment_id: str) -> int:
    if validate(root, experiment_id):
        return 2
    jobs = _read(_path(root, experiment_id)).get("resources", {}).get("slurm_job_ids", [])
    if not jobs:
        print("no Slurm job IDs")
        return 0
    joined = ",".join(str(job) for job in jobs)
    result = subprocess.run(
        ["sacct", "-j", joined, "--format=JobID,State,ExitCode,Elapsed", "-n", "-X"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        print(result.stderr.strip(), file=sys.stderr)
        return result.returncode
    print(result.stdout.rstrip())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    for command in ("validate", "status"):
        sub = subparsers.add_parser(command)
        sub.add_argument("experiment_id")
    attach = subparsers.add_parser("attach-job")
    attach.add_argument("experiment_id")
    attach.add_argument("job_id", type=int)
    args = parser.parse_args()
    root = _root(args.repo_root)
    if args.command == "list":
        return list_records(root)
    if args.command == "validate":
        return validate(root, args.experiment_id)
    if args.command == "attach-job":
        return attach_job(root, args.experiment_id, args.job_id)
    return job_status(root, args.experiment_id)


if __name__ == "__main__":
    raise SystemExit(main())
