#!/usr/bin/env python3
"""Matched all-endpoint HIP IRC_TOPO validation for a CS²-GAD campaign."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--search-root", type=Path, required=True)
    common.add_argument("--h5", required=True)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--noise", type=float, required=True)
    common.add_argument("--n-samples", type=int, default=287)
    common.add_argument("--split", default="test")
    common.add_argument("--irc-steps", type=int, default=500)
    common.add_argument("--rmsd-threshold", type=float, default=0.3)
    common.add_argument("--device", default="cuda")
    common.add_argument("--n-shards", type=int, default=1)
    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--resume", action="store_true")
    aggregate = sub.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--expected-shards", type=int, required=True)
    return parser.parse_args()


def _noise_root(args: argparse.Namespace) -> Path:
    return args.search_root / f"noise_{args.noise:.2f}A"


def _irc_root(args: argparse.Namespace) -> Path:
    return args.search_root / "irc_topo" / f"noise_{args.noise:.2f}A"


def _protocol(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    source_path = _noise_root(args) / "protocol.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    payload = {
        "schema_version": 1,
        "validator": "HIP-Hessian Sella IRC, all terminal endpoints",
        "search_protocol_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "search_runner_sha256": source["script_sha256"],
        "noise_angstrom": args.noise,
        "selected_sample_ids": list(range(args.n_samples)),
        "split": args.split,
        "irc_steps_per_direction": args.irc_steps,
        "rmsd_threshold_angstrom": args.rmsd_threshold,
        "topology": "direction-agnostic element-labelled bond-graph isomorphism",
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets": source["assets"],
    }
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    root = _irc_root(args)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol.json"
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(canonical)
        os.replace(temporary, path)
    if json.loads(path.read_text()) != payload:
        raise RuntimeError(f"IRC protocol mismatch in {path}; use a new search root")
    return payload, digest


def _jsonable_result(result: Any) -> dict[str, Any]:
    payload = dataclasses.asdict(result)
    for key in ("forward_coords", "reverse_coords"):
        value = payload[key]
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
    return payload


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def worker(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_id < args.n_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < n-shards")
    torch.set_num_threads(
        max(1, int(os.environ.get("GADPLUS_WORKER_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    )
    _, protocol_sha256 = _protocol(args)
    source_root = _noise_root(args) / "samples"
    source_paths = [source_root / f"sample_{i:03d}.json" for i in range(args.n_samples)]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} source sample files; first={missing[:3]}")

    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.search.irc_sella_hip import run_irc_sella_hip

    dataset = Transition1xDataset(
        args.h5, split=args.split, max_samples=args.n_samples,
        transform=UsePos("pos_transition"),
    )
    if len(dataset) != args.n_samples:
        raise RuntimeError(f"expected {args.n_samples} dataset samples, found {len(dataset)}")
    calculator = load_hip_calculator(args.checkpoint, device=args.device, hessian_method="predict")
    predict_fn = make_hip_predict_fn(calculator)
    device = torch.device(args.device)
    assigned = [i for i in range(args.n_samples) if i % args.n_shards == args.shard_id]
    sample_root = _irc_root(args) / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id in assigned:
        output_path = sample_root / f"sample_{sample_id:03d}.json"
        if args.resume and output_path.exists():
            try:
                previous = json.loads(output_path.read_text())
            except (OSError, json.JSONDecodeError):
                previous = {}
            if (
                previous.get("protocol_sha256") == protocol_sha256
                and previous.get("irc_valid")
            ):
                rows.append(previous)
                continue
        source = json.loads(source_paths[sample_id].read_text())
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_strict_ts": bool(source.get("strict_ts")),
            "source_calculator_valid": bool(source.get("calculator_valid")),
            "protocol_sha256": protocol_sha256,
            "irc_valid": False,
            "topology_intended": False,
            "error": "",
        }
        started = time.monotonic()
        try:
            if not source.get("calculator_valid") or "final_coords" not in source:
                raise RuntimeError(f"invalid source terminal: {source.get('error', '')}")
            sample = dataset[sample_id]
            ts_coords = torch.as_tensor(
                source["final_coords"], dtype=torch.float32, device=device,
            ).reshape(-1, 3)
            atomic_nums = sample.z.to(device=device, dtype=torch.long)
            reactant = sample.pos_reactant.to(device=device, dtype=torch.float32)
            product = None
            if bool(sample.has_product.item()):
                product = sample.pos_product.to(device=device, dtype=torch.float32)
            result = run_irc_sella_hip(
                ts_coords=ts_coords,
                atomic_nums=atomic_nums,
                predict_fn=predict_fn,
                reactant_coords=reactant,
                product_coords=product,
                rmsd_threshold=args.rmsd_threshold,
                max_steps=args.irc_steps,
                logfile=None,
            )
            result_payload = _jsonable_result(result)
            row.update(result_payload)
            row["irc_valid"] = bool(
                result.forward_coords is not None and result.reverse_coords is not None
            )
            row["topology_intended"] = bool(result.topology_intended)
        except Exception as exc:  # noqa: BLE001 - retain all failed denominators.
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_time_s"] = time.monotonic() - started
        rows.append(row)
        _write_json_atomic(output_path, row)
        print(
            f"IRC noise={args.noise:.2f} sample={sample_id:03d} "
            f"valid={row['irc_valid']} intended={row['topology_intended']} "
            f"source_strict={row['source_strict_ts']} wall_s={row['wall_time_s']:.1f} "
            f"error={row['error']!r}",
            flush=True,
        )
    shard_root = _irc_root(args) / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        shard_root / f"shard_{args.shard_id:03d}.json",
        {"shard_id": args.shard_id, "sample_ids": assigned},
    )


def aggregate(args: argparse.Namespace) -> None:
    _, protocol_sha256 = _protocol(args)
    root = _irc_root(args)
    shard_paths = sorted((root / "shards").glob("shard_*.json"))
    if len(shard_paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(shard_paths)}")
    manifests = [json.loads(path.read_text()) for path in shard_paths]
    if sorted(int(m["shard_id"]) for m in manifests) != list(range(args.expected_shards)):
        raise RuntimeError("IRC shard IDs are incomplete or duplicated")
    assigned = [int(i) for manifest in manifests for i in manifest["sample_ids"]]
    expected = set(range(args.n_samples))
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise RuntimeError("IRC shards do not partition all planned sample IDs exactly once")
    paths = sorted((root / "samples").glob("sample_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    ids = [int(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise RuntimeError("IRC sample coverage is not exactly the planned denominator")
    if any(row.get("protocol_sha256") != protocol_sha256 for row in rows):
        raise RuntimeError("IRC sample protocol digest mismatch")
    valid = [row for row in rows if row["irc_valid"]]
    intended = [row for row in rows if row["topology_intended"]]
    strict = [row for row in rows if row["source_strict_ts"]]
    strict_intended = [row for row in strict if row["topology_intended"]]
    walls = [float(row["wall_time_s"]) for row in rows]
    summary = {
        "method": "CS2-GAD terminal + HIP-Hessian Sella IRC",
        "noise_angstrom": args.noise,
        "planned": args.n_samples,
        "rows": len(rows),
        "irc_valid": len(valid),
        "topology_intended": len(intended),
        "topology_intended_rate_planned": len(intended) / args.n_samples,
        "source_strict_ts": len(strict),
        "strict_and_topology_intended": len(strict_intended),
        "topology_intended_rate_within_strict": (
            len(strict_intended) / len(strict) if strict else math.nan
        ),
        "irc_errors": len(rows) - len(valid),
        "median_wall_time_s": statistics.median(walls) if walls else math.nan,
        "protocol_sha256": protocol_sha256,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "results.csv").open("w", newline="") as handle:
        excluded = {"forward_coords", "reverse_coords"}
        fields = sorted({key for row in rows for key in row if key not in excluded})
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
