#!/usr/bin/env python3
"""Matched Transition1x/HIP benchmark for Competitive Soft-Spectral GAD.

The start construction reproduces the historical HIP grid: for each noise
cell, seed one CPU generator once and draw ``randn_like`` sequentially in the
filtered Transition1x test-set order. Shards regenerate that immutable start
table before selecting their assigned sample IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import torch


FROZEN_START_SHA256 = {
    0.10: "c4817f12c171a4deadc4907c11d71b8c3465ab599ec219560d1576f9e9c44a9a",
    0.15: "846ab8ed7d4fa47c86bfbd50209916bb14217c21414960921a5aed7aeefea161",
    0.20: "fdb17613f0d30c8ef6db96def7a06b812762fccaa9615c93907a1c3e393ba156",
}
FROZEN_ASSET_SHA256 = {
    "transition1x_h5": "6a20f8a3f49c50d462270d10d4c44ca102e788072e2096a91d70b5a0f598b629",
    "hip_checkpoint": "154d658f9c5d0b082a9c4893f3978038494d2499794a5ac647448fe397f2d1cb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--h5", default=os.environ.get("GADPLUS_T1X_H5", "data/transition1x.h5"))
    common.add_argument("--checkpoint", default=os.environ.get("GADPLUS_HIP_CHECKPOINT", "models/hip_v2.ckpt"))
    common.add_argument("--split", default="test")
    common.add_argument("--n-samples", type=int, default=287)
    common.add_argument("--sample-ids", default="")
    common.add_argument("--noise", type=float, required=True)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--max-steps", type=int, default=5000)
    common.add_argument("--step-fraction", type=float, default=0.01)
    common.add_argument("--spectral-temperature", type=float, default=0.01)
    common.add_argument("--device", default="cuda")
    common.add_argument("--n-shards", type=int, default=12)
    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument(
        "--resume",
        action="store_true",
        help="reuse already completed calculator-valid sample JSON files",
    )
    aggregate = sub.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--expected-shards", type=int, required=True)
    return parser.parse_args()


def _selected_ids(args: argparse.Namespace, available: int) -> list[int]:
    if args.sample_ids.strip():
        values = [int(item) for item in args.sample_ids.split(",") if item.strip()]
    else:
        values = list(range(min(args.n_samples, available)))
    if not values or min(values) < 0 or max(values) >= available:
        raise ValueError(f"sample IDs must be in [0, {available})")
    return values


def _historical_starts(dataset, noise: float, seed: int) -> list[torch.Tensor]:
    """Reproduce historical sequential float32 CPU noise without leaking RNG state."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        return [sample.pos.clone() + torch.randn_like(sample.pos) * noise for sample in dataset]
    finally:
        torch.random.set_rng_state(state)


def _protocol_payload(args: argparse.Namespace, selected: list[int]) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    assets = {}
    for label, value in (("transition1x_h5", args.h5), ("hip_checkpoint", args.checkpoint)):
        path = Path(value).resolve()
        stat = path.stat()
        assets[label] = {
            "path": str(path),
            "size_bytes": stat.st_size,
            "expected_sha256": FROZEN_ASSET_SHA256[label],
        }
    payload = {
        "schema_version": 1,
        "method": "CS2-GAD",
        "internal_gate_variant": "competitive_subspace",
        "split": args.split,
        "selected_sample_ids": selected,
        "noise_angstrom": args.noise,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "step_fraction": args.step_fraction,
        "spectral_temperature": args.spectral_temperature,
        "force_threshold_eV_per_A": 0.01,
        "force_criterion": "fmax",
        "index_threshold": 1.0e-4,
        "noise_provenance": "sequential float32 CPU torch.randn_like draws in filtered split order",
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "assets": assets,
    }
    if args.split == "test" and args.n_samples == 287 and args.seed == 42:
        payload["start_table_sha256"] = FROZEN_START_SHA256.get(round(args.noise, 2))
    return payload


def _start_table_sha256(starts: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for start in starts:
        digest.update(start.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _ensure_protocol(noise_root: Path, payload: dict[str, Any]) -> str:
    """Create or verify the immutable per-cell protocol and return its digest."""
    noise_root.mkdir(parents=True, exist_ok=True)
    path = noise_root / "protocol.json"
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(canonical)
        os.replace(temporary, path)
    if json.loads(path.read_text()) != payload:
        raise RuntimeError(
            f"protocol mismatch in {path}; use a new output root rather than mixing runs"
        )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _terminal_class(row: dict[str, Any]) -> str:
    if not row.get("calculator_valid"):
        return "calculator_error"
    n_neg = int(row["final_n_neg"])
    if n_neg == 0:
        return "index_zero"
    if n_neg > 1:
        return "multi_negative"
    if float(row["final_fmax"]) >= 0.01:
        return "index_one_force_limited"
    return "strict_ts"


def worker(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_id < args.n_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < n-shards")
    torch.set_num_threads(
        max(1, int(os.environ.get("GADPLUS_WORKER_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    )

    from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    from gadplus.search.intrinsic_gad import IntrinsicGADConfig, run_intrinsic_gad

    dataset = Transition1xDataset(
        args.h5, split=args.split, max_samples=args.n_samples,
        transform=UsePos("pos_transition"),
    )
    starts = _historical_starts(dataset, args.noise, args.seed)
    selected = _selected_ids(args, len(dataset))
    expected_start_digest = _protocol_payload(args, selected).get("start_table_sha256")
    if expected_start_digest is not None:
        observed_start_digest = _start_table_sha256(starts)
        if observed_start_digest != expected_start_digest:
            raise RuntimeError(
                "historical start-table digest mismatch: "
                f"expected {expected_start_digest}, observed {observed_start_digest}"
            )
    noise_root = args.output_root / f"noise_{args.noise:.2f}A"
    protocol_sha256 = _ensure_protocol(noise_root, _protocol_payload(args, selected))
    assigned = [
        sample_id for index, sample_id in enumerate(selected)
        if index % args.n_shards == args.shard_id
    ]
    calculator = load_hip_calculator(args.checkpoint, device=args.device, hessian_method="predict")
    predict_fn = make_hip_predict_fn(calculator)
    config = IntrinsicGADConfig(
        max_steps=args.max_steps,
        force_threshold=0.01,
        force_criterion="fmax",
        index_threshold=1.0e-4,
        spectral_temperature=args.spectral_temperature,
        step_fraction=args.step_fraction,
        gate_variant="competitive_subspace",
        record_history=False,
    )
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    sample_root = noise_root / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    for sample_id in assigned:
        sample_path = sample_root / f"sample_{sample_id:03d}.json"
        if args.resume and sample_path.exists():
            try:
                previous = json.loads(sample_path.read_text())
            except (OSError, json.JSONDecodeError):
                previous = {}
            if (
                previous.get("calculator_valid")
                and previous.get("protocol_sha256") == protocol_sha256
            ):
                rows.append(previous)
                continue
        sample = dataset[sample_id]
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "noise_angstrom": args.noise,
            "seed": args.seed,
            "formula": str(getattr(sample, "formula", "")),
            "rxn": str(getattr(sample, "rxn", "")),
            "method": "CS2-GAD",
            "gate_variant": "competitive_subspace",
            "protocol_sha256": protocol_sha256,
            "calculator_valid": False,
            "strict_ts": False,
            "error": "",
        }
        try:
            coords = starts[sample_id].to(device=device, dtype=torch.float64)
            atomic_nums = sample.z.to(device=device, dtype=torch.long)
            result = run_intrinsic_gad(predict_fn, coords, atomic_nums, config)
            row.update(
                {
                    "calculator_valid": True,
                    "strict_ts": bool(result.final_n_neg == 1 and result.final_force_max < 0.01),
                    "converged": bool(result.converged),
                    "total_steps": int(result.total_steps),
                    "n_evaluations": int(result.n_evaluations),
                    "wall_time_s": float(result.wall_time_s),
                    "final_energy_eV": float(result.final_energy),
                    "final_n_neg": int(result.final_n_neg),
                    "final_fmax": float(result.final_force_max),
                    "final_lambda1": float(result.final_eig0),
                    "final_lambda2": float(result.final_eig1),
                    "final_gate": float(result.final_gate_weight),
                    "failure_type": result.failure_type or "",
                    "final_coords": result.final_coords.tolist(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-sample failures.
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        _write_json_atomic(sample_path, row)
        print(
            f"noise={args.noise:.2f} sample={sample_id:03d} "
            f"valid={row['calculator_valid']} strict={row['strict_ts']} "
            f"steps={row.get('total_steps', 'NA')} "
            f"wall_s={row.get('wall_time_s', 'NA')} error={row['error']!r}",
            flush=True,
        )

    shard_root = noise_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        shard_root / f"shard_{args.shard_id:03d}.json",
        {"shard_id": args.shard_id, "sample_ids": assigned, "rows": rows},
    )


def aggregate(args: argparse.Namespace) -> None:
    noise_root = args.output_root / f"noise_{args.noise:.2f}A"
    selected = _selected_ids(args, args.n_samples)
    protocol_sha256 = _ensure_protocol(noise_root, _protocol_payload(args, selected))
    shard_paths = sorted((noise_root / "shards").glob("shard_*.json"))
    if len(shard_paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(shard_paths)}")
    shard_payloads = [json.loads(path.read_text()) for path in shard_paths]
    shard_ids = [int(payload["shard_id"]) for payload in shard_payloads]
    expected_shard_ids = list(range(args.expected_shards))
    if sorted(shard_ids) != expected_shard_ids:
        raise RuntimeError(
            f"shard IDs must be exactly {expected_shard_ids}, found {sorted(shard_ids)}"
        )
    rows = [
        json.loads(path.read_text())
        for path in sorted((noise_root / "samples").glob("sample_*.json"))
    ]
    expected_ids = set(selected)
    row_ids = [int(row["sample_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("duplicate sample IDs in sample JSON files")
    if set(row_ids) != expected_ids:
        missing = sorted(expected_ids - set(row_ids))
        extra = sorted(set(row_ids) - expected_ids)
        raise RuntimeError(f"sample coverage mismatch: missing={missing}, extra={extra}")
    assigned_ids = [
        int(sample_id)
        for payload in shard_payloads
        for sample_id in payload["sample_ids"]
    ]
    if len(assigned_ids) != len(set(assigned_ids)) or set(assigned_ids) != expected_ids:
        raise RuntimeError("shard manifests do not partition the expected sample IDs exactly once")
    valid = [row for row in rows if row["calculator_valid"]]
    strict = [row for row in valid if row["strict_ts"]]
    mismatched_protocol = [
        int(row["sample_id"])
        for row in rows
        if row.get("protocol_sha256") != protocol_sha256
    ]
    if mismatched_protocol:
        raise RuntimeError(
            f"sample protocol digest mismatch for IDs {mismatched_protocol}"
        )
    planned = len(expected_ids)
    terminal_classes: dict[str, int] = {}
    for row in rows:
        label = _terminal_class(row)
        terminal_classes[label] = terminal_classes.get(label, 0) + 1
    valid_evaluations = [int(row["n_evaluations"]) for row in valid]
    valid_steps = [int(row["total_steps"]) for row in valid]
    valid_wall = [float(row["wall_time_s"]) for row in valid]
    summary = {
        "method": "CS2-GAD",
        "internal_gate_variant": "competitive_subspace",
        "noise_angstrom": args.noise,
        "planned": planned,
        "rows": len(rows),
        "calculator_valid": len(valid),
        "strict_ts": len(strict),
        "strict_rate_planned": len(strict) / planned if planned else math.nan,
        "strict_rate_valid": len(strict) / len(valid) if valid else math.nan,
        "errors": len(rows) - len(valid),
        "terminal_class_counts": terminal_classes,
        "median_evaluations_valid": (
            statistics.median(valid_evaluations) if valid_evaluations else math.nan
        ),
        "median_steps_valid": statistics.median(valid_steps) if valid_steps else math.nan,
        "median_wall_time_s_valid": (
            statistics.median(valid_wall) if valid_wall else math.nan
        ),
        "protocol_sha256": protocol_sha256,
        "configuration": {
            "max_steps": args.max_steps,
            "step_fraction": args.step_fraction,
            "spectral_temperature": args.spectral_temperature,
            "force_threshold": 0.01,
            "index_threshold": 1.0e-4,
            "seed": args.seed,
            "noise_provenance": "historical sequential float32 CPU draws in filtered test order",
        },
    }
    (noise_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (noise_root / "results.csv").open("w", newline="") as handle:
        fields = sorted({key for row in rows for key in row if key != "final_coords"})
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
