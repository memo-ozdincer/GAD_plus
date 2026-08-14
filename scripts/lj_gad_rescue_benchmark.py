#!/usr/bin/env python3
"""Post-hoc GAD-only rescue sweep for the six multi-size LJ union misses.

This is an exploratory recovery experiment, not an amendment to the frozen
1,344-trajectory benchmark.  The parent aggregate determines the six starts;
each receives the same predeclared 32-profile grid (192 trajectories total).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from gadplus.calculator.lennard_jones import lj_atomic_nums, make_lj_predict_fn
from gadplus.core.convergence import force_max
from gadplus.search.intrinsic_gad import IntrinsicGADConfig, run_intrinsic_gad
from lj_intrinsic_noise_sweep import _downhill_endpoints
from lj_multisize_cs2_benchmark import _make_start, _spectrum


PARENT_RESULTS_SHA256 = "5ae8b48846bc6b4fdeaf7b7408d574c69bb4e08051fc6d19912174ff275da809"
PARENT_PROTOCOL_SHA256 = "07863dad05285f7fb99a5a5c31ebab6beb2eb3b314cab316aab3c9a766b660d4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument(
        "--parent-root",
        type=Path,
        default=Path("/scratch/memoozd/gadplus/runs/lj-multisize-cs2-2076554"),
    )
    sub.add_parser("prepare", parents=[common])
    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--task-id", type=int, required=True)
    aggregate = sub.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--expected-tasks", type=int, default=192)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _start_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["size"]),
        str(row["basin"]),
        str(row["start_family"]),
        float(row["level_sigma"]),
        int(row["sample_id"]),
        int(row["seed"]),
    )


def _profiles() -> list[dict[str, Any]]:
    profiles = []
    gates = {
        "intrinsic_lambda2": ("lambda2", (0.01, 0.025, 0.05, 0.10)),
        "cs2": ("competitive_subspace", (0.0025, 0.005, 0.01, 0.02)),
    }
    for method, (gate, etas) in gates.items():
        for eta, tau_s, max_steps in itertools.product(
            etas, (0.005, 0.01), (2000, 5000)
        ):
            profiles.append(
                {
                    "method": method,
                    "gate_variant": gate,
                    "step_fraction": eta,
                    "spectral_temperature": tau_s,
                    "max_steps": max_steps,
                }
            )
    assert len(profiles) == 32
    return profiles


def _load_parent(parent_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol_path = parent_root / "protocol.json"
    results_path = parent_root / "all_results.json"
    if _sha256(protocol_path) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("parent protocol SHA256 mismatch")
    if _sha256(results_path) != PARENT_RESULTS_SHA256:
        raise RuntimeError("parent aggregate SHA256 mismatch")
    return json.loads(protocol_path.read_text()), json.loads(results_path.read_text())


def _misses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["method"] in {"intrinsic_lambda2", "cs2"}:
            paired[_start_key(row)][row["method"]] = row
    misses = []
    for key, methods in sorted(paired.items()):
        if set(methods) != {"intrinsic_lambda2", "cs2"}:
            raise RuntimeError(f"incomplete parent GAD pair for {key}")
        if not any(bool(row["strict_ts"]) for row in methods.values()):
            baseline = methods["intrinsic_lambda2"]
            misses.append(
                {
                    "size": key[0],
                    "basin": key[1],
                    "start_family": key[2],
                    "level_sigma": key[3],
                    "sample_id": key[4],
                    "seed": key[5],
                    "initial_energy": baseline["initial_energy"],
                    "initial_n_neg": baseline["initial_n_neg"],
                    "initial_fmax": baseline["initial_fmax"],
                    "parent_terminal": {
                        name: {
                            "final_n_neg": row["final_n_neg"],
                            "final_fmax": row["final_fmax"],
                            "steps": row["steps"],
                        }
                        for name, row in sorted(methods.items())
                    },
                }
            )
    if len(misses) != 6:
        raise RuntimeError(f"expected exactly six parent union misses, found {len(misses)}")
    return misses


def _payload(parent_root: Path, misses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "classification": "post-hoc exploratory GAD-only recovery sweep",
        "parent_root": str(parent_root.resolve()),
        "parent_results_sha256": PARENT_RESULTS_SHA256,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "strict_gate": {"n_neg": 1, "fmax": 0.01, "index_threshold": 1.0e-4},
        "endpoint_gate": {"n_neg": 0, "fmax": 1.0e-5},
        "misses": misses,
        "profiles": _profiles(),
        "planned_tasks": len(misses) * len(_profiles()),
    }


def _ensure_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent_protocol, rows = _load_parent(args.parent_root)
    misses = _misses(rows)
    payload = _payload(args.parent_root, misses)
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "protocol.json"
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != canonical:
        raise RuntimeError(f"protocol mismatch in {path}; use a new output root")
    if not path.exists():
        path.write_text(canonical)
    return parent_protocol, misses


def prepare(args: argparse.Namespace) -> None:
    _, misses = _ensure_protocol(args)
    print(json.dumps({"misses": misses, "profiles": _profiles()}, indent=2), flush=True)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    parent_protocol, misses = _ensure_protocol(args)
    profiles = _profiles()
    planned = len(misses) * len(profiles)
    if not 0 <= args.task_id < planned:
        raise ValueError(f"task-id must be in [0, {planned})")
    miss_index, profile_index = divmod(args.task_id, len(profiles))
    miss, profile = misses[miss_index], profiles[profile_index]
    row: dict[str, Any] = {
        "task_id": args.task_id,
        "miss_index": miss_index,
        "profile_index": profile_index,
        **{key: miss[key] for key in (
            "size", "basin", "start_family", "level_sigma", "sample_id", "seed"
        )},
        **profile,
        "calculator_valid": False,
        "strict_ts": False,
        "endpoint_minima": False,
        "error": "",
    }
    try:
        reference = next(
            item for item in parent_protocol["basins"]
            if item["size"] == miss["size"] and item["basin"] == miss["basin"]
        )
        minimum = torch.tensor(reference["coords"], dtype=torch.float64)
        atomic_nums = lj_atomic_nums(miss["size"])
        predictor = make_lj_predict_fn()
        _, _, modes = _spectrum(predictor, minimum, atomic_nums)
        start, mode_index = _make_start(
            minimum,
            modes,
            atomic_nums,
            miss["start_family"],
            miss["level_sigma"],
            miss["sample_id"],
            miss["seed"],
        )
        initial_output, initial_eigenvalues, _ = _spectrum(predictor, start, atomic_nums)
        observed = {
            "initial_energy": float(initial_output["energy"].item()),
            "initial_n_neg": int((initial_eigenvalues < -1.0e-4).sum().item()),
            "initial_fmax": force_max(initial_output["forces"]),
        }
        if any(observed[key] != miss[key] for key in observed):
            raise RuntimeError(f"parent start mismatch: expected={miss}, observed={observed}")
        config = IntrinsicGADConfig(
            max_steps=profile["max_steps"],
            spectral_temperature=profile["spectral_temperature"],
            step_fraction=profile["step_fraction"],
            gate_variant=profile["gate_variant"],
            force_threshold=0.01,
            force_criterion="fmax",
            record_history=False,
        )
        result = run_intrinsic_gad(predictor, start, atomic_nums, config)
        final_coords = result.final_coords.to(torch.float64)
        final_output, final_eigenvalues, _ = _spectrum(predictor, final_coords, atomic_nums)
        final_n_neg = int((final_eigenvalues < -1.0e-4).sum().item())
        final_fmax = force_max(final_output["forces"])
        strict = final_n_neg == 1 and final_fmax < 0.01
        endpoints = (
            _downhill_endpoints(predictor, atomic_nums, final_coords, displacement=0.03)
            if strict else []
        )
        endpoint_minima = len(endpoints) == 2 and all(
            endpoint["n_neg"] == 0 and endpoint["force_max"] < 1.0e-5
            for endpoint in endpoints
        )
        row.update(
            {
                "calculator_valid": True,
                "mode_index": mode_index,
                **observed,
                "strict_ts": strict,
                "endpoint_minima": endpoint_minima,
                "steps": int(result.total_steps),
                "evaluations": int(result.n_evaluations),
                "wall_time_s": float(result.wall_time_s),
                "final_energy": float(final_output["energy"].item()),
                "final_n_neg": final_n_neg,
                "final_fmax": final_fmax,
                "final_lambda1": float(final_eigenvalues[0].item()),
                "final_lambda2": float(final_eigenvalues[1].item()),
                "endpoint_energies": [float(endpoint["energy"]) for endpoint in endpoints],
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve every task outcome.
        row["error"] = f"{type(exc).__name__}: {exc}"
    task_root = args.output_root / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(task_root / f"task_{args.task_id:03d}.json", row)


def aggregate(args: argparse.Namespace) -> None:
    _, misses = _ensure_protocol(args)
    paths = sorted((args.output_root / "tasks").glob("task_*.json"))
    if len(paths) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} tasks, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    if {int(row["task_id"]) for row in rows} != set(range(args.expected_tasks)):
        raise RuntimeError("task IDs do not cover the declared grid exactly")
    if any(not row.get("calculator_valid") for row in rows):
        failed = [row["task_id"] for row in rows if not row.get("calculator_valid")]
        raise RuntimeError(f"calculator-invalid rescue tasks: {failed}")
    by_miss: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_profile: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_miss[int(row["miss_index"])].append(row)
        by_profile[int(row["profile_index"])].append(row)
    miss_summary = []
    for index, miss in enumerate(misses):
        group = by_miss[index]
        strict = [row for row in group if row["strict_ts"]]
        endpoints = [row for row in group if row["endpoint_minima"]]
        miss_summary.append(
            {
                **{key: miss[key] for key in (
                    "size", "basin", "start_family", "level_sigma", "sample_id", "seed"
                )},
                "profiles": len(group),
                "strict_profiles": len(strict),
                "endpoint_profiles": len(endpoints),
                "best_strict_evaluations": min(
                    (row["evaluations"] for row in strict), default=None
                ),
                "rescued": bool(strict),
            }
        )
    profile_summary = []
    for index, profile in enumerate(_profiles()):
        group = by_profile[index]
        profile_summary.append(
            {
                "profile_index": index,
                **profile,
                "strict": sum(bool(row["strict_ts"]) for row in group),
                "endpoint_minima": sum(bool(row["endpoint_minima"]) for row in group),
                "planned": len(group),
            }
        )
    payload = {
        "classification": "post-hoc exploratory; not parent test evidence",
        "planned": args.expected_tasks,
        "calculator_valid": len(rows),
        "parent_gad_union_strict": 442,
        "parent_unique_starts": 448,
        "misses_rescued_by_any_profile": sum(item["rescued"] for item in miss_summary),
        "exploratory_union_strict": 442 + sum(item["rescued"] for item in miss_summary),
        "misses": miss_summary,
        "profiles": profile_summary,
    }
    _atomic_json(args.output_root / "all_results.json", rows)
    _atomic_json(args.output_root / "summary.json", payload)
    lines = [
        "# GAD-only LJ rescue sweep",
        "",
        "Post-hoc exploratory recovery only; the frozen parent result remains 442/448 for the paired GAD union.",
        "",
        f"Any-profile recovery: {payload['misses_rescued_by_any_profile']}/6; exploratory union: {payload['exploratory_union_strict']}/448.",
        "",
        "| LJ | Basin | Start | Level | Sample | Strict profiles | Endpoint profiles | Best evals |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in miss_summary:
        lines.append(
            f"| {item['size']} | {item['basin']} | {item['start_family']} | "
            f"{item['level_sigma']:.2f} | {item['sample_id']} | "
            f"{item['strict_profiles']}/{item['profiles']} | "
            f"{item['endpoint_profiles']}/{item['profiles']} | "
            f"{item['best_strict_evaluations'] or 'n/a'} |"
        )
    (args.output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
