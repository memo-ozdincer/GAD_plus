#!/usr/bin/env python
"""Benchmark SCINE DFTB0 and HIP Hessian timings by molecule size.

The HIP backend supports true PyG batching. SCINE Sparrow is a single-geometry
calculator, so this script can either project serial batch times or measure
serial/process-parallel batches of independent Sparrow Hessian calls.
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
from transition1x import Dataloader

from gadplus.calculator.hip import load_hip_calculator
from gadplus.calculator.scine import load_scine_calculator
from gadplus.paths import hip_checkpoint_path, transition1x_h5_path

_SCINE_WORKER_CALC = None


@dataclass(frozen=True)
class System:
    n_atoms: int
    formula: str
    rxn: str
    coords: np.ndarray
    atomic_nums: np.ndarray


@dataclass(frozen=True)
class TimingRow:
    backend: str
    n_atoms: int
    batch_size: int
    formula: str
    rxn: str
    repeats: int
    warmup: int
    time_s_median: float
    time_s_mean: float
    time_s_min: float
    time_s_max: float
    per_structure_s_median: float
    timing_mode: str
    hessian_shape: str
    scine_workers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=transition1x_h5_path())
    parser.add_argument("--checkpoint", type=Path, default=hip_checkpoint_path())
    parser.add_argument("--split", default="train", choices=["data", "train", "val", "test"])
    parser.add_argument("--atom-min", type=int, default=5)
    parser.add_argument("--atom-max", type=int, default=30)
    parser.add_argument(
        "--synthetic-atom-counts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Use deterministic synthetic organic geometries for these atom counts "
            "instead of sampling Transition1x. Currently supports 25 and 30."
        ),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/hessians"))
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["scine", "hip"],
        default=["scine", "hip"],
        help="Calculator backends to benchmark.",
    )
    parser.add_argument(
        "--scine-workers",
        type=int,
        default=min(30, int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 16))),
        help="Number of worker processes for measured SCINE batches.",
    )
    parser.add_argument(
        "--scine-large-batches",
        choices=["project", "measure"],
        default="measure",
        help=(
            "For batch sizes >1, either project serial SCINE time from batch size 1 "
            "or actually measure a serial/process-parallel loop."
        ),
    )
    return parser.parse_args()


def _synthetic_c7h16o2() -> System:
    """Loose heptanediol-like C7H16O2 geometry with 25 atoms."""
    coords: list[list[float]] = []
    atomic_nums: list[int] = []

    carbon_positions = [
        [1.54 * i, 0.35 * ((-1) ** i), 0.0]
        for i in range(7)
    ]
    oxygen_positions = [
        [carbon_positions[0][0] - 1.43, carbon_positions[0][1], 0.0],
        [carbon_positions[-1][0] + 1.43, carbon_positions[-1][1], 0.0],
    ]

    for pos in carbon_positions:
        coords.append(pos)
        atomic_nums.append(6)
    for pos in oxygen_positions:
        coords.append(pos)
        atomic_nums.append(8)

    for i, pos in enumerate(carbon_positions):
        side = -1.0 if i % 2 else 1.0
        coords.append([pos[0], pos[1] + side * 0.95, 0.85])
        coords.append([pos[0], pos[1] + side * 0.95, -0.85])
        atomic_nums.extend([1, 1])

    coords.append([oxygen_positions[0][0] - 0.75, oxygen_positions[0][1], 0.55])
    coords.append([oxygen_positions[1][0] + 0.75, oxygen_positions[1][1], 0.55])
    atomic_nums.extend([1, 1])

    arr = np.asarray(coords, dtype=np.float64)
    arr = arr - arr.mean(axis=0, keepdims=True)
    return System(
        n_atoms=25,
        formula="C7H16O2",
        rxn="synthetic_heptanediol",
        coords=arr,
        atomic_nums=np.asarray(atomic_nums, dtype=np.int64),
    )


def _synthetic_c10h20() -> System:
    """Loose cyclodecane-like C10H20 geometry with 30 atoms."""
    coords: list[list[float]] = []
    atomic_nums: list[int] = []
    n_carbons = 10
    radius = 1.54 / (2.0 * np.sin(np.pi / n_carbons))

    carbon_positions = []
    for i in range(n_carbons):
        theta = 2.0 * np.pi * i / n_carbons
        carbon_positions.append(
            [radius * np.cos(theta), radius * np.sin(theta), 0.25 * ((-1) ** i)]
        )

    for pos in carbon_positions:
        coords.append(pos)
        atomic_nums.append(6)

    for i, pos in enumerate(carbon_positions):
        xy = np.asarray(pos[:2], dtype=np.float64)
        radial = xy / max(np.linalg.norm(xy), 1.0e-12)
        coords.append([pos[0] + 1.05 * radial[0], pos[1] + 1.05 * radial[1], pos[2] + 0.65])
        coords.append([pos[0] + 1.05 * radial[0], pos[1] + 1.05 * radial[1], pos[2] - 0.65])
        atomic_nums.extend([1, 1])

    arr = np.asarray(coords, dtype=np.float64)
    arr = arr - arr.mean(axis=0, keepdims=True)
    return System(
        n_atoms=30,
        formula="C10H20",
        rxn="synthetic_cyclodecane",
        coords=arr,
        atomic_nums=np.asarray(atomic_nums, dtype=np.int64),
    )


def collect_synthetic_systems(atom_counts: list[int]) -> list[System]:
    builders = {
        25: _synthetic_c7h16o2,
        30: _synthetic_c10h20,
    }
    systems = []
    for atom_count in atom_counts:
        if atom_count not in builders:
            raise ValueError(
                f"No synthetic geometry is defined for {atom_count} atoms. "
                f"Supported: {sorted(builders)}"
            )
        systems.append(builders[atom_count]())
    return systems


def _init_scine_worker(functional: str) -> None:
    """Initialize one Sparrow calculator per process."""
    global _SCINE_WORKER_CALC

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _SCINE_WORKER_CALC = load_scine_calculator(functional=functional, device="cpu")


def _compute_scine_hessian_shape(payload: tuple[list[list[float]], list[int]]) -> str:
    """Compute one SCINE Hessian in a worker and return its shape string."""
    if _SCINE_WORKER_CALC is None:
        _init_scine_worker("DFTB0")
    coords_list, atomic_nums_list = payload
    coords = torch.tensor(coords_list, dtype=torch.float64)
    atomic_nums = torch.tensor(atomic_nums_list, dtype=torch.long)
    out = _SCINE_WORKER_CALC.compute(coords, atomic_nums, do_hessian=True)
    return "x".join(str(dim) for dim in tuple(out["hessian"].shape))


def collect_systems(args: argparse.Namespace) -> list[System]:
    wanted = set(range(args.atom_min, args.atom_max + 1))
    systems: dict[int, System] = {}

    for idx, item in enumerate(Dataloader(str(args.h5), datasplit=args.split, only_final=True)):
        ts = item["transition_state"]
        z = np.asarray(ts["atomic_numbers"], dtype=np.int64)
        n_atoms = int(z.shape[0])
        if n_atoms in wanted and n_atoms not in systems:
            systems[n_atoms] = System(
                n_atoms=n_atoms,
                formula=str(ts.get("formula", item.get("formula", ""))),
                rxn=str(ts.get("rxn", item.get("rxn", ""))),
                coords=np.asarray(ts["positions"], dtype=np.float64).reshape(n_atoms, 3),
                atomic_nums=z,
            )
        if wanted.issubset(systems) or idx + 1 >= args.max_scan:
            break

    missing = sorted(wanted.difference(systems))
    if missing:
        print(f"WARNING: no {args.split} transition-state system found for atom counts: {missing}")
    return [systems[n_atoms] for n_atoms in sorted(systems)]


def make_hip_batch(system: System, batch_size: int) -> Batch:
    data_list = []
    for _ in range(batch_size):
        data_list.append(
            TGData(
                pos=torch.as_tensor(system.coords, dtype=torch.float32),
                z=torch.as_tensor(system.atomic_nums, dtype=torch.int64),
                charges=torch.as_tensor(system.atomic_nums, dtype=torch.int64),
                natoms=torch.tensor([system.n_atoms], dtype=torch.int64),
                cell=None,
                pbc=torch.tensor(False, dtype=torch.bool),
            )
        )
    return Batch.from_data_list(data_list)


def sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(repeats: int, warmup: int, fn) -> tuple[list[float], str]:
    shape = ""
    for _ in range(warmup):
        shape = fn()

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        shape = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return times, shape


def summarize(
    *,
    backend: str,
    system: System,
    batch_size: int,
    repeats: int,
    warmup: int,
    times: Iterable[float],
    timing_mode: str,
    hessian_shape: str,
) -> TimingRow:
    times = list(times)
    median = statistics.median(times)
    return TimingRow(
        backend=backend,
        n_atoms=system.n_atoms,
        batch_size=batch_size,
        formula=system.formula,
        rxn=system.rxn,
        repeats=repeats,
        warmup=warmup,
        time_s_median=median,
        time_s_mean=statistics.mean(times),
        time_s_min=min(times),
        time_s_max=max(times),
        per_structure_s_median=median / batch_size,
        timing_mode=timing_mode,
        hessian_shape=hessian_shape,
        scine_workers=0 if backend == "hip" else 1,
    )


def time_hip(args: argparse.Namespace, systems: list[System]) -> list[TimingRow]:
    calc = load_hip_calculator(str(args.checkpoint), device=args.device, hessian_method="predict")
    rows: list[TimingRow] = []

    for system in systems:
        for batch_size in args.batch_sizes:
            batch = make_hip_batch(system, batch_size)

            def run_once() -> str:
                sync_if_cuda(args.device)
                out = calc.predict(batch=batch, do_hessian=True)
                sync_if_cuda(args.device)
                return "x".join(str(dim) for dim in tuple(out["hessian"].shape))

            times, shape = timed(args.repeats, args.warmup, run_once)
            row = summarize(
                backend="hip",
                system=system,
                batch_size=batch_size,
                repeats=args.repeats,
                warmup=args.warmup,
                times=times,
                timing_mode="measured_true_batch",
                hessian_shape=shape,
            )
            print(
                f"HIP n={system.n_atoms:2d} batch={batch_size:3d}: "
                f"{row.time_s_median:.4f}s median"
            )
            rows.append(row)
    return rows


def time_scine(args: argparse.Namespace, systems: list[System]) -> list[TimingRow]:
    calc = load_scine_calculator(functional="DFTB0", device="cpu")
    rows: list[TimingRow] = []

    pool = None
    if args.scine_large_batches == "measure" and args.scine_workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=args.scine_workers,
            initializer=_init_scine_worker,
            initargs=("DFTB0",),
        )

    try:
        for system in systems:
            coords = torch.as_tensor(system.coords, dtype=torch.float64)
            atomic_nums = torch.as_tensor(system.atomic_nums, dtype=torch.long)
            payload = (system.coords.tolist(), system.atomic_nums.astype(int).tolist())

            def run_single() -> str:
                out = calc.compute(coords, atomic_nums, do_hessian=True)
                return "x".join(str(dim) for dim in tuple(out["hessian"].shape))

            single_times, shape = timed(args.repeats, args.warmup, run_single)
            single_row = summarize(
                backend="scine_dftb0",
                system=system,
                batch_size=1,
                repeats=args.repeats,
                warmup=args.warmup,
                times=single_times,
                timing_mode="measured_single_geometry",
                hessian_shape=shape,
            )
            object.__setattr__(single_row, "scine_workers", 1)
            print(f"SCINE n={system.n_atoms:2d} batch=  1: {single_row.time_s_median:.4f}s median")
            rows.append(single_row)

            for batch_size in args.batch_sizes:
                if batch_size == 1:
                    continue
                if args.scine_large_batches == "measure":

                    if pool is None:

                        def run_loop() -> str:
                            loop_shape = ""
                            for _ in range(batch_size):
                                loop_shape = run_single()
                            return loop_shape

                        times, loop_shape = timed(args.repeats, args.warmup, run_loop)
                        timing_mode = "measured_serial_loop"
                        scine_workers = 1
                    else:

                        def run_parallel_batch() -> str:
                            loop_shape = ""
                            for loop_shape in pool.map(
                                _compute_scine_hessian_shape,
                                [payload] * batch_size,
                                chunksize=1,
                            ):
                                pass
                            return loop_shape

                        times, loop_shape = timed(args.repeats, args.warmup, run_parallel_batch)
                        timing_mode = "measured_process_pool"
                        scine_workers = args.scine_workers
                else:
                    times = [t * batch_size for t in single_times]
                    loop_shape = shape
                    timing_mode = "projected_serial_from_single"
                    scine_workers = 1

                row = summarize(
                    backend="scine_dftb0",
                    system=system,
                    batch_size=batch_size,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    times=times,
                    timing_mode=timing_mode,
                    hessian_shape=loop_shape,
                )
                object.__setattr__(row, "scine_workers", scine_workers)
                print(
                    f"SCINE n={system.n_atoms:2d} batch={batch_size:3d}: "
                    f"{row.time_s_median:.4f}s median ({timing_mode}, workers={scine_workers})"
                )
                rows.append(row)
    finally:
        if pool is not None:
            pool.shutdown()
    return rows


def write_outputs(args: argparse.Namespace, rows: list[TimingRow], systems: list[System]) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "hessian_timing.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    metadata = {
        "created_at": run_id,
        "h5": str(args.h5),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "atom_range": [args.atom_min, args.atom_max],
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "device": args.device,
        "backends": args.backends,
        "scine_large_batches": args.scine_large_batches,
        "scine_workers": args.scine_workers,
        "systems": [
            {
                "n_atoms": s.n_atoms,
                "formula": s.formula,
                "rxn": s.rxn,
            }
            for s in systems
        ],
    }
    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'metadata.json'}")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if args.synthetic_atom_counts:
        systems = collect_synthetic_systems(args.synthetic_atom_counts)
    else:
        systems = collect_systems(args)
    if not systems:
        raise RuntimeError("No systems found for requested atom-count range.")
    print(f"Benchmarking atom counts: {[system.n_atoms for system in systems]}")

    rows = []
    if "scine" in args.backends:
        rows.extend(time_scine(args, systems))
    if "hip" in args.backends:
        rows.extend(time_hip(args, systems))
    write_outputs(args, rows, systems)


if __name__ == "__main__":
    main()
