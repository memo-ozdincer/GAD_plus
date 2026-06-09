#!/usr/bin/env python
"""Benchmark dxtb GFN1/GFN2 Hessian batches.

dxtb Hessians are AD-based but single-system only, so "batch" timing here means
repeating one molecule batch_size times. CPU can use a process pool; GPU is run
serially by default to avoid multiple worker processes contending for one GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys

import torch

from gadplus.calculator.xtb import load_xtb_calculator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_hessians import System, collect_synthetic_systems


_DXTB_WORKER_CALC = None
_DXTB_WORKER_DEVICE = "cpu"


@dataclass(frozen=True)
class DxtbTimingRow:
    backend: str
    method: str
    device: str
    n_atoms: int
    formula: str
    rxn: str
    batch_size: int
    repeats: int
    warmup: int
    workers: int
    time_s_median: float
    time_s_mean: float
    time_s_min: float
    time_s_max: float
    per_structure_s_median: float
    timing_mode: str
    hessian_shape: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-atom-counts", type=int, nargs="+", default=[25, 30])
    parser.add_argument("--methods", nargs="+", choices=["gfn1", "gfn2"], default=["gfn1", "gfn2"])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/dxtb_hessians"))
    return parser.parse_args()


def _init_dxtb_worker(method: str, device: str) -> None:
    global _DXTB_WORKER_CALC, _DXTB_WORKER_DEVICE

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _DXTB_WORKER_DEVICE = device
    _DXTB_WORKER_CALC = load_xtb_calculator(method=method, device=device)


def _compute_dxtb_hessian_shape(payload: tuple[list[list[float]], list[int]]) -> str:
    if _DXTB_WORKER_CALC is None:
        raise RuntimeError("dxtb worker was not initialized")
    coords_list, atomic_nums_list = payload
    coords = torch.tensor(coords_list, dtype=torch.float64, device=_DXTB_WORKER_DEVICE)
    atomic_nums = torch.tensor(atomic_nums_list, dtype=torch.long, device=_DXTB_WORKER_DEVICE)
    out = _DXTB_WORKER_CALC.compute(coords, atomic_nums, do_hessian=True)
    return "x".join(str(dim) for dim in tuple(out["hessian"].shape))


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
        sync_if_cuda("cuda")
        times.append(time.perf_counter() - t0)
    return times, shape


def summarize(
    *,
    method: str,
    device: str,
    system: System,
    batch_size: int,
    repeats: int,
    warmup: int,
    workers: int,
    times: list[float],
    timing_mode: str,
    hessian_shape: str,
) -> DxtbTimingRow:
    median = statistics.median(times)
    return DxtbTimingRow(
        backend="dxtb",
        method=method,
        device=device,
        n_atoms=system.n_atoms,
        formula=system.formula,
        rxn=system.rxn,
        batch_size=batch_size,
        repeats=repeats,
        warmup=warmup,
        workers=workers,
        time_s_median=median,
        time_s_mean=statistics.mean(times),
        time_s_min=min(times),
        time_s_max=max(times),
        per_structure_s_median=median / batch_size,
        timing_mode=timing_mode,
        hessian_shape=hessian_shape,
    )


def time_serial(
    *,
    method: str,
    device: str,
    system: System,
    batch_size: int,
    repeats: int,
    warmup: int,
) -> DxtbTimingRow:
    calc = load_xtb_calculator(method=method, device=device)
    coords = torch.tensor(system.coords, dtype=torch.float64, device=device)
    atomic_nums = torch.tensor(system.atomic_nums, dtype=torch.long, device=device)

    def run_batch() -> str:
        shape = ""
        for _ in range(batch_size):
            sync_if_cuda(device)
            out = calc.compute(coords, atomic_nums, do_hessian=True)
            sync_if_cuda(device)
            shape = "x".join(str(dim) for dim in tuple(out["hessian"].shape))
        return shape

    times, shape = timed(repeats, warmup, run_batch)
    return summarize(
        method=method,
        device=device,
        system=system,
        batch_size=batch_size,
        repeats=repeats,
        warmup=warmup,
        workers=1,
        times=times,
        timing_mode="serial_repeated_single_system",
        hessian_shape=shape,
    )


def time_cpu_process_pool(
    *,
    method: str,
    system: System,
    batch_size: int,
    repeats: int,
    warmup: int,
    workers: int,
) -> DxtbTimingRow:
    payload = (system.coords.tolist(), system.atomic_nums.astype(int).tolist())

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_dxtb_worker,
        initargs=(method, "cpu"),
    ) as pool:

        def run_batch() -> str:
            shape = ""
            for shape in pool.map(
                _compute_dxtb_hessian_shape,
                [payload] * batch_size,
                chunksize=1,
            ):
                pass
            return shape

        times, shape = timed(repeats, warmup, run_batch)

    return summarize(
        method=method,
        device="cpu",
        system=system,
        batch_size=batch_size,
        repeats=repeats,
        warmup=warmup,
        workers=workers,
        times=times,
        timing_mode="process_pool_repeated_single_system",
        hessian_shape=shape,
    )


def write_outputs(args: argparse.Namespace, rows: list[DxtbTimingRow]) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dxtb_hessian_timing.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    metadata = {
        "created_at": run_id,
        "synthetic_atom_counts": args.synthetic_atom_counts,
        "methods": args.methods,
        "devices": args.devices,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cpu_workers": args.cpu_workers,
    }
    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'metadata.json'}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)

    systems = collect_synthetic_systems(args.synthetic_atom_counts)
    rows: list[DxtbTimingRow] = []
    for system in systems:
        for method in args.methods:
            for device in args.devices:
                if device == "cuda" and not torch.cuda.is_available():
                    print(f"Skipping {method} {system.n_atoms} atoms on cuda: CUDA unavailable")
                    continue
                if device == "cpu" and args.cpu_workers > 1:
                    row = time_cpu_process_pool(
                        method=method,
                        system=system,
                        batch_size=args.batch_size,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        workers=args.cpu_workers,
                    )
                else:
                    row = time_serial(
                        method=method,
                        device=device,
                        system=system,
                        batch_size=args.batch_size,
                        repeats=args.repeats,
                        warmup=args.warmup,
                    )
                print(
                    f"dxtb {method} {device} n={system.n_atoms} batch={args.batch_size}: "
                    f"{row.time_s_median:.3f}s"
                )
                rows.append(row)
    write_outputs(args, rows)


if __name__ == "__main__":
    main()
