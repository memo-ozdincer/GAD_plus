#!/usr/bin/env python
"""Summarize LJ energies for equilibrium sample arrays.

This is a diagnostic, not a formal goodness-of-fit test. The empirical energy
distribution is not expected to be proportional to exp(-beta E) unless the
configuration-space density of states and confinement measure are also known.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LJ_RM_FACTOR = 2.0 ** (1.0 / 6.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute LJ-plus-harmonic-oscillator energy statistics for copied "
            "equilibrium sample arrays."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Input .npy/.npz files. Defaults to data/test_split_LJ*.npy and "
            "data/*.npz when omitted."
        ),
    )
    parser.add_argument("--key", default=None, help="Array key to load from .npz files.")
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=None,
        help="Override atom count. Otherwise inferred from flattened coordinate length.",
    )
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument(
        "--oscillator-scale",
        type=float,
        default=1.0,
        help="Scale k for the harmonic regularizer k * sum_i |r_i - r_com|^2.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--overlap-distance",
        type=float,
        default=0.75,
        help="Report fraction of samples with any pair distance below this value.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Inverse temperature used only for shifted Boltzmann-weight diagnostics.",
    )
    return parser.parse_args()


def default_paths() -> list[Path]:
    paths = sorted((ROOT / "data").glob("test_split_LJ*.npy"))
    paths.extend(sorted((ROOT / "data").glob("*.npz")))
    return paths


def load_array(path: Path, key: str | None) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if key is not None:
                if key not in loaded.files:
                    raise KeyError(f"{path} has keys {loaded.files}, not {key!r}")
                return np.asarray(loaded[key])
            candidates = [
                name
                for name in loaded.files
                if np.asarray(loaded[name]).ndim >= 2
                and np.asarray(loaded[name]).shape[-1] % 3 == 0
            ]
            if len(candidates) != 1:
                raise KeyError(
                    f"{path} needs --key; coordinate-like keys found: {candidates}, "
                    f"all keys: {loaded.files}"
                )
            return np.asarray(loaded[candidates[0]])
    return np.asarray(loaded)


def normalize_coords(array: np.ndarray, n_atoms_override: int | None) -> tuple[np.ndarray, int]:
    arr = np.asarray(array)
    if arr.ndim == 0:
        raise ValueError("coordinate array is scalar")

    if arr.ndim >= 2 and arr.shape[-1] == 3:
        n_atoms = n_atoms_override or int(arr.shape[-2])
        if arr.shape[-2] != n_atoms:
            raise ValueError(f"expected {n_atoms} atoms, got trailing shape {arr.shape[-2:]}")
        return arr.reshape(-1, n_atoms, 3).astype(np.float64, copy=False), n_atoms

    flat_dim = int(arr.shape[-1])
    if n_atoms_override is None:
        if flat_dim % 3 != 0:
            raise ValueError(f"cannot infer atom count from trailing dimension {flat_dim}")
        n_atoms = flat_dim // 3
    else:
        n_atoms = n_atoms_override
        if flat_dim != 3 * n_atoms:
            raise ValueError(f"expected trailing dimension {3 * n_atoms}, got {flat_dim}")

    return arr.reshape(-1, n_atoms, 3).astype(np.float64, copy=False), n_atoms


def finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": float(values.size),
            "finite": 0.0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p01": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    quantiles = np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": float(values.size),
        "finite": float(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(finite.max()),
    }


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def print_summary(name: str, values: np.ndarray) -> None:
    stats = finite_summary(values)
    print(f"\n{name}")
    print(
        "  "
        + " ".join(
            [
                f"count={int(stats['count'])}",
                f"finite={int(stats['finite'])}",
                f"mean={format_float(stats['mean'])}",
                f"std={format_float(stats['std'])}",
                f"min={format_float(stats['min'])}",
                f"p01={format_float(stats['p01'])}",
                f"p05={format_float(stats['p05'])}",
                f"p25={format_float(stats['p25'])}",
                f"p50={format_float(stats['p50'])}",
                f"p75={format_float(stats['p75'])}",
                f"p95={format_float(stats['p95'])}",
                f"p99={format_float(stats['p99'])}",
                f"max={format_float(stats['max'])}",
            ]
        )
    )


def lj_rm(sigma: float) -> float:
    return LJ_RM_FACTOR * sigma


def dissociation_distance_threshold(n_atoms: int, sigma: float) -> float:
    return 2.0 * lj_rm(sigma) * (n_atoms ** (1.0 / 3.0))


def evaluate_samples(
    coords_np: np.ndarray,
    *,
    epsilon: float,
    sigma: float,
    oscillator_scale: float,
    batch_size: int,
) -> dict[str, np.ndarray]:
    n_samples, n_atoms, _ = coords_np.shape
    rm = lj_rm(sigma)

    energy_pair: list[np.ndarray] = []
    energy_harmonic: list[np.ndarray] = []
    pair_min: list[np.ndarray] = []
    pair_mean: list[np.ndarray] = []
    pair_max: list[np.ndarray] = []
    radius_gyration: list[np.ndarray] = []
    radius_max: list[np.ndarray] = []

    upper = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
    for start in range(0, n_samples, batch_size):
        batch_np = coords_np[start : start + batch_size]
        coords = batch_np - batch_np.mean(axis=1, keepdims=True)
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        pairs = np.maximum(distances[:, upper], 1.0e-3)
        rm_over_r_6 = (rm / pairs) ** 6
        energy_pair.append(epsilon * (rm_over_r_6**2 - 2.0 * rm_over_r_6).sum(axis=1))
        energy_harmonic.append(oscillator_scale * np.square(coords).sum(axis=(1, 2)))
        pair_min.append(pairs.min(axis=1))
        pair_mean.append(pairs.mean(axis=1))
        pair_max.append(pairs.max(axis=1))
        radii = np.linalg.norm(coords, axis=-1)
        radius_gyration.append(np.sqrt(np.square(radii).mean(axis=1)))
        radius_max.append(radii.max(axis=1))

    pair = np.concatenate(energy_pair)
    harmonic = np.concatenate(energy_harmonic)
    return {
        "energy_pair_lj": pair,
        "energy_harmonic": harmonic,
        "energy_total": pair + harmonic,
        "pair_min": np.concatenate(pair_min),
        "pair_mean": np.concatenate(pair_mean),
        "pair_max": np.concatenate(pair_max),
        "radius_gyration": np.concatenate(radius_gyration),
        "radius_max": np.concatenate(radius_max),
    }


def boltzmann_weight_summary(energy: np.ndarray, beta: float) -> None:
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        print("\nBoltzmann shifted weights")
        print("  no finite energies")
        return

    shifted_logw = -beta * (finite - finite.min())
    shifted_logw = np.clip(shifted_logw, -745.0, 0.0)
    weights = np.exp(shifted_logw)
    weight_sum = weights.sum()
    probs = weights / weight_sum
    ess = 1.0 / np.square(probs).sum()
    print("\nBoltzmann shifted weights")
    print(
        "  "
        + " ".join(
            [
                f"beta={format_float(beta)}",
                "energy_min_shifted_to=0",
                f"effective_sample_size={format_float(float(ess))}",
                f"ess_fraction={format_float(float(ess / finite.size))}",
                f"max_weight_fraction={format_float(float(probs.max()))}",
            ]
        )
    )


def analyze_path(path: Path, args: argparse.Namespace) -> bool:
    print(f"\n=== {path} ===")
    try:
        array = load_array(path, args.key)
        coords, n_atoms = normalize_coords(array, args.n_atoms)
    except Exception as exc:
        print(f"ERROR: could not load coordinates: {type(exc).__name__}: {exc}")
        return False

    print(
        "loaded "
        f"raw_shape={tuple(array.shape)} dtype={array.dtype} "
        f"normalized_shape={coords.shape} n_atoms={n_atoms}"
    )
    print(
        "energy model "
        f"epsilon={args.epsilon:g} sigma={args.sigma:g} "
        f"rm={lj_rm(args.sigma):g} "
        f"oscillator=linear oscillator_scale={args.oscillator_scale:g}"
    )

    results = evaluate_samples(
        coords,
        epsilon=args.epsilon,
        sigma=args.sigma,
        oscillator_scale=args.oscillator_scale,
        batch_size=args.batch_size,
    )

    for name in [
        "energy_pair_lj",
        "energy_harmonic",
        "energy_total",
        "pair_min",
        "pair_mean",
        "pair_max",
        "radius_gyration",
        "radius_max",
    ]:
        print_summary(name, results[name])

    dissociation_threshold = dissociation_distance_threshold(n_atoms, args.sigma)
    overlap_fraction = float((results["pair_min"] < args.overlap_distance).mean())
    dissociation_fraction = float((results["pair_max"] > dissociation_threshold).mean())
    print("\ngeometry flags")
    print(
        "  "
        + " ".join(
            [
                f"overlap_distance={args.overlap_distance:g}",
                f"overlap_fraction={format_float(overlap_fraction)}",
                f"dissociation_threshold={format_float(dissociation_threshold)}",
                f"dissociation_fraction={format_float(dissociation_fraction)}",
            ]
        )
    )
    boltzmann_weight_summary(results["energy_total"], args.beta)
    return True


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.sigma <= 0:
        raise SystemExit("--sigma must be positive")
    if args.oscillator_scale < 0:
        raise SystemExit("--oscillator-scale must be non-negative")

    paths = args.paths or default_paths()
    if not paths:
        raise SystemExit("No input files found. Pass paths or place files under data/.")

    ok_count = 0
    for path in paths:
        ok_count += int(analyze_path(path, args))
    if ok_count == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
