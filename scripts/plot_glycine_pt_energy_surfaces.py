#!/usr/bin/env python
"""Plot HIP and ORCA 2D energy surfaces for the glycine proton-transfer scan."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EV_TO_KCALMOL = 23.060548867


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("runs/glycine_pt_scan"),
        help="Directory containing HIP glycine proton-transfer scan outputs.",
    )
    parser.add_argument(
        "--orca-dir",
        type=Path,
        default=Path("orca_wb97x_631gd_glycine_pt_nh_oh_scan_80"),
        help="Directory containing ORCA package outputs and metadata.csv.",
    )
    parser.add_argument(
        "--hip-energies",
        type=Path,
        default=None,
        help="HIP energies table. Defaults to scan-dir/hip_v2_energies.csv.",
    )
    parser.add_argument(
        "--orca-energies",
        type=Path,
        default=None,
        help=(
            "ORCA energies table. Defaults to orca-dir/metadata.csv; "
            "scan-dir/orca_energies.csv is also supported."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--energy-contour-step",
        type=float,
        default=10.0,
        help="Contour spacing for relative energies in kcal/mol.",
    )
    parser.add_argument(
        "--energy-vmax",
        type=float,
        default=None,
        help="Optional upper color limit for relative energy plots in kcal/mol.",
    )
    parser.add_argument(
        "--linecut-q-oh",
        type=float,
        nargs="*",
        default=(1.15, 1.75, 2.15),
        help="q_oh values to use for q_nh line cuts.",
    )
    parser.add_argument(
        "--linecut-q-nh",
        type=float,
        nargs="*",
        default=(1.0, 1.65, 2.3),
        help="q_nh values to use for q_oh line cuts.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_hip(path: Path) -> pd.DataFrame:
    df = read_table(path)
    require_columns(df, {"grid_id", "q_nh", "q_oh", "hip_v2_energy"}, str(path))
    out = df[["grid_id", "q_nh", "q_oh", "hip_v2_energy"]].copy()
    out["grid_id"] = out["grid_id"].astype(int)

    if "hip_v2_energy_relative" in df.columns:
        out["hip_relative_kcalmol"] = df["hip_v2_energy_relative"].astype(float) * EV_TO_KCALMOL
    else:
        energy_ev = out["hip_v2_energy"].astype(float)
        out["hip_relative_kcalmol"] = (energy_ev - energy_ev.min()) * EV_TO_KCALMOL

    return out.sort_values("grid_id").reset_index(drop=True)


def load_orca(path: Path) -> pd.DataFrame:
    df = read_table(path)
    if {"q_nh_angstrom", "q_oh_angstrom"}.issubset(df.columns):
        q_nh_col = "q_nh_angstrom"
        q_oh_col = "q_oh_angstrom"
    else:
        q_nh_col = "q_nh"
        q_oh_col = "q_oh"

    grid_col = "grid_id" if "grid_id" in df.columns else "job_id"
    require_columns(df, {grid_col, q_nh_col, q_oh_col, "energy_relative_kcalmol"}, str(path))
    out = df[[grid_col, q_nh_col, q_oh_col, "energy_relative_kcalmol"]].copy()
    out = out.rename(
        columns={
            grid_col: "grid_id",
            q_nh_col: "q_nh",
            q_oh_col: "q_oh",
            "energy_relative_kcalmol": "orca_relative_kcalmol",
        }
    )
    out["grid_id"] = out["grid_id"].astype(str).str.replace("grid_", "", regex=False).astype(int)
    return out.sort_values("grid_id").reset_index(drop=True)


def merge_surfaces(hip: pd.DataFrame, orca: pd.DataFrame) -> pd.DataFrame:
    merged = orca.merge(hip, on="grid_id", suffixes=("_orca", "_hip"), validate="one_to_one")
    if not np.allclose(merged["q_nh_orca"], merged["q_nh_hip"]):
        raise ValueError("HIP and ORCA q_nh values do not match by grid_id")
    if not np.allclose(merged["q_oh_orca"], merged["q_oh_hip"]):
        raise ValueError("HIP and ORCA q_oh values do not match by grid_id")

    merged["q_nh"] = merged["q_nh_orca"]
    merged["q_oh"] = merged["q_oh_orca"]
    merged["hip_minus_orca_kcalmol"] = (
        merged["hip_relative_kcalmol"] - merged["orca_relative_kcalmol"]
    )
    return merged[
        [
            "grid_id",
            "q_nh",
            "q_oh",
            "orca_relative_kcalmol",
            "hip_relative_kcalmol",
            "hip_minus_orca_kcalmol",
            "hip_v2_energy",
        ]
    ].sort_values("grid_id")


def as_grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    pivot = (
        df.pivot(index="q_oh", columns="q_nh", values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    return x, y, z


def contour_levels(values: np.ndarray, step: float, vmax: float | None = None) -> np.ndarray:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return np.array([0.0, step])
    lo = np.floor(finite.min() / step) * step
    hi_raw = finite.max() if vmax is None else min(float(vmax), finite.max())
    hi = np.ceil(hi_raw / step) * step
    if hi <= lo:
        hi = lo + step
    return np.arange(lo, hi + 0.5 * step, step)


def add_min_marker(ax: plt.Axes, df: pd.DataFrame, value_col: str, label: str) -> None:
    row = df.loc[df[value_col].idxmin()]
    ax.plot(row["q_nh"], row["q_oh"], marker="*", color="white", markeredgecolor="black", ms=12)
    ax.text(
        row["q_nh"] + 0.025,
        row["q_oh"] + 0.025,
        f"{label} min\n{int(row['grid_id'])}",
        color="white",
        fontsize=8,
        weight="bold",
        path_effects=[],
    )


def plot_surface(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    cmap: str,
    contour_step: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    contour_source: str | None = None,
    contour_color: str = "k",
) -> None:
    x, y, z = as_grid(df, value_col)
    if contour_step is None:
        mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        levels = contour_levels(z.compressed(), contour_step, vmax=vmax)
        mesh = ax.contourf(x, y, z, levels=levels, cmap=cmap, extend="max")

    if contour_source is not None:
        _, _, contour_z = as_grid(df, contour_source)
        levels = contour_levels(contour_z.compressed(), contour_step or 10.0)
        ax.contour(x, y, contour_z, levels=levels, colors=contour_color, linewidths=0.65, alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel(r"$q_\mathrm{NH}=d(\mathrm{N4,H9})$ [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}=d(\mathrm{O3,H9})$ [$\AA$]")
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)


def save_surface_comparison(df: pd.DataFrame, output_dir: Path, dpi: int, step: float, vmax: float | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7), constrained_layout=True)
    plot_surface(
        axes[0],
        df,
        "orca_relative_kcalmol",
        "ORCA wB97X/6-31G(d)",
        r"relative energy [kcal mol$^{-1}$]",
        "turbo",
        contour_step=step,
        vmax=vmax,
    )
    add_min_marker(axes[0], df, "orca_relative_kcalmol", "ORCA")
    plot_surface(
        axes[1],
        df,
        "hip_relative_kcalmol",
        "HIP v2",
        r"relative energy [kcal mol$^{-1}$]",
        "turbo",
        contour_step=step,
        vmax=vmax,
        contour_source="orca_relative_kcalmol",
    )
    add_min_marker(axes[1], df, "hip_relative_kcalmol", "HIP")

    abs_err = float(np.nanmax(np.abs(df["hip_minus_orca_kcalmol"])))
    err_lim = max(abs_err, 1.0)
    plot_surface(
        axes[2],
        df,
        "hip_minus_orca_kcalmol",
        "HIP - ORCA",
        r"relative energy error [kcal mol$^{-1}$]",
        "coolwarm",
        contour_step=None,
        vmin=-err_lim,
        vmax=err_lim,
        contour_source="orca_relative_kcalmol",
    )
    fig.suptitle("Glycine intramolecular proton-transfer scan, Transition1x test sample 5")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"glycine_pt_energy_surfaces.{suffix}", dpi=dpi)
    plt.close(fig)


def save_overlay(df: pd.DataFrame, output_dir: Path, dpi: int, step: float, vmax: float | None) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    plot_surface(
        ax,
        df,
        "orca_relative_kcalmol",
        "ORCA surface with HIP contours",
        r"ORCA relative energy [kcal mol$^{-1}$]",
        "turbo",
        contour_step=step,
        vmax=vmax,
    )
    x, y, z_hip = as_grid(df, "hip_relative_kcalmol")
    levels = contour_levels(z_hip.compressed(), step, vmax=vmax)
    ax.contour(x, y, z_hip, levels=levels, colors="white", linewidths=0.85, linestyles="--")
    add_min_marker(ax, df, "orca_relative_kcalmol", "ORCA")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"glycine_pt_orca_with_hip_contours.{suffix}", dpi=dpi)
    plt.close(fig)


def save_parity(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    x = df["orca_relative_kcalmol"].to_numpy(dtype=float)
    y = df["hip_relative_kcalmol"].to_numpy(dtype=float)
    sc = ax.scatter(x, y, c=df["q_nh"], s=42, cmap="viridis", edgecolor="k", linewidth=0.3)
    lim = [0.0, max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.03]
    ax.plot(lim, lim, "k--", linewidth=1.0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"ORCA relative energy [kcal mol$^{-1}$]")
    ax.set_ylabel(r"HIP relative energy [kcal mol$^{-1}$]")
    ax.set_title("HIP vs ORCA energy parity")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$q_\mathrm{NH}$ [$\AA$]")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"glycine_pt_hip_orca_parity.{suffix}", dpi=dpi)
    plt.close(fig)


def nearest_values(available: np.ndarray, requested: tuple[float, ...]) -> list[float]:
    out = []
    for value in requested:
        nearest = float(available[np.argmin(np.abs(available - value))])
        if nearest not in out:
            out.append(nearest)
    return out


def save_linecuts(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    requested_q_oh: tuple[float, ...],
    requested_q_nh: tuple[float, ...],
) -> None:
    q_oh_values = nearest_values(df["q_oh"].drop_duplicates().to_numpy(dtype=float), requested_q_oh)
    q_nh_values = nearest_values(df["q_nh"].drop_duplicates().to_numpy(dtype=float), requested_q_nh)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    for q_oh in q_oh_values:
        rows = df[np.isclose(df["q_oh"], q_oh)].sort_values("q_nh")
        axes[0].plot(rows["q_nh"], rows["orca_relative_kcalmol"], "o-", label=f"ORCA qOH={q_oh:.2f}")
        axes[0].plot(rows["q_nh"], rows["hip_relative_kcalmol"], "o--", label=f"HIP qOH={q_oh:.2f}")
    axes[0].set_xlabel(r"$q_\mathrm{NH}$ [$\AA$]")
    axes[0].set_ylabel(r"relative energy [kcal mol$^{-1}$]")
    axes[0].set_title(r"Cuts along $q_\mathrm{NH}$")
    axes[0].legend(fontsize=7, ncols=1)

    for q_nh in q_nh_values:
        rows = df[np.isclose(df["q_nh"], q_nh)].sort_values("q_oh")
        axes[1].plot(rows["q_oh"], rows["orca_relative_kcalmol"], "o-", label=f"ORCA qNH={q_nh:.2f}")
        axes[1].plot(rows["q_oh"], rows["hip_relative_kcalmol"], "o--", label=f"HIP qNH={q_nh:.2f}")
    axes[1].set_xlabel(r"$q_\mathrm{OH}$ [$\AA$]")
    axes[1].set_ylabel(r"relative energy [kcal mol$^{-1}$]")
    axes[1].set_title(r"Cuts along $q_\mathrm{OH}$")
    axes[1].legend(fontsize=7, ncols=1)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"glycine_pt_energy_linecuts.{suffix}", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir
    orca_dir = args.orca_dir
    hip_path = args.hip_energies or scan_dir / "hip_v2_energies.csv"
    orca_path = args.orca_energies or orca_dir / "metadata.csv"
    output_dir = args.output_dir or scan_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = merge_surfaces(load_hip(hip_path), load_orca(orca_path))
    df.to_csv(output_dir / "glycine_pt_energy_surface_data.csv", index=False)

    save_surface_comparison(
        df=df,
        output_dir=output_dir,
        dpi=args.dpi,
        step=args.energy_contour_step,
        vmax=args.energy_vmax,
    )
    save_overlay(df, output_dir, args.dpi, args.energy_contour_step, args.energy_vmax)
    save_parity(df, output_dir, args.dpi)
    save_linecuts(
        df,
        output_dir,
        args.dpi,
        requested_q_oh=tuple(args.linecut_q_oh),
        requested_q_nh=tuple(args.linecut_q_nh),
    )

    print(f"Wrote plots and merged data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
