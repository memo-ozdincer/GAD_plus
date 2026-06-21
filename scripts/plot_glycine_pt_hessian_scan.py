#!/usr/bin/env python
"""Plot glycine proton-transfer PES and Hessian-comparison diagnostics.

Defaults match the current Transition1x test sample 5 scan:
q_nh = d(N4, H9), q_oh = d(O3, H9). ORCA wB97X/6-31G(d) is treated as the
reference. HIP/MLIP Hessians are assumed to be Cartesian Hessians in eV/A^2 on
the same grid.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HARTREE_TO_EV = 27.211386245988
EV_TO_KCALMOL = 23.060548867
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR2_TO_EV_PER_ANG2 = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**2)
MASS_BY_Z = {
    1: 1.008,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    15: 30.974,
    16: 32.065,
    17: 35.453,
}


@dataclass
class ModelData:
    label: str
    hessians_ev_ang2: np.ndarray
    energies_ev: np.ndarray | None = None


@dataclass
class VibDiagnostics:
    evals: np.ndarray
    modes: np.ndarray
    n_negative: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("runs/glycine_pt_scan"),
        help="Directory containing HIP scan outputs.",
    )
    parser.add_argument(
        "--orca-dir",
        type=Path,
        default=Path("orca_wb97x_631gd_glycine_pt_nh_oh_scan_80"),
        help="Directory containing ORCA HDF5 and metadata.",
    )
    parser.add_argument(
        "--hip-arrays",
        type=Path,
        default=None,
        help="HIP NPZ with hessians_cartesian. Defaults to scan-dir/hip_v2_arrays.npz.",
    )
    parser.add_argument(
        "--hip-predictions",
        type=Path,
        default=None,
        help="HIP predictions CSV/parquet. Defaults to scan-dir/hip_v2_predictions.csv.",
    )
    parser.add_argument(
        "--mlip-arrays",
        type=Path,
        default=None,
        help="Optional MLIP-autograd NPZ on the same grid.",
    )
    parser.add_argument("--mlip-label", default="MLIP autograd")
    parser.add_argument("--hessian-key", default="hessians_cartesian")
    parser.add_argument("--energy-key", default="energies")
    parser.add_argument("--n-eigs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_orca(orca_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    metadata = pd.read_csv(orca_dir / "metadata.csv")
    h5_path = orca_dir / "h5" / "glycine_pt_scan.h5"
    with h5py.File(h5_path, "r") as h5:
        arrays = {
            "grid_id": np.asarray(h5["grid_id"][:], dtype=int),
            "atomic_numbers": np.asarray(h5["atomic_numbers"][:], dtype=int),
            "coords_angstrom": np.asarray(h5["coordinates_bohr"][:], dtype=float)
            * BOHR_TO_ANGSTROM,
            "energy_hartree": np.asarray(h5["energy_hartree"][:], dtype=float),
            "hessian_ev_ang2": np.asarray(h5["hessian_hartree_per_bohr2"][:], dtype=float)
            * HARTREE_PER_BOHR2_TO_EV_PER_ANG2,
            "q_nh": np.asarray(h5["q_nh_angstrom"][:], dtype=float),
            "q_oh": np.asarray(h5["q_oh_angstrom"][:], dtype=float),
        }
    return metadata, arrays


def load_npz_model(
    label: str,
    npz_path: Path,
    hessian_key: str,
    energy_key: str,
    n_grid: int,
) -> ModelData:
    data = np.load(npz_path)
    if hessian_key not in data:
        raise KeyError(f"{npz_path} does not contain {hessian_key!r}; keys={data.files}")
    hessians = np.asarray(data[hessian_key], dtype=float)
    if hessians.shape[0] != n_grid:
        raise ValueError(f"{label} has {hessians.shape[0]} Hessians, expected {n_grid}")
    energies = np.asarray(data[energy_key], dtype=float) if energy_key in data else None
    return ModelData(label=label, hessians_ev_ang2=hessians, energies_ev=energies)


def load_hip_model(
    scan_dir: Path,
    arrays_path: Path,
    predictions_path: Path,
    hessian_key: str,
) -> ModelData:
    predictions = read_table(predictions_path).sort_values("grid_id")
    model = load_npz_model(
        label="HIP direct",
        npz_path=arrays_path,
        hessian_key=hessian_key,
        energy_key="energies",
        n_grid=len(predictions),
    )
    model.energies_ev = predictions["hip_v2_energy"].to_numpy(dtype=float)
    return model


def symmetrize(hessians: np.ndarray) -> np.ndarray:
    return 0.5 * (hessians + np.swapaxes(hessians, -1, -2))


def frob_relative_error(model_h: np.ndarray, ref_h: np.ndarray) -> np.ndarray:
    diff = symmetrize(model_h) - symmetrize(ref_h)
    denom = np.linalg.norm(ref_h.reshape(ref_h.shape[0], -1), axis=1)
    numer = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    return numer / np.maximum(denom, 1e-12)


def reaction_center_error(
    model_h: np.ndarray,
    ref_h: np.ndarray,
    atoms: tuple[int, ...] = (3, 4, 9),
) -> np.ndarray:
    idx = np.array([3 * atom + comp for atom in atoms for comp in range(3)], dtype=int)
    model_block = symmetrize(model_h)[:, idx[:, None], idx]
    ref_block = symmetrize(ref_h)[:, idx[:, None], idx]
    denom = np.linalg.norm(ref_block.reshape(ref_block.shape[0], -1), axis=1)
    numer = np.linalg.norm((model_block - ref_block).reshape(model_block.shape[0], -1), axis=1)
    return numer / np.maximum(denom, 1e-12)


def compute_vib_diagnostics(
    hessians_ev_ang2: np.ndarray,
    coords_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    n_eigs: int,
) -> VibDiagnostics:
    n_grid = hessians_ev_ang2.shape[0]
    eval_rows = []
    mode_rows = []
    n_negative = []
    for idx in range(n_grid):
        evals_np, modes_np = vibrational_eigh(
            hessian_ev_ang2=hessians_ev_ang2[idx],
            coords_angstrom=coords_angstrom[idx],
            atomic_numbers=atomic_numbers[idx],
        )
        eval_rows.append(evals_np[:n_eigs])
        mode_rows.append(modes_np[:, :n_eigs])
        n_negative.append(int((evals_np < -1e-6).sum()))
    return VibDiagnostics(
        evals=np.stack(eval_rows),
        modes=np.stack(mode_rows),
        n_negative=np.asarray(n_negative, dtype=int),
    )


def mode_overlap(model_modes: np.ndarray, ref_modes: np.ndarray, mode_index: int = 0) -> np.ndarray:
    model = model_modes[:, :, mode_index]
    ref = ref_modes[:, :, mode_index]
    dots = np.einsum("ij,ij->i", model, ref)
    model_norm = np.linalg.norm(model, axis=1)
    ref_norm = np.linalg.norm(ref, axis=1)
    return np.abs(dots) / np.maximum(model_norm * ref_norm, 1e-12)


def eckart_generators(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    masses = np.asarray(masses, dtype=float).reshape(-1)
    n_atoms = xyz.shape[0]
    sqrt_m = np.sqrt(masses)
    sqrt_m3 = np.repeat(sqrt_m, 3)

    com = (xyz * masses[:, None]).sum(axis=0) / masses.sum()
    rel = xyz - com[None, :]

    cols = []
    for axis in np.eye(3):
        col = sqrt_m3 * np.tile(axis, n_atoms)
        cols.append(col / max(np.linalg.norm(col), 1e-12))

    rx, ry, rz = rel[:, 0], rel[:, 1], rel[:, 2]
    rotations = (
        np.stack([np.zeros_like(rx), -rz, ry], axis=1),
        np.stack([rz, np.zeros_like(ry), -rx], axis=1),
        np.stack([-ry, rx, np.zeros_like(rz)], axis=1),
    )
    for rot in rotations:
        col = (rot * sqrt_m[:, None]).reshape(-1)
        norm = np.linalg.norm(col)
        if norm > 1e-12:
            cols.append(col / norm)
    return np.stack(cols, axis=1)


def vibrational_basis(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    generators = eckart_generators(coords, masses)
    q, r = np.linalg.qr(generators, mode="reduced")
    diag = np.abs(np.diag(r))
    rank = max(int((diag > 1e-6).sum()), 1)
    u, _, _ = np.linalg.svd(q[:, :rank], full_matrices=True)
    return u[:, rank:]


def vibrational_eigh(
    hessian_ev_ang2: np.ndarray,
    coords_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    atomic_numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    masses = np.array([MASS_BY_Z[int(z)] for z in atomic_numbers], dtype=float)
    n_atoms = atomic_numbers.size
    hessian = np.asarray(hessian_ev_ang2, dtype=float).reshape(3 * n_atoms, 3 * n_atoms)
    hessian = 0.5 * (hessian + hessian.T)
    m3 = np.repeat(masses, 3)
    hessian_mw = hessian / np.sqrt(np.outer(m3, m3))
    q_vib = vibrational_basis(coords_angstrom, masses)
    hessian_red = q_vib.T @ hessian_mw @ q_vib
    hessian_red = 0.5 * (hessian_red + hessian_red.T)
    evals, evecs_red = np.linalg.eigh(hessian_red)
    return evals, q_vib @ evecs_red


def to_grid(df: pd.DataFrame, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tmp = df[["q_nh", "q_oh"]].copy()
    tmp["value"] = values
    pivot = tmp.pivot(index="q_oh", columns="q_nh", values="value").sort_index()
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    return x, y, z


def heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    levels: int = 15,
    contour_values: np.ndarray | None = None,
) -> None:
    x, y, z = to_grid(df, values)
    mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap)
    if contour_values is not None:
        _, _, contour_z = to_grid(df, contour_values)
        ax.contour(x, y, contour_z, levels=levels, colors="k", linewidths=0.45, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel(r"$q_\mathrm{NH}$ = d(N4,H9) [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}$ = d(O3,H9) [$\AA$]")
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)


def save_energy_figure(
    df: pd.DataFrame,
    orca_energy_kcal: np.ndarray,
    models: list[ModelData],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2 + len(models), figsize=(5.2 * (2 + len(models)), 4.4))
    axes = np.atleast_1d(axes)
    heatmap(
        axes[0],
        df,
        orca_energy_kcal,
        "ORCA wB97X/6-31G(d) PES",
        r"relative energy [kcal mol$^{-1}$]",
        cmap="magma",
    )
    for ax, model in zip(axes[1:], models, strict=False):
        if model.energies_ev is None:
            ax.axis("off")
            ax.set_title(f"{model.label}: no energies")
            continue
        model_rel_kcal = (model.energies_ev - np.nanmin(model.energies_ev)) * EV_TO_KCALMOL
        heatmap(
            ax,
            df,
            model_rel_kcal,
            f"{model.label} PES",
            r"relative energy [kcal mol$^{-1}$]",
            cmap="magma",
            contour_values=orca_energy_kcal,
        )
    if models:
        last_ax = axes[-1]
        model = models[0]
        if model.energies_ev is not None:
            model_rel_kcal = (model.energies_ev - np.nanmin(model.energies_ev)) * EV_TO_KCALMOL
            heatmap(
                last_ax,
                df,
                model_rel_kcal - orca_energy_kcal,
                f"{model.label} - ORCA energy error",
                r"relative energy error [kcal mol$^{-1}$]",
                cmap="coolwarm",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "glycine_pt_energy_surfaces.png", dpi=dpi)
    plt.close(fig)


def save_hessian_metric_figure(
    df: pd.DataFrame,
    orca_energy_kcal: np.ndarray,
    ref_diag: VibDiagnostics,
    model: ModelData,
    model_diag: VibDiagnostics,
    ref_hessians: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    rel_frob = frob_relative_error(model.hessians_ev_ang2, ref_hessians)
    rc_frob = reaction_center_error(model.hessians_ev_ang2, ref_hessians)
    eig0_error = model_diag.evals[:, 0] - ref_diag.evals[:, 0]
    overlap0 = mode_overlap(model_diag.modes, ref_diag.modes, mode_index=0)
    nneg_delta = model_diag.n_negative - ref_diag.n_negative

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.7))
    heatmap(
        axes[0, 0],
        df,
        rel_frob,
        f"{model.label}: full Hessian error",
        r"$||H-H_\mathrm{DFT}||_F / ||H_\mathrm{DFT}||_F$",
        cmap="viridis",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[0, 1],
        df,
        rc_frob,
        f"{model.label}: reaction-center error",
        r"relative Frobenius error",
        cmap="viridis",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[0, 2],
        df,
        eig0_error,
        f"{model.label}: lowest eigenvalue error",
        r"$\lambda_0 - \lambda_{0,\mathrm{DFT}}$ [eV A$^{-2}$ amu$^{-1}$]",
        cmap="coolwarm",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[1, 0],
        df,
        overlap0,
        f"{model.label}: unstable-mode overlap",
        r"$|\langle v_0, v_{0,\mathrm{DFT}}\rangle|$",
        cmap="magma",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[1, 1],
        df,
        ref_diag.n_negative,
        "ORCA number of negative modes",
        "count",
        cmap="viridis",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[1, 2],
        df,
        nneg_delta,
        f"{model.label}: negative-mode count error",
        r"$n_\mathrm{neg} - n_{\mathrm{neg,DFT}}$",
        cmap="coolwarm",
        contour_values=orca_energy_kcal,
    )
    fig.tight_layout()
    safe_label = model.label.lower().replace(" ", "_").replace("/", "_")
    fig.savefig(output_dir / f"glycine_pt_hessian_metrics_{safe_label}.png", dpi=dpi)
    plt.close(fig)

    metrics = df[["grid_id", "q_nh", "q_oh"]].copy()
    metrics[f"{safe_label}_relative_hessian_error"] = rel_frob
    metrics[f"{safe_label}_reaction_center_error"] = rc_frob
    metrics[f"{safe_label}_eig0_error"] = eig0_error
    metrics[f"{safe_label}_mode0_overlap"] = overlap0
    metrics[f"{safe_label}_nneg_delta"] = nneg_delta
    return metrics


def save_low_mode_figure(
    df: pd.DataFrame,
    ref_diag: VibDiagnostics,
    model_diags: dict[str, VibDiagnostics],
    output_dir: Path,
    dpi: int,
) -> int:
    # Pick the DFT point closest to an index-1 saddle with the most negative lowest mode.
    index1 = np.where(ref_diag.n_negative == 1)[0]
    if len(index1) > 0:
        selected = int(index1[np.argmin(ref_diag.evals[index1, 0])])
    else:
        selected = int(np.argmin(ref_diag.evals[:, 0]))

    row = df.iloc[selected]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    mode_ids = np.arange(ref_diag.evals.shape[1])
    ax.plot(mode_ids, ref_diag.evals[selected], "o-", label="ORCA DFT")
    for label, diag in model_diags.items():
        ax.plot(mode_ids, diag.evals[selected], "o--", label=label)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("vibrational mode index")
    ax.set_ylabel(r"projected Hessian eigenvalue [eV A$^{-2}$ amu$^{-1}$]")
    ax.set_title(
        f"Low-mode spectrum at grid {int(row.grid_id)} "
        f"($q_{{NH}}$={row.q_nh:.3f} A, $q_{{OH}}$={row.q_oh:.3f} A)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "glycine_pt_low_mode_spectrum.png", dpi=dpi)
    plt.close(fig)
    return selected


def save_reaction_center_blocks(
    selected: int,
    df: pd.DataFrame,
    ref_hessians: np.ndarray,
    models: list[ModelData],
    output_dir: Path,
    dpi: int,
    atoms: tuple[int, ...] = (3, 4, 9),
) -> None:
    idx = np.array([3 * atom + comp for atom in atoms for comp in range(3)], dtype=int)
    labels = [f"{atom}{axis}" for atom in atoms for axis in ("x", "y", "z")]
    row = df.iloc[selected]

    ncols = 1 + len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2), squeeze=False)
    ref_block = symmetrize(ref_hessians)[selected][idx[:, None], idx]
    vmax = np.nanmax(np.abs(ref_block))
    im = axes[0, 0].imshow(ref_block, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("ORCA reaction-center Hessian")
    for ax in axes[0]:
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    for ax, model in zip(axes[0, 1:], models, strict=False):
        block = symmetrize(model.hessians_ev_ang2)[selected][idx[:, None], idx] - ref_block
        vmax_diff = np.nanmax(np.abs(block))
        im = ax.imshow(block, cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)
        ax.set_title(f"{model.label} - ORCA")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Reaction-center block at grid {int(row.grid_id)} "
        f"($q_{{NH}}$={row.q_nh:.3f} A, $q_{{OH}}$={row.q_oh:.3f} A)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "glycine_pt_reaction_center_hessian_blocks.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir
    output_dir = args.output_dir or scan_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    hip_arrays = args.hip_arrays or scan_dir / "hip_v2_arrays.npz"
    hip_predictions = args.hip_predictions or scan_dir / "hip_v2_predictions.csv"

    _, orca = load_orca(args.orca_dir)
    df = read_table(hip_predictions).sort_values("grid_id").reset_index(drop=True)
    grid_ids = df["grid_id"].to_numpy(dtype=int)
    order = np.argsort(orca["grid_id"])
    for key in ("grid_id", "atomic_numbers", "coords_angstrom", "energy_hartree", "hessian_ev_ang2"):
        orca[key] = orca[key][order]
    if not np.array_equal(grid_ids, orca["grid_id"]):
        raise ValueError("HIP predictions and ORCA HDF5 grid_id ordering do not match")

    orca_rel_kcal = (orca["energy_hartree"] - np.nanmin(orca["energy_hartree"])) * HARTREE_TO_EV
    orca_rel_kcal = orca_rel_kcal * EV_TO_KCALMOL
    ref_hessians = orca["hessian_ev_ang2"]
    coords_angstrom = orca["coords_angstrom"]
    atomic_numbers = orca["atomic_numbers"]

    models = [
        load_hip_model(scan_dir, hip_arrays, hip_predictions, args.hessian_key),
    ]
    if args.mlip_arrays is not None:
        models.append(
            load_npz_model(
                label=args.mlip_label,
                npz_path=args.mlip_arrays,
                hessian_key=args.hessian_key,
                energy_key=args.energy_key,
                n_grid=len(df),
            )
        )

    print("Computing ORCA vibrational diagnostics...", flush=True)
    ref_diag = compute_vib_diagnostics(
        ref_hessians, coords_angstrom, atomic_numbers, n_eigs=args.n_eigs
    )
    model_diags: dict[str, VibDiagnostics] = {}
    for model in models:
        print(f"Computing {model.label} vibrational diagnostics...", flush=True)
        model_diags[model.label] = compute_vib_diagnostics(
            model.hessians_ev_ang2, coords_angstrom, atomic_numbers, n_eigs=args.n_eigs
        )

    save_energy_figure(df, orca_rel_kcal, models, output_dir, args.dpi)

    metrics_frames = [df[["grid_id", "q_nh", "q_oh"]].copy()]
    metrics_frames[0]["orca_energy_relative_kcalmol"] = orca_rel_kcal
    metrics_frames[0]["orca_eig0"] = ref_diag.evals[:, 0]
    metrics_frames[0]["orca_n_negative"] = ref_diag.n_negative
    for model in models:
        metrics = save_hessian_metric_figure(
            df=df,
            orca_energy_kcal=orca_rel_kcal,
            ref_diag=ref_diag,
            model=model,
            model_diag=model_diags[model.label],
            ref_hessians=ref_hessians,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        metrics_frames.append(metrics.drop(columns=["q_nh", "q_oh"]))

    selected = save_low_mode_figure(df, ref_diag, model_diags, output_dir, args.dpi)
    save_reaction_center_blocks(selected, df, ref_hessians, models, output_dir, args.dpi)

    metrics_df = metrics_frames[0]
    for frame in metrics_frames[1:]:
        metrics_df = metrics_df.merge(frame, on="grid_id", how="left")
    metrics_path = output_dir / "glycine_pt_hessian_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Wrote plots to {output_dir}", flush=True)
    print(f"Wrote metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
