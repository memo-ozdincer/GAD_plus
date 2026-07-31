#!/usr/bin/env python
"""Interactive 3D trajectory visualization from Parquet outputs.

Reads existing `traj_*.parquet` and `summary_*.parquet` files and writes an
interactive Plotly HTML animation for one (run_id, sample_id) trajectory.

Example:
  python scripts/visualize_3d.py \
      --traj-dir /lustre07/scratch/memoozd/gadplus/runs/noise_survey_300 \
      --pick fast \
      --noise-pm 50 \
      --method gad_projected \
      --split train
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write as ase_write
from ase.neighborlist import natural_cutoffs, neighbor_list

from plotting_style import apply_plot_style, palette_hex

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
apply_plot_style()


ELEMENT_COLORS = {
    1: palette_hex(7),  # H
    6: palette_hex(5),  # C
    7: palette_hex(0),  # N
    8: palette_hex(3),  # O
    9: palette_hex(2),  # F
    15: palette_hex(1),  # P
    16: palette_hex(8),  # S
    17: palette_hex(2),  # Cl
    35: palette_hex(5),  # Br
    53: palette_hex(4),  # I
}


def _auto_path(candidates: list[str], label: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{label} not found in candidates: {candidates}")


def _summary_df(summary_glob: str) -> pd.DataFrame:
    return duckdb.execute(f"SELECT * FROM '{summary_glob}'").df()


def _parquet_columns(parquet_glob: str) -> set[str]:
    schema_df = duckdb.execute(f"DESCRIBE SELECT * FROM '{parquet_glob}'").df()
    return set(schema_df["column_name"].tolist())


def _pick_target_row(
    sdf: pd.DataFrame,
    pick: str,
    method: str | None,
    noise_pm: int | None,
) -> pd.Series:
    df = sdf.copy()
    if method is not None:
        if "search_method" in df.columns:
            df = df[df["search_method"] == method]
        elif "method" in df.columns:
            df = df[df["method"] == method]
    if noise_pm is not None and "noise_pm" in df.columns:
        df = df[df["noise_pm"] == noise_pm]

    if len(df) == 0:
        raise ValueError("No summary rows match the provided filters")

    if pick == "first":
        return df.iloc[0]

    if "converged" not in df.columns:
        raise ValueError("Summary does not include 'converged', cannot use pick mode")

    if pick == "fast":
        conv = df[df["converged"] == True]
        if len(conv) == 0:
            raise ValueError("No converged rows available for pick=fast")
        if "converged_step" in conv.columns:
            idx = conv["converged_step"].idxmin()
            return conv.loc[idx]
        return conv.iloc[0]

    if pick == "slow":
        conv = df[df["converged"] == True]
        if len(conv) == 0:
            raise ValueError("No converged rows available for pick=slow")
        if "converged_step" in conv.columns:
            idx = conv["converged_step"].idxmax()
            return conv.loc[idx]
        return conv.iloc[-1]

    if pick == "failure":
        fail = df[df["converged"] == False]
        if len(fail) == 0:
            raise ValueError("No failed rows available for pick=failure")
        if "total_steps" in fail.columns:
            idx = fail["total_steps"].idxmax()
            return fail.loc[idx]
        return fail.iloc[0]

    raise ValueError(f"Unsupported pick mode: {pick}")


def _load_traj(traj_glob: str, run_id: str, sample_id: int) -> pd.DataFrame:
    available = _parquet_columns(traj_glob)
    required = ["step", "coords_flat"]
    optional = ["energy", "force_norm", "force_max", "force_rms", "n_neg", "eig0", "eig1"]
    missing_required = [col for col in required if col not in available]
    if missing_required:
        raise ValueError(
            f"Trajectory parquet is missing required columns: {missing_required}"
        )
    select_cols = required + [col for col in optional if col in available]
    return duckdb.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM '{traj_glob}'
        WHERE run_id = '{run_id}' AND sample_id = {sample_id}
        ORDER BY step
        """
    ).df()


def _load_atomic_numbers(h5_path: str, split: str, sample_id: int) -> tuple[np.ndarray, str]:
    from gadplus.data.transition1x import Transition1xDataset, UsePos

    ds = Transition1xDataset(
        h5_path,
        split=split,
        max_samples=sample_id + 1,
        transform=UsePos("pos_transition"),
    )
    if sample_id >= len(ds):
        raise IndexError(
            f"sample_id={sample_id} out of bounds for split={split} with len={len(ds)}"
        )
    sample = ds[sample_id]
    z = sample.z.detach().cpu().numpy().astype(int)
    formula = getattr(sample, "formula", f"sample_{sample_id}")
    return z, str(formula)


def _coords_from_flat(coords_flat: list[float], n_atoms: int) -> np.ndarray:
    arr = np.asarray(coords_flat, dtype=float)
    if arr.size != n_atoms * 3:
        raise ValueError(f"coords length mismatch: got {arr.size}, expected {n_atoms * 3}")
    return arr.reshape(n_atoms, 3)


def _downsample_indices(n: int, max_frames: int | None, stride: int | None) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if stride is not None and stride > 1:
        idx = np.arange(0, n, stride, dtype=int)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        return idx
    if max_frames is None or n <= max_frames:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, max_frames).round().astype(int)
    return np.unique(idx)


def _bond_edges(coords: np.ndarray, z: np.ndarray, cutoff_scale: float) -> list[tuple[int, int]]:
    atoms = Atoms(numbers=z.tolist(), positions=coords)
    cutoffs = natural_cutoffs(atoms, mult=cutoff_scale)
    i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)
    edges: set[tuple[int, int]] = set()
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        a, b = (int(i), int(j))
        if a == b:
            continue
        if a > b:
            a, b = b, a
        edges.add((a, b))
    return sorted(edges)


def _bond_lines(
    coords: np.ndarray, edges: Iterable[tuple[int, int]]
) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for i, j in edges:
        xs += [float(coords[i, 0]), float(coords[j, 0]), None]
        ys += [float(coords[i, 1]), float(coords[j, 1]), None]
        zs += [float(coords[i, 2]), float(coords[j, 2]), None]
    return xs, ys, zs


def _axis_bounds(
    all_coords: np.ndarray, pad: float = 0.7
) -> tuple[list[float], list[float], list[float]]:
    xyz_min = all_coords.min(axis=0)
    xyz_max = all_coords.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    span = float(np.max(xyz_max - xyz_min)) + 2 * pad
    half = 0.5 * span
    xr = [float(center[0] - half), float(center[0] + half)]
    yr = [float(center[1] - half), float(center[1] + half)]
    zr = [float(center[2] - half), float(center[2] + half)]
    return xr, yr, zr


def _trace_title(row: pd.Series, formula: str, run_id: str, sample_id: int) -> str:
    method = row.get("search_method", row.get("method", "unknown"))
    noise = row.get("noise_pm", "?")
    conv = row.get("converged", None)
    if conv is True:
        state = "CONVERGED"
    elif conv is False:
        state = "FAILED"
    else:
        state = "UNKNOWN"
    return f"{formula} | method={method} | noise={noise}pm | sample={sample_id} | run={run_id} | {state}"


def _viewer_slug(run_id: str, sample_id: int) -> str:
    safe_run_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_id)
    return f"{safe_run_id}_{sample_id}"


def _frame_comment(tdf: pd.DataFrame, idx: int) -> str:
    row = tdf.iloc[idx]
    metrics: list[str] = [f"step={int(row['step'])}"]
    for col, label, fmt in [
        ("energy", "E", "{:.6f}"),
        ("force_norm", "force_norm", "{:.5f}"),
        ("force_rms", "force_rms", "{:.5f}"),
        ("force_max", "fmax", "{:.5f}"),
        ("n_neg", "n_neg", "{}"),
    ]:
        if col in tdf.columns and pd.notna(row[col]):
            value = int(row[col]) if col == "n_neg" else float(row[col])
            metrics.append(f"{label}={fmt.format(value)}")
    return " | ".join(metrics)


def _build_atoms_frames(
    coords_series: list[np.ndarray],
    z: np.ndarray,
    tdf: pd.DataFrame,
) -> list[Atoms]:
    frames: list[Atoms] = []
    for idx, coords in enumerate(coords_series):
        atoms = Atoms(numbers=z.tolist(), positions=coords)
        atoms.info["comment"] = _frame_comment(tdf, idx)
        frames.append(atoms)
    return frames


def _write_viewer_bundle(
    base_dir: str,
    run_id: str,
    sample_id: int,
    formula: str,
    atoms_frames: list[Atoms],
) -> tuple[str, str, str]:
    slug = _viewer_slug(run_id, sample_id)
    bundle_dir = os.path.join(base_dir, "viewer_bundle", slug)
    sequence_dir = os.path.join(bundle_dir, "frames_xyz")
    os.makedirs(sequence_dir, exist_ok=True)

    multi_xyz = os.path.join(bundle_dir, f"{slug}.xyz")
    ase_write(multi_xyz, atoms_frames)

    for idx, atoms in enumerate(atoms_frames):
        frame_path = os.path.join(sequence_dir, f"frame_{idx:04d}.xyz")
        ase_write(frame_path, atoms)

    readme_path = os.path.join(bundle_dir, "README_viewers.md")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(
            "\n".join(
                [
                    f"# Trajectory Viewer Bundle: {formula}",
                    "",
                    "Recommended VS Code extensions:",
                    "- `arianjamasb.protein-viewer`",
                    "- `stevenyu.nano-protein-viewer`",
                    "",
                    "Suggested usage:",
                    f"- Open `{os.path.basename(multi_xyz)}` with Protein Viewer for a single-file trajectory.",
                    f"- Open the `frames_xyz/` folder with Nano Protein Viewer for frame-by-frame browsing or sequence playback.",
                    "",
                    "Files:",
                    f"- Multi-frame XYZ: `{os.path.basename(multi_xyz)}`",
                    f"- Per-frame XYZ folder: `{os.path.basename(sequence_dir)}/`",
                ]
            )
            + "\n"
        )

    return bundle_dir, multi_xyz, sequence_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj-dir",
        type=str,
        required=True,
        help="Directory with summary_*.parquet and traj_*.parquet",
    )
    parser.add_argument("--summary-pattern", type=str, default="summary_*.parquet")
    parser.add_argument("--traj-pattern", type=str, default="traj_*.parquet")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument(
        "--pick", type=str, default="first", choices=["first", "fast", "slow", "failure"]
    )
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--noise-pm", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--bond-source", type=str, default="first", choices=["first", "none"])
    parser.add_argument("--cutoff-scale", type=float, default=1.2)
    parser.add_argument(
        "--output-mode",
        type=str,
        default="viewer",
        choices=["viewer", "plotly", "both"],
        help="Write VS Code viewer bundle, Plotly HTML, or both",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-html", type=str, default=None)
    args = parser.parse_args()

    go = None
    if args.output_mode in {"plotly", "both"}:
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError(
                "plotly is required for --output-mode plotly/both. "
                "Use --output-mode viewer or install with: pip install plotly"
            ) from exc

    summary_glob = os.path.join(args.traj_dir, args.summary_pattern)
    traj_glob = os.path.join(args.traj_dir, args.traj_pattern)
    sdf = _summary_df(summary_glob)
    if len(sdf) == 0:
        raise ValueError(f"No summary rows found in {summary_glob}")

    if args.run_id is not None and args.sample_id is not None:
        row = sdf[(sdf["run_id"] == args.run_id) & (sdf["sample_id"] == args.sample_id)]
        if len(row) == 0:
            raise ValueError("Requested (run_id, sample_id) not found in summary parquet")
        row = row.iloc[0]
    else:
        row = _pick_target_row(sdf, args.pick, args.method, args.noise_pm)

    run_id = str(row["run_id"])
    sample_id = int(row["sample_id"])

    tdf = _load_traj(traj_glob, run_id, sample_id)
    if len(tdf) == 0:
        raise ValueError(f"No trajectory rows found for run_id={run_id}, sample_id={sample_id}")

    h5_path = args.h5_path or _auto_path(
        [
            "/lustre06/project/6033559/memoozd/data/transition1x.h5",
            "/project/rrg-aspuru/memoozd/data/transition1x.h5",
        ],
        "Transition1x HDF5",
    )
    z, formula = _load_atomic_numbers(h5_path, args.split, sample_id)
    n_atoms = len(z)

    frame_idx = _downsample_indices(len(tdf), args.max_frames, args.stride)
    tdf = tdf.iloc[frame_idx].reset_index(drop=True)

    coords_series = [_coords_from_flat(v, n_atoms) for v in tdf["coords_flat"].tolist()]
    atoms_frames = _build_atoms_frames(coords_series, z, tdf)
    all_coords = np.concatenate(coords_series, axis=0)
    xr, yr, zr = _axis_bounds(all_coords)
    out_root = args.output_dir or os.path.join(args.traj_dir, "plots")
    os.makedirs(out_root, exist_ok=True)

    bundle_dir = None
    multi_xyz = None
    sequence_dir = None
    if args.output_mode in {"viewer", "both"}:
        bundle_dir, multi_xyz, sequence_dir = _write_viewer_bundle(
            out_root, run_id, sample_id, formula, atoms_frames
        )

    if args.bond_source == "first":
        edges = _bond_edges(coords_series[0], z, args.cutoff_scale)
    else:
        edges = []

    atom_colors = [ELEMENT_COLORS.get(int(zi), palette_hex(7)) for zi in z]
    atom_sizes = [14 if int(zi) == 1 else 18 for zi in z]

    def frame_data(coords: np.ndarray, idx: int):
        atoms_trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker={"size": atom_sizes, "color": atom_colors, "opacity": 0.96},
            text=[f"atom={i} Z={int(z[i])}" for i in range(n_atoms)],
            hovertemplate="%{text}<extra></extra>",
            name="atoms",
        )

        traces = [atoms_trace]
        if edges:
            bx, by, bz = _bond_lines(coords, edges)
            bond_trace = go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="lines",
                line={"width": 5, "color": "rgba(120,120,120,0.85)"},
                hoverinfo="skip",
                name="bonds",
            )
            traces.insert(0, bond_trace)

        return traces, _frame_comment(tdf, idx)

    out_html = None
    if args.output_mode in {"plotly", "both"}:
        traces0, subtitle0 = frame_data(coords_series[0], 0)
        frames = []
        for i, coords in enumerate(coords_series):
            traces_i, subtitle_i = frame_data(coords, i)
            frames.append(
                go.Frame(
                    data=traces_i,
                    name=str(i),
                    layout={
                        "title": {
                            "text": f"{_trace_title(row, formula, run_id, sample_id)}<br><sup>{subtitle_i}</sup>"
                        }
                    },
                )
            )

        fig = go.Figure(data=traces0, frames=frames)
        fig.update_layout(
            title={
                "text": f"{_trace_title(row, formula, run_id, sample_id)}<br><sup>{subtitle0}</sup>"
            },
            scene={
                "xaxis": {"range": xr, "title": "x (A)", "showgrid": True},
                "yaxis": {"range": yr, "title": "y (A)", "showgrid": True},
                "zaxis": {"range": zr, "title": "z (A)", "showgrid": True},
                "aspectmode": "cube",
            },
            margin={"l": 0, "r": 0, "t": 70, "b": 0},
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.05,
                    "y": 0.0,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True},
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "x": 0.17,
                    "y": 0.0,
                    "len": 0.8,
                    "steps": [
                        {
                            "method": "animate",
                            "args": [
                                [str(i)],
                                {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
                            ],
                            "label": str(int(tdf.iloc[i]["step"])),
                        }
                        for i in range(len(tdf))
                    ],
                }
            ],
        )

        if args.output_html is None:
            out_html = os.path.join(out_root, f"traj3d_{run_id}_{sample_id}.html")
        else:
            out_html = args.output_html
            os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)

        fig.write_html(out_html, include_plotlyjs="cdn")

    if bundle_dir is not None:
        print(f"Saved viewer bundle: {bundle_dir}")
        print(f"Protein Viewer XYZ: {multi_xyz}")
        print(f"Nano Protein Viewer frames: {sequence_dir}")
    if out_html is not None:
        print(f"Saved 3D trajectory HTML: {out_html}")
    print(f"Frames: {len(tdf)} | run_id={run_id} | sample_id={sample_id} | formula={formula}")


if __name__ == "__main__":
    main()
