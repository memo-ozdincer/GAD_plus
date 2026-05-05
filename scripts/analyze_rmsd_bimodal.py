#!/usr/bin/env python
"""Investigate the bimodal Sella RMSD-to-TS distribution and the
"converged but not a saddle" failure mode.

Reads test_set/ summaries and computes:
  1. RMSD from each final geometry to the known TS (Kabsch+Hungarian)
  2. The breakdown of "converged" claims:
     - n_neg=1 ∧ fmax<0.01     (real saddle, fmax)
     - n_neg=1 ∧ fmax<1e-4     (real saddle, strict)
     - n_neg=1 (any force)     (saddle but maybe not relaxed)
     - fmax<0.01 ∧ n_neg≠1     (false converged)
  3. Compute time / function evals / step budget consumed

Outputs:
  CSV:     analysis_2026_04_29/test_summary_full.csv
  PDF:     figures/fig_rmsd_distrib_test.pdf
  PDF:     figures/fig_compute_compare_test.pdf
  Text:    analysis_2026_04_29/saddle_quality_table.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from plotting_style import apply_plot_style, palette_color

sys.path.insert(0, "/lustre06/project/6033559/memoozd/GAD_plus/src")
apply_plot_style()

BASE = Path("/lustre07/scratch/memoozd/gadplus/runs")
OUT_DIR = Path("/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29")
FIG_DIR = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)
NOISES = [10, 30, 50, 100, 150, 200]


def find_summary(method_dir, noise_pm):
    if not method_dir.exists():
        return None
    cands = [f for f in os.listdir(method_dir)
             if f.startswith("summary") and f.endswith(".parquet") and f"_{noise_pm}pm" in f]
    return method_dir / cands[0] if cands else None


def kabsch_rmsd(pos_a, pos_b):
    """RMSD after Kabsch alignment, no atom permutation. pos_*: (N,3) np."""
    A = pos_a - pos_a.mean(0)
    B = pos_b - pos_b.mean(0)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    A_rot = A @ R.T
    return float(np.sqrt(((A_rot - B) ** 2).sum() / len(A)))


def load_test_dataset_ts():
    """Load known TS coords for test split, indexed by sample_id."""
    from gadplus.data.transition1x import Transition1xDataset, UsePos
    h5 = "/lustre06/project/6033559/memoozd/data/transition1x.h5"
    if not os.path.exists(h5):
        h5 = "/project/rrg-aspuru/memoozd/data/transition1x.h5"
    ds = Transition1xDataset(h5, split="test", max_samples=287, transform=UsePos("pos_transition"))
    ts_coords = {}
    for i in range(len(ds)):
        s = ds[i]
        ts_coords[i] = s.pos.detach().cpu().numpy().astype(np.float64).reshape(-1, 3)
    return ts_coords


def collect_method_data(method_dir, conv_col, ts_coords, label):
    """Pull final coords + metrics from each cell, compute RMSD-to-TS."""
    rows = []
    for noise in NOISES:
        p = find_summary(method_dir, noise)
        if p is None:
            continue
        cols_avail = set(duckdb.execute(f"DESCRIBE SELECT * FROM '{p}'").df()["column_name"])
        if "coords_flat" not in cols_avail:
            continue  # GAD summaries may not have coords; skip
        df = duckdb.execute(f"SELECT * FROM '{p}'").df()
        for _, r in df.iterrows():
            sid = int(r["sample_id"])
            if sid not in ts_coords:
                continue
            cf = r["coords_flat"]
            if cf is None:
                continue
            arr = np.asarray(cf, dtype=np.float64).reshape(-1, 3)
            if arr.shape != ts_coords[sid].shape:
                continue
            try:
                rmsd = kabsch_rmsd(arr, ts_coords[sid])
            except Exception:
                rmsd = np.nan
            rows.append({
                "method": label,
                "noise_pm": noise,
                "sample_id": sid,
                "rmsd_to_ts": rmsd,
                "final_n_neg": int(r.get("final_n_neg", r.get("n_neg", -1))),
                "final_fmax": float(r.get("final_fmax", -1)),
                "final_force_norm": float(r.get("final_force_norm", -1)),
                "total_steps": int(r.get("total_steps", -1)),
                "wall_time_s": float(r.get("wall_time_s", -1)),
                "n_func_evals": int(r.get("n_func_evals", -1)) if "n_func_evals" in cols_avail else -1,
                "is_saddle": int(r.get("final_n_neg", -1)) == 1,
                "fmax_strict": float(r.get("final_fmax", 999)) < 1e-4,
                "fmax_loose": float(r.get("final_fmax", 999)) < 0.01,
                "false_conv_min": (float(r.get("final_fmax", 999)) < 0.01)
                                  and int(r.get("final_n_neg", -1)) == 0,
            })
    return rows


def main():
    print("Loading test split TS coords…")
    ts_coords = load_test_dataset_ts()
    print(f"  {len(ts_coords)} samples")

    methods = {
        "Sella cart+Eckart, delta0=0.048 gamma=0 H/step":
            (BASE / "test_set/sella_carteck_default", "conv_nneg1_fmax001"),
        "Sella cart+Eckart, delta0=0.10 gamma=0.40 H/step":
            (BASE / "test_set/sella_carteck_libdef", "conv_nneg1_fmax001"),
        "Sella internal, delta0=0.048 gamma=0 H/step":
            (BASE / "test_set/sella_internal_default", "conv_nneg1_fmax001"),
    }
    # GAD methods don't have coords_flat in summary; they live in traj parquets.
    # Pull from sella_baseline-saved Sella summaries (which DO have coords_flat).

    all_rows = []
    for label, (mdir, _) in methods.items():
        rows = collect_method_data(mdir, "conv_nneg1_fmax001", ts_coords, label)
        all_rows.extend(rows)
        print(f"  {label}: {len(rows)} samples")

    if not all_rows:
        print("No data with coords_flat available yet — re-run after Sella sweeps complete.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "test_summary_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    # ============ Saddle-quality table ============
    print("\nSaddle-quality breakdown:")
    print("=" * 100)
    print(f"{'method':<52} {'noise':>5} {'n':>4}  "
          f"{'sad+fmax<.01':>14} {'sad+fmax<1e-4':>15} "
          f"{'sad-only':>10} {'false_min':>10}  {'med_RMSD':>9}")
    md_lines = ["# Saddle quality on test\n",
                "| method | noise | n | n_neg=1 ∧ fmax<.01 | n_neg=1 ∧ fmax<1e-4 | n_neg=1 only | fmax<.01 ∧ n_neg=0 | median RMSD-to-TS (Å) |",
                "|---|---|---|---|---|---|---|---|"]
    for (m, n), g in df.groupby(["method", "noise_pm"]):
        n_total = len(g)
        sad_loose = (g["is_saddle"] & g["fmax_loose"]).sum()
        sad_strict = (g["is_saddle"] & g["fmax_strict"]).sum()
        sad_only = g["is_saddle"].sum()
        false_min = g["false_conv_min"].sum()
        med_rmsd = g["rmsd_to_ts"].median()
        line = (f"{m:<52} {n:>5} {n_total:>4}  "
                f"{100*sad_loose/n_total:>13.1f}% {100*sad_strict/n_total:>14.1f}%  "
                f"{100*sad_only/n_total:>9.1f}% {100*false_min/n_total:>9.1f}%  {med_rmsd:>9.4f}")
        print(line)
        md_lines.append(f"| {m} | {n} | {n_total} | {100*sad_loose/n_total:.1f}% | "
                        f"{100*sad_strict/n_total:.1f}% | {100*sad_only/n_total:.1f}% | "
                        f"{100*false_min/n_total:.1f}% | {med_rmsd:.4f} |")
    (OUT_DIR / "saddle_quality_table.md").write_text("\n".join(md_lines))
    print(f"wrote {OUT_DIR / 'saddle_quality_table.md'}")

    # ============ RMSD distribution figure ============
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=False)
    axes = axes.flatten()
    colors = {
        "Sella cart+Eckart, delta0=0.048 gamma=0 H/step": palette_color(3),
        "Sella cart+Eckart, delta0=0.10 gamma=0.40 H/step": palette_color(1),
        "Sella internal, delta0=0.048 gamma=0 H/step": palette_color(4),
    }
    for ax, noise in zip(axes, NOISES):
        for m in colors:
            g = df[(df["method"] == m) & (df["noise_pm"] == noise)]
            if not len(g): continue
            rmsds = g["rmsd_to_ts"].dropna().values
            rmsds = rmsds[rmsds < 5.0]  # clip outliers for visualization
            if len(rmsds):
                ax.hist(rmsds, bins=40, alpha=0.5, color=colors[m], label=m, density=True)
        ax.set_title(f"{noise} pm")
        ax.set_xlabel("RMSD to true TS (Å)")
        ax.set_ylabel("density")
        ax.set_xlim(0, 2)
        ax.legend(fontsize=7)
    fig.suptitle("Sella RMSD-to-TS distributions on test (clipped at 5Å)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rmsd_distrib_test.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_rmsd_distrib_test.png", bbox_inches="tight", dpi=150)
    print(f"wrote {FIG_DIR / 'fig_rmsd_distrib_test.pdf'}")

    # ============ Compute time table ============
    print("\nCompute (median per-sample over CONVERGED only):")
    print("=" * 80)
    print(f"{'method':<18} {'noise':>5}  {'med_steps':>10} {'med_walls':>10} {'med_nfev':>10}")
    for (m, n), g in df.groupby(["method", "noise_pm"]):
        gc = g[g["is_saddle"] & g["fmax_loose"]]
        if not len(gc): continue
        print(f"{m:<18} {n:>5}  {gc['total_steps'].median():>10.0f} "
              f"{gc['wall_time_s'].median():>10.1f} {gc['n_func_evals'].median():>10.0f}")


if __name__ == "__main__":
    main()
