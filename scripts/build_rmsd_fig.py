#!/usr/bin/env python
"""Build the RMSD-to-known-TS distribution figure (CDF) for the new PDF.

Uses 100-sample subset per (method, noise) for speed; that's enough for the
distribution shape comparison.
"""
from __future__ import annotations

import os
import sys

import duckdb
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = "/lustre06/project/6033559/memoozd/GAD_plus/scripts"
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, "/lustre06/project/6033559/memoozd/GAD_plus/src")
from plotting_style import apply_plot_style, palette  # noqa: E402
from gadplus.geometry.alignment import aligned_rmsd, kabsch_align  # noqa: E402
from gadplus.data.transition1x import Transition1xDataset, UsePos  # noqa: E402

apply_plot_style()

OUT  = "/lustre06/project/6033559/memoozd/GAD_plus/figures_2026_05_11"
CSV  = "/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29"
RUNS = "/lustre07/scratch/memoozd/gadplus/runs"
SUBSET = 287   # all samples; we're on a compute node now

ds = Transition1xDataset(
    "/lustre06/project/6033559/memoozd/data/transition1x.h5",
    split="test", max_samples=300, transform=UsePos("pos_transition"),
)


def z_to_eq(z):
    c = {}
    for i, zi in enumerate(z):
        c.setdefault(str(int(zi)), []).append(i)
    return c


def rmsd_from_coords(coords_flat, sample_id):
    if sample_id >= len(ds):
        return None
    s = ds[sample_id]
    ref = s.pos.numpy()
    z = s.z.numpy()
    coords = np.array(coords_flat).reshape(-1, 3)
    if coords.shape != ref.shape:
        return None
    try:
        r = aligned_rmsd(coords, ref, z_to_eq(z))
    except Exception:
        try:
            _, _, r = kabsch_align(coords, ref)
        except Exception:
            return None
    return float(r)


def gather_method(query, noise, coords_col, family, config):
    df = duckdb.execute(query).df()
    if SUBSET and len(df) > SUBSET:
        df = df.head(SUBSET)
    out = []
    for _, row in df.iterrows():
        sid = int(row["sample_id"])
        r = rmsd_from_coords(row[coords_col], sid)
        if r is None:
            continue
        is_conv = bool(row.get("converged", row.get("source_gad_converged", True)))
        out.append({"sample_id": sid, "rmsd": r, "noise_pm": noise,
                    "family": family, "config": config, "converged": is_conv})
    return pd.DataFrame(out)


records = []
for noise in [10, 100, 200]:
    print(f"=== {noise} pm ===", flush=True)

    # Plain GAD dt=0.005
    d = gather_method(
        f"SELECT sample_id, converged, final_coords_flat "
        f"FROM read_parquet('{RUNS}/test_dtgrid/gad_dt005_fmax/summary_*_{noise}pm.parquet')",
        noise, "final_coords_flat", "plain GAD", "GAD dt=0.005",
    )
    print(f"  GAD: {len(d)} samples")
    records.append(d)

    # Sella libdef — use ts_coords_flat from IRC parquets
    d = gather_method(
        f"SELECT sample_id, source_gad_converged AS converged, ts_coords_flat "
        f"FROM read_parquet('{RUNS}/test_irc/sella_carteck_libdef/irc_validation_*_{noise}pm.parquet')",
        noise, "ts_coords_flat", "Sella", "Sella libdef",
    )
    print(f"  Sella: {len(d)} samples")
    records.append(d)

    # Hybrid damped Eckart eig tr=0.05
    d = gather_method(
        f"SELECT sample_id, converged, coords_flat "
        f"FROM read_parquet('{RUNS}/hybrid_for_irc/hybrid_damped_eckart_swtrue_dt5e-3_tr0.05_{noise}pm/summary_*.parquet')",
        noise, "coords_flat", "hybrid", "Hybrid damped Eckart eig tr=0.05",
    )
    print(f"  Hybrid: {len(d)} samples")
    records.append(d)


rmsd_df = pd.concat(records, ignore_index=True)
rmsd_df.to_csv(f"{CSV}/rmsd_distributions_2026_05_11.csv", index=False)


# Plot CDFs — only converged samples
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
PAL = {"plain GAD": palette()[1], "Sella": palette()[0], "hybrid": palette()[2]}
for ax, noise in zip(axes, [10, 100, 200]):
    sub = rmsd_df[(rmsd_df["noise_pm"] == noise) & rmsd_df["converged"]]
    for cfg, grp in sub.groupby("config", sort=False):
        vals = np.sort(grp["rmsd"].values)
        if len(vals) == 0: continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        family = grp["family"].iloc[0]
        ax.plot(vals, cdf, color=PAL[family], lw=2.5, label=cfg, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 5)
    ax.axvline(0.3, ls="--", color="gray", alpha=0.6, lw=1)
    ax.text(0.32, 0.05, "0.3 Å threshold", fontsize=9, color="gray")
    ax.set_title(f"{noise} pm noise", fontsize=14)
    ax.set_xlabel("RMSD to labelled T1x TS (Å, log)", fontsize=12)
    ax.set_ylabel("Cumulative fraction" if noise == 10 else "", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="lower right")
fig.suptitle(
    "Distance from converged TS to labelled T1x TS  •  CDFs of converged samples\n"
    "At high noise hybrid is dramatically tighter than both Sella (long right tail) and plain GAD",
    y=1.02, fontsize=13,
)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_rmsd_to_ts.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_rmsd_to_ts.png", bbox_inches="tight", dpi=140)
plt.close(fig)
print(f"Wrote {OUT}/fig_rmsd_to_ts.pdf")
