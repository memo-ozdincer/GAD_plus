#!/usr/bin/env python
"""Master figure set for IRC_COMPREHENSIVE_2026-04-20 updated with IRC data.

Produces:
  Per-method (5 methods):
    fig_method_{slug}_conv_line.pdf    — conv-rate line chart (method's native gate)
    fig_method_{slug}_irc_bars_topo    — IRC TOPO proportional stacked bars
  Main comparisons:
    fig_cmp_conv_5methods              — 5-method convergence rate line chart (old criterion)
    fig_cmp_irc_topo_5methods          — 5-method IRC TOPO-intended line chart (main narrative, uncluttered)
    fig_cmp_irc_all_classes_5methods   — APPENDIX: cluttered, all 3 outcomes per method (18 lines)
    fig_cmp_irc_rmsd_all_classes_5methods — same for RMSD
  Gate-vs-IRC overlap:
    fig_gate_vs_irc_{slug}             — per-method gate/IRC sample overlap
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)
NOISES = [10, 30, 50, 100, 150, 200]

C_INT  = "#2ca02c"
C_HALF = "#f0b432"
C_UN   = "#d62728"

# method_key -> (summary_paths, conv_col, irc_dir, color, marker, label)
METHODS = {
    "gad_eckart": (
        [f"/lustre07/scratch/memoozd/gadplus/runs/round2/summary_gad_dt003_{n}pm.parquet" for n in [10,30,50]]
      + [f"/lustre07/scratch/memoozd/gadplus/runs/round3/summary_gad_dt003_{n}pm.parquet" for n in [100,150,200]],
        "converged",
        "/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_allendpoints",
        "#1f77b4", "o", "GAD Eckart",
    ),
    "gad_no_eckart": (
        [f"/lustre07/scratch/memoozd/gadplus/runs/gad_no_eckart/summary_gad_dt003_no_eckart_{n}pm.parquet" for n in NOISES],
        "converged",
        "/lustre07/scratch/memoozd/gadplus/runs/irc_gad_no_eckart",
        "#17becf", "D", "GAD no-Eckart",
    ),
    "sella_carte_2k": (
        [f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/summary_sella_cartesian_eckart_fmax0p01_{n}pm.parquet" for n in NOISES],
        "conv_nneg1_fmax001",
        "/lustre07/scratch/memoozd/gadplus/runs/irc_sella_carte_2000",
        "#d62728", "s", "Sella cart+Eckart",
    ),
    "sella_cart_2k": (
        [f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/summary_sella_cartesian_fmax0p01_{n}pm.parquet" for n in NOISES],
        "conv_nneg1_fmax001",
        "/lustre07/scratch/memoozd/gadplus/runs/irc_sella_cart_2000",
        "#ff7f0e", "^", "Sella cart no-Eckart",
    ),
    "sella_int_2k": (
        [f"/lustre07/scratch/memoozd/gadplus/runs/sella_2000/summary_sella_internal_fmax0p01_{n}pm.parquet" for n in NOISES],
        "conv_nneg1_fmax001",
        "/lustre07/scratch/memoozd/gadplus/runs/irc_sella_int_2000",
        "#9467bd", "v", "Sella internal",
    ),
}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {name}")


def load_summary(paths, conv_col):
    """Return (noise -> (n, n_conv)) for given method."""
    d = {}
    for p, noise in zip(paths, NOISES):
        if not os.path.exists(p):
            d[noise] = (0, 0); continue
        r = duckdb.execute(f"SELECT COUNT(*), SUM(CASE WHEN {conv_col} THEN 1 ELSE 0 END) FROM '{p}'").fetchone()
        d[noise] = (r[0] or 0, r[1] or 0)
    return d


def load_irc(irc_dir):
    """Return df with noise_pm, intended, half_intended, topology_intended, topology_half_intended, sample_id."""
    return duckdb.execute(f"SELECT * FROM '{irc_dir}/*.parquet'").df() if os.path.exists(irc_dir) else None


# ---------------- Per-method: conv line + bar chart ----------------

def fig_conv_line(method_key):
    paths, conv_col, _, color, marker, label = METHODS[method_key]
    summary = load_summary(paths, conv_col)
    rates, ns = [], []
    for n in NOISES:
        tot, c = summary[n]
        if tot > 0:
            rates.append(100 * c / 300); ns.append(c)
        else:
            rates.append(np.nan); ns.append(0)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(NOISES, rates, "-", color=color, marker=marker, linewidth=2.4, markersize=10,
            markerfacecolor="white", markeredgewidth=2, label=label)
    for x, r, n in zip(NOISES, rates, ns):
        if not np.isnan(r):
            ax.annotate(f"{n}/300\n{r:.1f}%", (x, r), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("convergence rate (%)", fontsize=11)
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.3)
    save(fig, f"fig_method_{method_key}_conv_line")


def fig_irc_bars_topo(method_key):
    _, _, irc_dir, *_ = METHODS[method_key]
    irc = load_irc(irc_dir)
    if irc is None:
        return
    ns, pi, ph, pu = [], [], [], []
    labels = []
    for n in NOISES:
        g = irc[irc["noise_pm"] == n]
        total = len(g)
        if total == 0:
            continue
        ni = int(g["topology_intended"].sum())
        nh = int((g["topology_half_intended"] & ~g["topology_intended"]).sum())
        nu = total - ni - nh
        ns.append(total)
        pi.append(100*ni/total); ph.append(100*nh/total); pu.append(100*nu/total)
        labels.append(f"{n} pm")
    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.bar(x, pi, 0.65, label="intended", color=C_INT, edgecolor="white", linewidth=0.6)
    ax.bar(x, ph, 0.65, bottom=pi, label="half-intended", color=C_HALF, edgecolor="white", linewidth=0.6)
    ax.bar(x, pu, 0.65, bottom=np.array(pi)+np.array(ph), label="unintended", color=C_UN, edgecolor="white", linewidth=0.6)
    for i, (a, b, c) in enumerate(zip(pi, ph, pu)):
        if a > 6: ax.text(x[i], a/2, f"{a:.1f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        if b > 6: ax.text(x[i], a+b/2, f"{b:.1f}%", ha="center", va="center", fontsize=8.5, color="black")
        if c > 6: ax.text(x[i], a+b+c/2, f"{c:.1f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    for i, n in enumerate(ns):
        ax.text(x[i], 101, f"n={n}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("fraction of IRC runs (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    save(fig, f"fig_method_{method_key}_irc_bars_topo")


# ---------------- Main comparisons ----------------

def fig_cmp_conv_5methods():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method_key in METHODS:
        paths, conv_col, _, color, marker, label = METHODS[method_key]
        summary = load_summary(paths, conv_col)
        rates = []
        xs = []
        for n in NOISES:
            tot, c = summary[n]
            if tot > 0:
                rates.append(100 * c / 300); xs.append(n)
        ax.plot(xs, rates, "-", color=color, marker=marker, linewidth=2.1, markersize=8,
                markerfacecolor="white", markeredgewidth=1.8, label=label)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("convergence rate (%, native gate)", fontsize=11)
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    save(fig, "fig_cmp_conv_5methods")


def fig_cmp_irc_topo_5methods():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method_key in METHODS:
        _, _, irc_dir, color, marker, label = METHODS[method_key]
        irc = load_irc(irc_dir)
        if irc is None:
            continue
        xs, ys = [], []
        for n in NOISES:
            g = irc[irc["noise_pm"] == n]
            if len(g):
                xs.append(n); ys.append(100 * g["topology_intended"].mean())
        ax.plot(xs, ys, "-", color=color, marker=marker, linewidth=2.1, markersize=8,
                markerfacecolor="white", markeredgewidth=1.8, label=label)
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("IRC TOPO-intended rate (%)", fontsize=11)
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    save(fig, "fig_cmp_irc_topo_5methods")


def fig_cmp_irc_all_classes(criterion="topo"):
    """3 classes × 5 methods = 15 lines, clustered. Appendix."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    int_col = "topology_intended" if criterion == "topo" else "intended"
    half_col = "topology_half_intended" if criterion == "topo" else "half_intended"
    for method_key in METHODS:
        _, _, irc_dir, color, marker, label = METHODS[method_key]
        irc = load_irc(irc_dir)
        if irc is None:
            continue
        xs, pi, ph, pu = [], [], [], []
        for n in NOISES:
            g = irc[irc["noise_pm"] == n]
            if len(g):
                tot = len(g)
                ni = int(g[int_col].sum())
                nh = int((g[half_col] & ~g[int_col]).sum())
                nu = tot - ni - nh
                xs.append(n); pi.append(100*ni/tot); ph.append(100*nh/tot); pu.append(100*nu/tot)
        ax.plot(xs, pi, "-",  color=color, marker=marker, linewidth=2, markersize=7,
                markerfacecolor="white", markeredgewidth=1.5, label=f"{label} intended")
        ax.plot(xs, ph, "--", color=color, marker=marker, linewidth=1.1, markersize=5,
                markerfacecolor=color, alpha=0.75, label=f"{label} half")
        ax.plot(xs, pu, ":",  color=color, marker=marker, linewidth=1.1, markersize=5,
                markerfacecolor="white", alpha=0.75, label=f"{label} unintended")
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    crit_label = "TOPO" if criterion == "topo" else "RMSD"
    ax.set_ylabel(f"IRC {crit_label} outcome rate (%)", fontsize=11)
    ax.set_xticks(NOISES)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1, framealpha=0.95)
    save(fig, f"fig_cmp_irc_{criterion}_all_classes_5methods")


# ---------------- Gate vs IRC overlap per method ----------------

def fig_gate_overlap(method_key):
    paths, conv_col, irc_dir, color, marker, label = METHODS[method_key]
    summary = {}
    for p, noise in zip(paths, NOISES):
        if not os.path.exists(p):
            continue
        df = duckdb.execute(f"SELECT sample_id, {conv_col} AS conv FROM '{p}'").df()
        summary[noise] = df
    irc = load_irc(irc_dir)
    if irc is None:
        return
    x = np.arange(len(NOISES))
    mat = np.zeros((4, len(NOISES)))
    ns_arr = []
    for i, noise in enumerate(NOISES):
        s = summary.get(noise)
        r = irc[irc["noise_pm"] == noise][["sample_id", "topology_intended"]]
        if s is None or len(r) == 0:
            mat[:, i] = np.nan; ns_arr.append(0); continue
        m = s.merge(r, on="sample_id", how="outer")
        conv = m["conv"].fillna(False).astype(bool)
        topo = m["topology_intended"].fillna(False).astype(bool)
        n = len(m)
        both = int((conv & topo).sum())
        g_only = int((conv & ~topo).sum())
        i_only = int((~conv & topo).sum())
        nei = int((~conv & ~topo).sum())
        ns_arr.append(n)
        mat[:, i] = [100*both/n, 100*g_only/n, 100*i_only/n, 100*nei/n]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [C_INT, C_HALF, "#9467bd", C_UN]
    cats = ["both", "gate only", "IRC only", "neither"]
    bottom = np.zeros(len(NOISES))
    for i, (c, cat) in enumerate(zip(colors, cats)):
        vals = mat[i]
        ax.bar(x, np.nan_to_num(vals), 0.65, bottom=bottom, label=cat, color=c, edgecolor="white", linewidth=0.6)
        for xi, v in enumerate(vals):
            if not np.isnan(v) and v > 5:
                ax.text(xi, bottom[xi] + v/2, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        bottom = bottom + np.nan_to_num(vals)
    for xi, n in enumerate(ns_arr):
        if n > 0:
            ax.text(x[xi], 101, f"n={n}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{n} pm" for n in NOISES])
    ax.set_xlabel("TS noise (pm)", fontsize=11)
    ax.set_ylabel("fraction of samples (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    save(fig, f"fig_gate_vs_irc_{method_key}")


def main():
    plt.rcParams.update({"font.size": 10, "text.usetex": False})
    for method_key in METHODS:
        fig_conv_line(method_key)
        fig_irc_bars_topo(method_key)
        fig_gate_overlap(method_key)
    fig_cmp_conv_5methods()
    fig_cmp_irc_topo_5methods()
    fig_cmp_irc_all_classes("topo")
    fig_cmp_irc_all_classes("rmsd")
    print("Done.")


if __name__ == "__main__":
    main()
