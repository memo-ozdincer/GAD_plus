#!/usr/bin/env python
"""Deep analysis of Eckart vs no-Eckart for the appendix of IRC_COMPREHENSIVE_2026-04-20.

Outputs numerical findings and figures for the Eckart deep-dive section.
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

from plotting_style import apply_plot_style, palette_color

apply_plot_style()

NOISES = [10, 30, 50, 100, 150, 200]
OUT = Path("/lustre06/project/6033559/memoozd/GAD_plus/figures")
OUT.mkdir(exist_ok=True)

# Use the canonical fmax-criterion GAD Eckart data + matching no-Eckart data
GAD_E = "/lustre07/scratch/memoozd/gadplus/runs/gad_eckart_fmax"          # GAD Eckart (fmax<0.01)
GAD_N = "/lustre07/scratch/memoozd/gadplus/runs/gad_no_eckart"            # GAD no-Eckart (fmax<0.01)
IRC_E = "/lustre07/scratch/memoozd/gadplus/runs/irc_gad_eckart_fmax"
IRC_N = "/lustre07/scratch/memoozd/gadplus/runs/irc_gad_no_eckart"

LINES = []
def log(s=""): print(s); LINES.append(s)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def load_summary(d, method, noise):
    p = f"{d}/summary_{method}_{noise}pm.parquet"
    return duckdb.execute(f"SELECT * FROM '{p}'").df() if os.path.exists(p) else None


# =================================================================
# A. Per-sample 4-quadrant agreement (converge under both? only one? neither?)
# =================================================================

log("="*78)
log("A. Per-sample agreement: GAD Eckart fmax vs GAD no-Eckart fmax")
log("="*78)
log()
log(f"{'noise':>6s} {'BOTH':>6s} {'Eonly':>6s} {'Nonly':>6s} {'NEITHER':>8s} {'agreement%':>12s}")
agree_data = []
for noise in NOISES:
    e = load_summary(GAD_E, "gad_dt003_fmax", noise)
    n = load_summary(GAD_N, "gad_dt003_no_eckart", noise)
    if e is None or n is None: continue
    m = e[["sample_id","converged"]].rename(columns={"converged":"e_conv"}).merge(
        n[["sample_id","converged"]].rename(columns={"converged":"n_conv"}),
        on="sample_id"
    )
    e_conv = m["e_conv"].astype(bool)
    n_conv = m["n_conv"].astype(bool)
    both = (e_conv & n_conv).sum()
    e_only = (e_conv & ~n_conv).sum()
    n_only = (~e_conv & n_conv).sum()
    neither = (~e_conv & ~n_conv).sum()
    agreement = (both + neither) / len(m)
    agree_data.append((noise, both, e_only, n_only, neither, agreement))
    log(f"{noise:>5d}pm {both:>6d} {e_only:>6d} {n_only:>6d} {neither:>8d} {100*agreement:>11.1f}%")

# =================================================================
# B. Step efficiency (median converged_step on shared converged samples)
# =================================================================

log()
log("="*78)
log("B. Step efficiency (samples that BOTH methods converged on)")
log("="*78)
log()
log(f"{'noise':>6s} {'n_shared':>9s} {'E med':>8s} {'N med':>8s} {'ratio':>7s} {'E p25-p75':>14s} {'N p25-p75':>14s}")
for noise in NOISES:
    e = load_summary(GAD_E, "gad_dt003_fmax", noise)
    n = load_summary(GAD_N, "gad_dt003_no_eckart", noise)
    if e is None or n is None: continue
    m = e[["sample_id","converged","converged_step"]].rename(columns={"converged":"e_conv","converged_step":"e_step"}).merge(
        n[["sample_id","converged","converged_step"]].rename(columns={"converged":"n_conv","converged_step":"n_step"}),
        on="sample_id"
    )
    shared = m[(m["e_conv"]) & (m["n_conv"])]
    if len(shared) == 0: continue
    e_med = shared["e_step"].median()
    n_med = shared["n_step"].median()
    ratio = n_med / e_med if e_med > 0 else float('nan')
    e_p25, e_p75 = shared["e_step"].quantile([0.25, 0.75])
    n_p25, n_p75 = shared["n_step"].quantile([0.25, 0.75])
    log(f"{noise:>5d}pm {len(shared):>9d} {e_med:>8.0f} {n_med:>8.0f} {ratio:>6.2f}x  [{e_p25:.0f},{e_p75:.0f}]   [{n_p25:.0f},{n_p75:.0f}]")

# =================================================================
# C. Asymmetry of "only converges in X" — which TSs do Eckart-only vs no-Eckart-only find?
# =================================================================

log()
log("="*78)
log("C. Final-state characterization for samples converging only by ONE method")
log("="*78)
log()
log("(comparing final_force_norm, final_eig0 for E-only vs N-only converged samples)")
log()
for noise in [10, 50, 150, 200]:
    e = load_summary(GAD_E, "gad_dt003_fmax", noise)
    n = load_summary(GAD_N, "gad_dt003_no_eckart", noise)
    if e is None or n is None: continue
    m = e[["sample_id","converged","final_force_norm","final_eig0"]].rename(
        columns={"converged":"e_conv","final_force_norm":"e_fn","final_eig0":"e_eig0"}).merge(
        n[["sample_id","converged","final_force_norm","final_eig0"]].rename(
            columns={"converged":"n_conv","final_force_norm":"n_fn","final_eig0":"n_eig0"}),
        on="sample_id")
    only_e = m[m["e_conv"] & ~m["n_conv"]]
    only_n = m[~m["e_conv"] & m["n_conv"]]
    log(f"  {noise}pm:  E-only n={len(only_e)} (median |eig0| {only_e['e_eig0'].abs().median():.3f}; n-method final_force {only_e['n_fn'].median():.4f})")
    log(f"            N-only n={len(only_n)} (median |eig0| {only_n['n_eig0'].abs().median():.3f}; e-method final_force {only_n['e_fn'].median():.4f})")

# =================================================================
# D. Final TS quality — same-sample comparison (both converged)
# =================================================================

log()
log("="*78)
log("D. Final TS quality: are Eckart-found and no-Eckart-found TSs the same?")
log("="*78)
log("(For samples converged by both: median final_eig0 magnitude, RMSD between final coords)")
log()
log(f"{'noise':>6s} {'n':>5s} {'|eig0| E':>10s} {'|eig0| N':>10s} {'final coord RMSD':>18s}")
for noise in NOISES:
    e = load_summary(GAD_E, "gad_dt003_fmax", noise)
    n = load_summary(GAD_N, "gad_dt003_no_eckart", noise)
    if e is None or n is None: continue
    m = e[["sample_id","converged","final_eig0"]].rename(columns={"converged":"e_conv","final_eig0":"e_eig0"}).merge(
        n[["sample_id","converged","final_eig0"]].rename(columns={"converged":"n_conv","final_eig0":"n_eig0"}),
        on="sample_id")
    shared = m[m["e_conv"] & m["n_conv"]]
    e_abs_eig = shared["e_eig0"].abs().median()
    n_abs_eig = shared["n_eig0"].abs().median()
    log(f"{noise:>5d}pm {len(shared):>5d} {e_abs_eig:>10.4f} {n_abs_eig:>10.4f} {'(see fig)':>18s}")

# =================================================================
# E. IRC outcome agreement
# =================================================================

log()
log("="*78)
log("E. IRC TOPO outcome agreement between Eckart and no-Eckart endpoints")
log("="*78)
log()
log(f"{'noise':>6s} {'BOTH':>5s} {'Eonly':>6s} {'Nonly':>6s} {'NEITHER':>8s} {'agree%':>9s}")
for noise in NOISES:
    e_irc = duckdb.execute(f"SELECT sample_id, topology_intended FROM '{IRC_E}/*.parquet' WHERE noise_pm = {noise}").df()
    n_irc = duckdb.execute(f"SELECT sample_id, topology_intended FROM '{IRC_N}/*.parquet' WHERE noise_pm = {noise}").df()
    m = e_irc.rename(columns={"topology_intended":"e_topo"}).merge(
        n_irc.rename(columns={"topology_intended":"n_topo"}), on="sample_id")
    e_t = m["e_topo"].astype(bool); n_t = m["n_topo"].astype(bool)
    both = (e_t & n_t).sum()
    e_only = (e_t & ~n_t).sum()
    n_only = (~e_t & n_t).sum()
    neither = (~e_t & ~n_t).sum()
    log(f"{noise:>5d}pm {both:>5d} {e_only:>6d} {n_only:>6d} {neither:>8d} {100*(both+neither)/len(m):>8.1f}%")

# =================================================================
# F. Step-distribution figures
# =================================================================

log()
log("="*78)
log("F. Generating step-distribution figures")
log("="*78)
log()

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for ax, noise in zip(axes.flat, NOISES):
    e = load_summary(GAD_E, "gad_dt003_fmax", noise)
    n = load_summary(GAD_N, "gad_dt003_no_eckart", noise)
    if e is None or n is None: continue
    e_steps = e[e["converged"]]["converged_step"].dropna()
    n_steps = n[n["converged"]]["converged_step"].dropna()
    bins = np.logspace(0.5, np.log10(2000), 40)
    ax.hist(e_steps, bins=bins, alpha=0.6, color=palette_color(0), label=f"Eckart (n={len(e_steps)})")
    ax.hist(n_steps, bins=bins, alpha=0.6, color=palette_color(9), label=f"no-Eckart (n={len(n_steps)})")
    ax.set_xscale("log")
    ax.set_title(f"{noise} pm", fontsize=10)
    ax.set_xlabel("converged_step (log)")
    if ax in axes[:,0]:
        ax.set_ylabel("# samples")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("Step distribution of converged samples — GAD Eckart vs no-Eckart", fontsize=11)
fig.tight_layout()
save(fig, "fig_eckart_step_distributions")
log("wrote fig_eckart_step_distributions")

# Step ratio distribution (per shared sample)
fig, ax = plt.subplots(figsize=(8, 5))
ratios_by_noise = []
for noise in NOISES:
    e = load_summary(GAD_E, "gad_dt003_fmax", noise)
    n = load_summary(GAD_N, "gad_dt003_no_eckart", noise)
    if e is None or n is None: continue
    m = e[["sample_id","converged","converged_step"]].rename(
        columns={"converged":"e_conv","converged_step":"e_step"}).merge(
        n[["sample_id","converged","converged_step"]].rename(
            columns={"converged":"n_conv","converged_step":"n_step"}),
        on="sample_id")
    shared = m[m["e_conv"] & m["n_conv"]]
    ratios_by_noise.append(shared["n_step"] / shared["e_step"])

bp = ax.boxplot(ratios_by_noise, positions=range(len(NOISES)), widths=0.6, patch_artist=True, showfliers=False)
for p in bp["boxes"]: p.set_facecolor(palette_color(9)); p.set_alpha(0.7)
ax.axhline(1, color=palette_color(7), lw=0.8, linestyle="--")
ax.axhline(1.33, color=palette_color(3), lw=1, linestyle=":", label="1.33× (median ratio)")
ax.set_xticks(range(len(NOISES)))
ax.set_xticklabels([f"{n} pm" for n in NOISES])
ax.set_xlabel("TS noise")
ax.set_ylabel("step ratio: no-Eckart / Eckart")
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")
save(fig, "fig_eckart_step_ratio")
log("wrote fig_eckart_step_ratio")

# Per-sample agreement bar chart
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(agree_data))
both = [d[1] for d in agree_data]
e_only = [d[2] for d in agree_data]
n_only = [d[3] for d in agree_data]
neither = [d[4] for d in agree_data]
ax.bar(x, both, label="both converge", color=palette_color(2))
ax.bar(x, e_only, bottom=both, label="Eckart only", color=palette_color(0))
ax.bar(x, n_only, bottom=np.array(both)+np.array(e_only), label="no-Eckart only", color=palette_color(9))
ax.bar(x, neither, bottom=np.array(both)+np.array(e_only)+np.array(n_only), label="neither", color=palette_color(3))
# skip per-bar annotations — counts shown in table
ax.set_xticks(x)
ax.set_xticklabels([f"{d[0]} pm" for d in agree_data])
ax.set_ylim(0, 305)
ax.set_xlabel("TS noise")
ax.set_ylabel("# samples")
ax.legend(loc="lower left", fontsize=9)
save(fig, "fig_eckart_per_sample_agreement")
log("wrote fig_eckart_per_sample_agreement")

# ---------------- fmax vs force_norm (briefer) ----------------

log()
log("="*78)
log("G. fmax vs force_norm criterion — brief analysis")
log("="*78)
log()

GAD_FN = [(f"/lustre07/scratch/memoozd/gadplus/runs/round2/summary_gad_dt003_{n}pm.parquet" if n <= 50
          else f"/lustre07/scratch/memoozd/gadplus/runs/round3/summary_gad_dt003_{n}pm.parquet")
          for n in NOISES]

log(f"{'noise':>6s} {'force_norm':>12s} {'fmax':>9s} {'drop':>7s}  fmax/force_norm ratio (median)")
for i, noise in enumerate(NOISES):
    fn_path = GAD_FN[i]
    fmax_path = f"{GAD_E}/summary_gad_dt003_fmax_{noise}pm.parquet"
    if not os.path.exists(fn_path) or not os.path.exists(fmax_path): continue
    fn_df = duckdb.execute(f"SELECT * FROM '{fn_path}'").df()
    fmax_df = duckdb.execute(f"SELECT * FROM '{fmax_path}'").df()
    fn_rate = 100 * fn_df["converged"].sum() / 300
    fmax_rate = 100 * fmax_df["converged"].sum() / 300
    drop = fn_rate - fmax_rate
    # Median ratio fmax/force_norm at converged geometries
    log(f"{noise:>5d}pm  {fn_rate:>10.1f}%  {fmax_rate:>7.1f}%  {drop:>+5.1f} pp")

# fmax vs force_norm figure
fig, ax = plt.subplots(figsize=(8, 5))
fn_rates = []
fmax_rates = []
for i, noise in enumerate(NOISES):
    fn_path = GAD_FN[i]
    fmax_path = f"{GAD_E}/summary_gad_dt003_fmax_{noise}pm.parquet"
    if not os.path.exists(fn_path) or not os.path.exists(fmax_path): continue
    fn = duckdb.execute(f"SELECT SUM(CASE WHEN converged THEN 1 ELSE 0 END), COUNT(*) FROM '{fn_path}'").fetchone()
    fmax_v = duckdb.execute(f"SELECT SUM(CASE WHEN converged THEN 1 ELSE 0 END), COUNT(*) FROM '{fmax_path}'").fetchone()
    fn_rates.append(100*fn[0]/300)
    fmax_rates.append(100*fmax_v[0]/300)
ax.plot(NOISES, fn_rates, "o-", color=palette_color(0), linewidth=2, markersize=8, markerfacecolor="white", markeredgewidth=2, label="force_norm criterion (looser)")
ax.plot(NOISES, fmax_rates, "s-", color=palette_color(3), linewidth=2, markersize=8, markerfacecolor="white", markeredgewidth=2, label="fmax criterion (stricter, canonical)")
for x, y1, y2 in zip(NOISES, fn_rates, fmax_rates):
    ax.annotate(f"{y1:.1f}", (x, y1), xytext=(8, 4), textcoords="offset points", fontsize=8, color=palette_color(0))
    ax.annotate(f"{y2:.1f}", (x, y2), xytext=(8, -10), textcoords="offset points", fontsize=8, color=palette_color(3))
ax.set_xlabel("TS noise (pm)"); ax.set_ylabel("convergence rate (%)")
ax.set_xticks(NOISES); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=10)
save(fig, "fig_fmax_vs_fnorm")
log("wrote fig_fmax_vs_fnorm")

with open("/tmp/eckart_deep_analysis.txt","w") as f: f.write("\n".join(LINES))
print("\nReport: /tmp/eckart_deep_analysis.txt")
