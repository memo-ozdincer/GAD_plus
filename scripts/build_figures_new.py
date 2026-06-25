#!/usr/bin/env python
"""Build the manuscript figure suite into figures_new/.

Figure-rich by design: several variants per claim so the best can be chosen.
Data sources (all test-287, self-consistent):
  - fresh per-sample IRC parquets  -> intended/partial/ts_error/convergence
      runs/irc_{gad,hybrid,sella}_test287/
  - analysis_2026_04_29/threshold_sweep_2026_05_16.csv -> fmax plateau
  - analysis_2026_04_29/master_2026_05_11.csv          -> wall-time, steps
Provenance per figure printed at build time.
"""
from __future__ import annotations
import glob, re, os
import duckdb, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/lustre06/project/6033559/memoozd/GAD_plus"
RUNS = "/lustre07/scratch/memoozd/gadplus/runs"
CSV  = f"{ROOT}/analysis_2026_04_29"
OUT  = f"{ROOT}/figures_new"
os.makedirs(OUT, exist_ok=True)
NOISES = [10, 30, 50, 100, 150, 200]

# colour-blind safe (Brewer Dark2)
C = {"GAD": "#1b9e77", "Hybrid": "#7570b3", "Sella": "#d95f02"}
TIERC = {"intended": "#1b9e77", "partial": "#e6ab02", "ts_error": "#9e9e9e"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": False})

IRC = {"GAD": f"{RUNS}/irc_gad_test287/*.parquet",
       "Hybrid": f"{RUNS}/irc_hybrid_test287/*/*.parquet",
       "Sella": f"{RUNS}/irc_sella_test287/*.parquet"}

def load_persample():
    """per (method,noise): n, conv, intended, partial, ts_error (counts + pct)."""
    rows = []
    for m, pat in IRC.items():
        for f in glob.glob(pat):
            npm = int(re.search(r"(\d+)pm", f).group(1))
            d = duckdb.execute(f"""
              SELECT {npm} noise, COUNT(*) n,
                SUM(CASE WHEN source_gad_converged THEN 1 ELSE 0 END) conv,
                SUM(CASE WHEN topology_intended THEN 1 ELSE 0 END) intended,
                SUM(CASE WHEN NOT topology_intended AND topology_half_intended THEN 1 ELSE 0 END) partial_t,
                SUM(CASE WHEN NOT topology_intended AND NOT topology_half_intended AND NOT source_gad_converged THEN 1 ELSE 0 END) ts_error
              FROM '{f}'""").df()
            d["method"] = m
            rows.append(d)
    df = pd.concat(rows).groupby(["method", "noise"], as_index=False).sum()
    df = df.rename(columns={"partial_t": "partial"})
    for k in ["conv", "intended", "partial", "ts_error"]:
        df[k + "_pct"] = 100 * df[k] / df["n"]
    return df.sort_values(["method", "noise"])

D = load_persample()
def series(method, col):
    s = D[D.method == method].set_index("noise")[col]
    return [s.get(n, np.nan) for n in NOISES]

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close(fig); print("  wrote", name)

# ---------------------------------------------------------------- A. intended
def figA():
    # A1 single-panel intended
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ["GAD", "Hybrid", "Sella"]:
        ax.plot(NOISES, series(m, "intended_pct"), "-o", color=C[m], lw=2.2, ms=6, label=m)
    ax.set(xlabel="initial-guess noise (pm)", ylabel="intended success rate (%)",
           title="IRC-validated intended success vs initial guess (test-287)")
    ax.legend(); save(fig, "fig_intended_success_single")

    # A2 two-panel intended + convergence
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, col, ttl in [(axes[0], "intended_pct", "(a) intended success rate"),
                         (axes[1], "conv_pct", "(b) convergence rate")]:
        for m in ["GAD", "Hybrid", "Sella"]:
            ax.plot(NOISES, series(m, col), "-o", color=C[m], lw=2.2, ms=5, label=m)
        ax.set(xlabel="noise (pm)", title=ttl); ax.set_ylim(0, 100)
    axes[0].set_ylabel("%"); axes[0].legend()
    save(fig, "fig_intended_success")  # the name the manuscript references

    # A3 intended with GAD-Sella delta shaded
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    g, s = np.array(series("GAD", "intended_pct")), np.array(series("Sella", "intended_pct"))
    ax.fill_between(NOISES, s, g, where=g >= s, color=C["GAD"], alpha=0.15)
    for m in ["GAD", "Hybrid", "Sella"]:
        ax.plot(NOISES, series(m, "intended_pct"), "-o", color=C[m], lw=2.2, ms=6, label=m)
    for x, gg, ss in zip(NOISES, g, s):
        if gg - ss >= 3:
            ax.annotate(f"+{gg-ss:.0f}", (x, (gg+ss)/2), color=C["GAD"], fontsize=9, ha="center")
    ax.set(xlabel="initial-guess noise (pm)", ylabel="intended success rate (%)",
           title="GAD advantage grows with noise (Δ vs Sella shaded)")
    ax.legend(); save(fig, "fig_intended_success_delta")

# ---------------------------------------------------------------- B. outcomes
def figB():
    bands = ["intended", "partial", "ts_error"]
    labels = {"intended": "intended (both endpoints)", "partial": "partial (one endpoint)",
              "ts_error": "failed to converge"}
    # B1 100%-stacked, faceted by method
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, m in zip(axes, ["GAD", "Hybrid", "Sella"]):
        bottom = np.zeros(len(NOISES))
        for b in bands:
            vals = np.array(series(m, b + "_pct"))
            ax.bar([str(n) for n in NOISES], vals, bottom=bottom, color=TIERC[b], width=0.8)
            bottom += vals
        ax.set(title=m, xlabel="noise (pm)"); ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of 287 reactions")
    fig.legend(handles=[Patch(color=TIERC[b], label=labels[b]) for b in bands],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig_outcome_stacked")  # manuscript name

    # B2 absolute counts, faceted
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, m in zip(axes, ["GAD", "Hybrid", "Sella"]):
        bottom = np.zeros(len(NOISES))
        for b in bands:
            vals = np.array(series(m, b))
            ax.bar([str(n) for n in NOISES], vals, bottom=bottom, color=TIERC[b], width=0.8)
            bottom += vals
        ax.set(title=m, xlabel="noise (pm)")
    axes[0].set_ylabel("count (/287)")
    fig.legend(handles=[Patch(color=TIERC[b], label=labels[b]) for b in bands],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig_outcome_stacked_counts")

    # B3 grouped intended-only bars (method side-by-side per noise)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    w = 0.27; x = np.arange(len(NOISES))
    for i, m in enumerate(["GAD", "Hybrid", "Sella"]):
        ax.bar(x + (i-1)*w, series(m, "intended_pct"), w, color=C[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in NOISES])
    ax.set(xlabel="noise (pm)", ylabel="intended success rate (%)",
           title="Intended success by method and noise")
    ax.legend(); save(fig, "fig_outcome_grouped_intended")

# ---------------------------------------------------------------- C. plateau
def figC():
    ts = pd.read_csv(f"{CSV}/threshold_sweep_2026_05_16.csv")
    thr = [("fmax_005", 0.05), ("fmax_023", 0.023), ("fmax_010", 0.01),
           ("fmax_005t", 0.005), ("fmax_001", 0.001)]
    cfg = {"GAD": "GAD dt=0.005", "Hybrid": "Hybrid damped Eckart eig tr=0.05",
           "Sella": "Sella cartesian Eckart untuned Hess.Freq.=1"}
    def curve(ax, npm):
        for m, c in cfg.items():
            row = ts[(ts.method == c) & (ts.noise_pm == npm)]
            if row.empty: continue
            ax.plot([t[1] for t in thr], [row.iloc[0][t[0]] for t in thr],
                    "-o", color=C[m], lw=2, ms=5, label=m)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.axvspan(0.005, 0.001, color="red", alpha=0.06)
        ax.set(xlabel=r"$F_{max}$ threshold (eV/Å, tighter →)", title=f"{npm} pm")
    # C1 two-panel 50 & 150
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    curve(axes[0], 50); curve(axes[1], 150)
    axes[0].set_ylabel("convergence rate (%)"); axes[0].legend()
    save(fig, "fig_fmax_plateau")  # manuscript name
    # C2 small-multiples all 6 noise
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, npm in zip(axes.ravel(), NOISES):
        curve(ax, npm)
    axes[0, 0].set_ylabel("conv (%)"); axes[1, 0].set_ylabel("conv (%)")
    axes[0, 0].legend(fontsize=8)
    save(fig, "fig_fmax_plateau_grid")

# ---------------------------------------------------------------- D. cost
def figD():
    m = pd.read_csv(f"{CSV}/master_2026_05_11.csv")
    cfg = {"GAD": "GAD dt=0.005", "Hybrid": "Hybrid damped Eckart eig tr=0.05",
           "Sella": "Sella cartesian Eckart untuned Hess.Freq.=1"}
    def col(method, c):
        sub = m[m.config == cfg[method]].set_index("noise_pm")
        return [sub[c].get(n, np.nan) for n in NOISES]
    # D1 wall/conv vs noise (log y)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for k in cfg:
        ax.plot(NOISES, col(k, "wall_per_conv"), "-o", color=C[k], lw=2.2, ms=6, label=k)
    ax.set_yscale("log")
    ax.set(xlabel="noise (pm)", ylabel="wall-time per converged TS (s)",
           title="Cost: wall-time per converged TS")
    ax.legend(); save(fig, "fig_walltime")  # manuscript name
    # D2 Pareto wall vs intended (use fresh intended)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for k in cfg:
        wall = col(k, "wall_per_conv"); inten = series(k, "intended_pct")
        ax.plot(wall, inten, "-o", color=C[k], lw=1.4, ms=7, label=k, alpha=0.9)
        for w_, i_, npm in zip(wall, inten, NOISES):
            ax.annotate(f"{npm}", (w_, i_), fontsize=7, color=C[k], xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlabel="wall-time per converged TS (s, log)", ylabel="intended success (%)",
           title="Pareto: cost vs chemical reliability (label = noise pm)")
    ax.legend(); save(fig, "fig_pareto")
    # D3 median steps
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for k in cfg:
        ax.plot(NOISES, col(k, "med_step"), "-o", color=C[k], lw=2.2, ms=6, label=k)
    ax.set_yscale("log")
    ax.set(xlabel="noise (pm)", ylabel="median converged-step count",
           title="Steps to convergence (cheap GAD steps vs few Newton steps)")
    ax.legend(); save(fig, "fig_steps")

# ---------------------------------------------------------------- F. conv != chem
def figF():
    m = pd.read_csv(f"{CSV}/master_2026_05_11.csv")
    d1, d3 = "Sella cartesian Eckart untuned Hess.Freq.=1", "Sella cartesian Eckart untuned Hess.Freq.=3"
    def g(c, col):
        s = m[m.config == c].set_index("noise_pm"); return [s[col].get(n, np.nan) for n in NOISES]
    fig, ax = plt.subplots(figsize=(8, 4.3)); x = np.arange(len(NOISES)); w = 0.2
    ax.bar(x-1.5*w, g(d3, "conv_pct"), w, color="#9ecae1", label="d=3 convergence")
    ax.bar(x-0.5*w, g(d1, "conv_pct"), w, color="#3182bd", label="d=1 convergence")
    ax.bar(x+0.5*w, g(d3, "topo_pct"), w, color="#fdae6b", label="d=3 intended")
    ax.bar(x+1.5*w, g(d1, "topo_pct"), w, color="#e6550d", label="d=1 intended")
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in NOISES])
    ax.set(xlabel="noise (pm)", ylabel="%",
           title="Convergence ≠ chemistry: d=3 converges more, d=1 is more intended")
    ax.legend(ncol=2, fontsize=9); save(fig, "fig_d1_vs_d3")

if __name__ == "__main__":
    print("building figures_new/ ...")
    figA(); figB(); figC(); figD(); figF()
    print("done ->", OUT)
