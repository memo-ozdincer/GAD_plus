#!/usr/bin/env python
"""Build the manuscript figure suite into figures_new/.

Metric naming (matches paper.tex):
  - IRC convergence            = forward+backward IRC graph-match both endpoints
  - Fmax/nneg convergence      = n_neg==1 AND fmax<tau (reached *a* saddle)

Data sources (all test-287, self-consistent):
  - fresh per-sample IRC parquets  -> intended/partial/ts_error/convergence
      runs/irc_{gad,hybrid,sella}_test287/
  - analysis_2026_04_29/threshold_sweep_2026_05_16.csv -> fmax plateau
  - analysis_2026_04_29/master_2026_05_11.csv          -> wall-time, steps
  - analysis_2026_04_29/reactant_0pm_2026_05_16.csv    -> reactant-start scope
  - paper Table (RMSD med/p95)                         -> rmsd vs noise
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

# canonical metric labels
IRC_LBL = "IRC convergence (%)"
FN_LBL  = r"$F_\mathrm{max}/n_\mathrm{neg}$ convergence (%)"
FN_SHORT = r"$F_\mathrm{max}/n_\mathrm{neg}$ conv (%)"

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

# ---------------------------------------------------------------- A. IRC convergence
def figA():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ["GAD", "Hybrid", "Sella"]:
        ax.plot(NOISES, series(m, "intended_pct"), "-o", color=C[m], lw=2.2, ms=6, label=m)
    ax.set(xlabel="initial-guess noise (pm)", ylabel=IRC_LBL,
           title="IRC convergence vs initial guess (test-287)")
    ax.legend(); save(fig, "fig_intended_success_single")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, col, ttl, yl in [(axes[0], "intended_pct", "(a) IRC convergence", IRC_LBL),
                             (axes[1], "conv_pct", "(b) "+FN_LBL.replace(" (%)",""), FN_LBL)]:
        for m in ["GAD", "Hybrid", "Sella"]:
            ax.plot(NOISES, series(m, col), "-o", color=C[m], lw=2.2, ms=5, label=m)
        ax.set(xlabel="noise (pm)", title=ttl, ylabel=yl); ax.set_ylim(0, 100)
    axes[0].legend()
    save(fig, "fig_intended_success")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    g, s = np.array(series("GAD", "intended_pct")), np.array(series("Sella", "intended_pct"))
    ax.fill_between(NOISES, s, g, where=g >= s, color=C["GAD"], alpha=0.15)
    for m in ["GAD", "Hybrid", "Sella"]:
        ax.plot(NOISES, series(m, "intended_pct"), "-o", color=C[m], lw=2.2, ms=6, label=m)
    for x, gg, ss in zip(NOISES, g, s):
        if gg - ss >= 3:
            ax.annotate(f"+{gg-ss:.0f}", (x, (gg+ss)/2), color=C["GAD"], fontsize=9, ha="center")
    ax.set(xlabel="initial-guess noise (pm)", ylabel=IRC_LBL,
           title="GAD IRC-convergence advantage grows with noise (Δ vs Sella shaded)")
    ax.legend(); save(fig, "fig_intended_success_delta")

# ---------------------------------------------------------------- A'. Fmax/nneg convergence
def figSC():
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m in ["GAD", "Hybrid", "Sella"]:
        ax.plot(NOISES, series(m, "conv_pct"), "-o", color=C[m], lw=2.2, ms=6, label=m)
    ax.set(xlabel="initial-guess noise (pm)", ylabel=FN_LBL,
           title=r"Secondary metric: $F_\mathrm{max}/n_\mathrm{neg}$ convergence vs noise")
    ax.legend(); save(fig, "fig_saddle_convergence")

# ---------------------------------------------------------------- B. outcomes
def figB():
    bands = ["intended", "partial", "ts_error"]
    labels = {"intended": "intended (both endpoints)", "partial": "partial (one endpoint)",
              "ts_error": "failed to converge"}
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
    save(fig, "fig_outcome_stacked")

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

    fig, ax = plt.subplots(figsize=(8, 4.3))
    w = 0.27; x = np.arange(len(NOISES))
    for i, m in enumerate(["GAD", "Hybrid", "Sella"]):
        ax.bar(x + (i-1)*w, series(m, "intended_pct"), w, color=C[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in NOISES])
    ax.set(xlabel="noise (pm)", ylabel=IRC_LBL, title="IRC convergence by method and noise")
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
        ax.set(xlabel=r"$F_\mathrm{max}$ threshold (eV/Å, tighter →)", title=f"{npm} pm")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    curve(axes[0], 50); curve(axes[1], 150)
    axes[0].set_ylabel(FN_LBL); axes[0].legend()
    save(fig, "fig_fmax_plateau")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, npm in zip(axes.ravel(), NOISES):
        curve(ax, npm)
    axes[0, 0].set_ylabel(FN_SHORT); axes[1, 0].set_ylabel(FN_SHORT)
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
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for k in cfg:
        ax.plot(NOISES, col(k, "wall_per_conv"), "-o", color=C[k], lw=2.2, ms=6, label=k)
    ax.set_yscale("log")
    ax.set(xlabel="noise (pm)", ylabel="wall-time per converged TS (s)",
           title="Cost: wall-time per converged TS")
    ax.legend(); save(fig, "fig_walltime")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for k in cfg:
        wall = col(k, "wall_per_conv"); inten = series(k, "intended_pct")
        ax.plot(wall, inten, "-o", color=C[k], lw=1.4, ms=7, label=k, alpha=0.9)
        for w_, i_, npm in zip(wall, inten, NOISES):
            ax.annotate(f"{npm}", (w_, i_), fontsize=7, color=C[k], xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlabel="wall-time per converged TS (s, log)", ylabel=IRC_LBL,
           title="Pareto: cost vs chemical reliability (label = noise pm)")
    ax.legend(); save(fig, "fig_pareto")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for k in cfg:
        ax.plot(NOISES, col(k, "med_step"), "-o", color=C[k], lw=2.2, ms=6, label=k)
    ax.set_yscale("log")
    ax.set(xlabel="noise (pm)", ylabel="median converged-step count",
           title="Steps to convergence (cheap GAD steps vs few Newton steps)")
    ax.legend(); save(fig, "fig_steps")

# ---------------------------------------------------------------- E. RMSD vs noise (paper Table)
def figRMSD():
    RMSD = {  # noise: (median, p95) from paper Table tab:rmsd
      "Sella":  {10:(0.008,0.073),50:(0.009,0.072),100:(0.009,0.201),150:(0.013,0.617),200:(0.017,0.838)},
      "GAD":    {10:(0.005,0.018),50:(0.011,0.028),100:(0.014,0.044),150:(0.016,0.088),200:(0.014,0.456)},
      "Hybrid": {10:(0.007,0.047),50:(0.007,0.049),100:(0.008,0.055),150:(0.007,0.062),200:(0.008,0.109)},
    }
    xs = [10,50,100,150,200]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for m in ["GAD", "Hybrid", "Sella"]:
        med = [RMSD[m][n][0] for n in xs]; p95 = [RMSD[m][n][1] for n in xs]
        ax.plot(xs, med, "-o", color=C[m], lw=2.2, ms=6, label=f"{m} (median)")
        ax.plot(xs, p95, "--s", color=C[m], lw=1.6, ms=5, alpha=0.8, label=f"{m} (p95)")
    ax.set(xlabel="initial-guess noise (pm)", ylabel="RMSD to true TS (Å)",
           title="Geometric quality of converged TS (median + p95)")
    ax.legend(fontsize=8, ncol=2); save(fig, "fig_rmsd_vs_noise")

# ---------------------------------------------------------------- F. conv != chem
def figF():
    m = pd.read_csv(f"{CSV}/master_2026_05_11.csv")
    d1, d3 = "Sella cartesian Eckart untuned Hess.Freq.=1", "Sella cartesian Eckart untuned Hess.Freq.=3"
    def g(c, col):
        s = m[m.config == c].set_index("noise_pm"); return [s[col].get(n, np.nan) for n in NOISES]
    fig, ax = plt.subplots(figsize=(8, 4.3)); x = np.arange(len(NOISES)); w = 0.2
    ax.bar(x-1.5*w, g(d3, "conv_pct"), w, color="#9ecae1", label="d=3 $F_\\mathrm{max}/n_\\mathrm{neg}$ conv")
    ax.bar(x-0.5*w, g(d1, "conv_pct"), w, color="#3182bd", label="d=1 $F_\\mathrm{max}/n_\\mathrm{neg}$ conv")
    ax.bar(x+0.5*w, g(d3, "topo_pct"), w, color="#fdae6b", label="d=3 IRC conv")
    ax.bar(x+1.5*w, g(d1, "topo_pct"), w, color="#e6550d", label="d=1 IRC conv")
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in NOISES])
    ax.set(xlabel="noise (pm)", ylabel="%",
           title="Convergence ≠ chemistry: d=3 has more $F/n$ conv, d=1 more IRC conv")
    ax.legend(ncol=2, fontsize=8); save(fig, "fig_d1_vs_d3")

# ---------------------------------------------------------------- G. reactant scope
def figRS():
    df = pd.read_csv(f"{CSV}/reactant_0pm_2026_05_16.csv")
    def val(sub):
        r = df[df.config.str.contains(sub, case=False, regex=False)]
        return float(r.iloc[0]["fmax_010"]) if len(r) else np.nan
    vals = {"GAD": val("GAD dt=0.005"), "Hybrid": val("Hybrid damped"),
            "Sella": val("Sella cartesian Eckart untuned Hess.Freq.=1")}
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ks = ["GAD", "Hybrid", "Sella"]
    ax.bar(ks, [vals[k] for k in ks], color=[C[k] for k in ks], width=0.62)
    for i, k in enumerate(ks):
        ax.text(i, vals[k]+1.5, f"{vals[k]:.0f}%", ha="center", fontsize=10)
    ax.set(ylabel=FN_LBL, ylim=(0, 100),
           title="Scope boundary: reactant-start convergence (0 pm)")
    save(fig, "fig_reactant_scope")

if __name__ == "__main__":
    print("building figures_new/ ...")
    figA(); figSC(); figB(); figC(); figD(); figRMSD(); figF(); figRS()
    print("done ->", OUT)
