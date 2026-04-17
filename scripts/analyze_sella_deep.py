#!/usr/bin/env python
"""Deep analysis of sella_hip IRC validation over gad_dt003 TSs.

Cross-links IRC endpoint data with the upstream TS-side convergence summary
(round2/round3) to answer:

A. Primary metrics per noise (all criteria)
B. Endpoint spectral breakdown — are IRC endpoints at true minima?
C. Topology-vs-RMSD gap — bond-graph right, geometry wrong
D. Directional asymmetry — which direction fails more?
E. Systematic sample failures across noise
F. TS quality (from GAD) vs IRC outcome
G. Wall-time distribution
H. Per-formula difficulty
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

IRC_DIR = "/lustre07/scratch/memoozd/gadplus/runs/irc_sellahip_full"
NOISE_TO_SURVEY = {
    10:  "round2", 30:  "round2", 50:  "round2",
    100: "round3", 150: "round3", 200: "round3",
}
SURVEY_ROOT = "/lustre07/scratch/memoozd/gadplus/runs"


def load_irc() -> pd.DataFrame:
    return duckdb.execute(f"SELECT * FROM '{IRC_DIR}/*.parquet'").df()


def load_ts_side() -> pd.DataFrame:
    rows = []
    for noise, sub in NOISE_TO_SURVEY.items():
        p = f"{SURVEY_ROOT}/{sub}/summary_gad_dt003_{noise}pm.parquet"
        df = duckdb.execute(f"SELECT * FROM '{p}' WHERE converged").df()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fmt_pct(n: float, d: float) -> str:
    if d == 0:
        return "  -   "
    return f"{100*n/d:5.1f}%"


def main() -> None:
    irc = load_irc()
    ts = load_ts_side()
    # IRC has sample_id + noise_pm; TS-side joined on same.
    m = irc.merge(
        ts[["sample_id", "noise_pm", "converged_step", "final_force_norm",
            "final_n_neg", "final_energy", "final_eig0", "total_steps"]],
        on=["sample_id", "noise_pm"], how="left",
        suffixes=("", "_ts"),
    )

    # --------------------------------------------------------------------
    section("A. Primary metrics (per noise)")
    rows = []
    for noise, g in m.groupby("noise_pm"):
        n = len(g)
        rows.append({
            "noise_pm": noise, "n": n,
            "RMSD-int%":     100 * g["intended"].sum() / n,
            "RMSD-half%":    100 * g["half_intended"].sum() / n,
            "TOPO-int%":     100 * g["topology_intended"].sum() / n,
            "TOPO-half%":    100 * g["topology_half_intended"].sum() / n,
            "err%":          100 * g["error"].notna().sum() / n,
            "fwd-to-R%":     100 * g["forward_graph_matches_reactant"].sum() / n,
            "fwd-to-P%":     100 * g["forward_graph_matches_product"].sum() / n,
            "rev-to-R%":     100 * g["reverse_graph_matches_reactant"].sum() / n,
            "rev-to-P%":     100 * g["reverse_graph_matches_product"].sum() / n,
            "any-dir-match%":100 * (
                g["forward_graph_matches_reactant"] |
                g["forward_graph_matches_product"]  |
                g["reverse_graph_matches_reactant"] |
                g["reverse_graph_matches_product"]
            ).sum() / n,
            "avg_wall_s":    g["wall_time_s"].mean(),
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # --------------------------------------------------------------------
    section("B. Endpoint spectral quality — are IRC endpoints true minima?")
    print("(n_neg_vib==0 AND min_vib_eig>0 ⇒ true minimum)")
    rows = []
    for noise, g in m.groupby("noise_pm"):
        fwd_min = (g["forward_n_neg_vib"] == 0).fillna(False)
        rev_min = (g["reverse_n_neg_vib"] == 0).fillna(False)
        both_min = fwd_min & rev_min
        either_min = fwd_min | rev_min
        neither_min = (~fwd_min) & (~rev_min)
        rows.append({
            "noise_pm":      noise,
            "n":             len(g),
            "fwd_at_min%":   100 * fwd_min.sum() / len(g),
            "rev_at_min%":   100 * rev_min.sum() / len(g),
            "both_at_min%":  100 * both_min.sum() / len(g),
            "neither_at_min%": 100 * neither_min.sum() / len(g),
            "fwd_has_neg%":  100 * (g["forward_n_neg_vib"] > 0).sum() / len(g),
            "rev_has_neg%":  100 * (g["reverse_n_neg_vib"] > 0).sum() / len(g),
            "median_fwd_mineig": g["forward_min_vib_eig"].median(),
            "median_rev_mineig": g["reverse_min_vib_eig"].median(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # --------------------------------------------------------------------
    section("C. Topology-vs-RMSD gap")
    print("Among TOPO-intended runs: what fraction also pass strict RMSD?")
    rows = []
    for noise, g in m.groupby("noise_pm"):
        topo_ok = g[g["topology_intended"]]
        rows.append({
            "noise_pm": noise,
            "n_topo_int": len(topo_ok),
            "of_which_rmsd_int": int(topo_ok["intended"].sum()),
            "rmsd_int_given_topo%":
                100 * topo_ok["intended"].sum() / max(1, len(topo_ok)),
            "topo_int_but_rmsd_fail": int(
                (topo_ok["intended"] == False).sum()),
            "median_rmsd_to_R_at_topo_int":
                topo_ok["rmsd_reactant"].median(),
            "median_rmsd_to_P_at_topo_int":
                topo_ok["rmsd_product"].median(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # --------------------------------------------------------------------
    section("D. Directional asymmetry")
    print("Which IRC direction hits its target more often?")
    rows = []
    for noise, g in m.groupby("noise_pm"):
        # Direction-agnostic labels from validate: 'intended' means (fR+rP) or (fP+rR) match.
        # Look at individual direction reliability: fraction where forward matched *something*.
        fwd_match = (g["forward_graph_matches_reactant"] |
                     g["forward_graph_matches_product"])
        rev_match = (g["reverse_graph_matches_reactant"] |
                     g["reverse_graph_matches_product"])
        rows.append({
            "noise_pm":            noise,
            "fwd_matches_either%": 100 * fwd_match.sum() / len(g),
            "rev_matches_either%": 100 * rev_match.sum() / len(g),
            "only_fwd_matches%":   100 * (fwd_match & ~rev_match).sum() / len(g),
            "only_rev_matches%":   100 * (rev_match & ~fwd_match).sum() / len(g),
            "both_match%":         100 * (fwd_match & rev_match).sum() / len(g),
            "neither%":            100 * (~fwd_match & ~rev_match).sum() / len(g),
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # --------------------------------------------------------------------
    section("E. Systematic failures — sample_ids that fail across noise levels")
    pivot_topo = m.pivot_table(
        index="sample_id", columns="noise_pm",
        values="topology_intended", aggfunc="first",
    )
    present = pivot_topo.notna()
    failing = (~pivot_topo.fillna(False)) & present
    fail_counts = failing.sum(axis=1)
    present_counts = present.sum(axis=1)

    consistent = ((present_counts >= 4) & (fail_counts == present_counts)).sum()
    print(f"Samples present at >=4 noise levels AND failing TOPO at ALL of them: {consistent}")
    print()
    worst = fail_counts[fail_counts >= 5].sort_values(ascending=False).head(20)
    if len(worst):
        print("Sample IDs failing TOPO at >= 5 noise levels:")
        for sid, cnt in worst.items():
            sub = m[m["sample_id"] == sid].iloc[0]
            f = sub["formula"]
            # Avg min endpoint RMSD across noise levels where present
            cases = m[m["sample_id"] == sid]
            best_rmsd = cases[["forward_rmsd_reactant","forward_rmsd_product",
                               "reverse_rmsd_reactant","reverse_rmsd_product"]
                              ].min(axis=1).median()
            fwd_nneg = cases["forward_n_neg_vib"].dropna().median()
            rev_nneg = cases["reverse_n_neg_vib"].dropna().median()
            print(f"  sid={sid:>4d} {f:>12s}  "
                  f"fails {int(cnt)}/{int(present_counts[sid])} noise levels  "
                  f"median_best_RMSD={best_rmsd:.2f}  "
                  f"median n_neg [f={fwd_nneg}, r={rev_nneg}]")
    else:
        print("(none)")

    # --------------------------------------------------------------------
    section("F. TS quality (from GAD) vs IRC outcome")
    print("Does IRC outcome correlate with how cleanly GAD converged?")
    # Bin on |final_eig0| at TS (stronger negative eig = sharper saddle, usually better)
    for metric_name, metric_col in [
        ("final_force_norm", "final_force_norm"),
        ("final_eig0",       "final_eig0"),
        ("converged_step",   "converged_step"),
        ("total_steps",      "total_steps"),
    ]:
        rows = []
        for noise, g in m.groupby("noise_pm"):
            g_ok = g[g["topology_intended"]]
            g_bad = g[~g["topology_intended"]]
            rows.append({
                "noise_pm": noise,
                f"{metric_name}_topo_int_median":
                    g_ok[metric_col].median(),
                f"{metric_name}_topo_fail_median":
                    g_bad[metric_col].median(),
                f"{metric_name}_topo_int_p90":
                    g_ok[metric_col].quantile(0.90),
                f"{metric_name}_topo_fail_p90":
                    g_bad[metric_col].quantile(0.90),
            })
        print(f"\n{metric_name} distribution by IRC outcome:")
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    # --------------------------------------------------------------------
    section("G. Wall-time statistics")
    stats = m.groupby("noise_pm")["wall_time_s"].describe(
        percentiles=[0.5, 0.9, 0.99]
    ).round(1)
    print(stats.to_string())

    # --------------------------------------------------------------------
    section("H. Per-formula difficulty (top 15 by failure count)")
    # How many TOPO-failed runs per formula, across all noise
    per_formula = m.groupby("formula").agg(
        n=("topology_intended", "size"),
        n_topo_int=("topology_intended", "sum"),
        n_topo_fail=("topology_intended", lambda s: (~s).sum()),
        avg_wall=("wall_time_s", "mean"),
    )
    per_formula["topo_int%"] = 100 * per_formula["n_topo_int"] / per_formula["n"]
    # Most-failing formulas (min 3 samples to be meaningful)
    hard = per_formula[per_formula["n"] >= 3].sort_values("topo_int%").head(15)
    print("Worst TOPO% formulas (n >= 3):")
    print(hard.round(1).to_string())

    print("\nBest TOPO% formulas (n >= 3):")
    easy = per_formula[per_formula["n"] >= 3].sort_values(
        "topo_int%", ascending=False
    ).head(10)
    print(easy.round(1).to_string())

    # --------------------------------------------------------------------
    section("Summary")
    total = len(m)
    print(f"Total IRC runs:       {total}")
    print(f"TOPO-intended:        {int(m['topology_intended'].sum())} "
          f"({100*m['topology_intended'].mean():.1f}%)")
    print(f"RMSD-intended:        {int(m['intended'].sum())} "
          f"({100*m['intended'].mean():.1f}%)")
    print(f"Errors:               {int(m['error'].notna().sum())}")
    # Valid-but-wrong = topology-failed but both endpoints at real minima
    fwd_min = (m["forward_n_neg_vib"] == 0).fillna(False)
    rev_min = (m["reverse_n_neg_vib"] == 0).fillna(False)
    vbw = (~m["topology_intended"]) & fwd_min & rev_min
    print(f"Valid-but-wrong:      {int(vbw.sum())} "
          f"({100*vbw.mean():.1f}%) "
          f"[IRC found real minima, just not the labeled R/P pair]")
    # Ridge-stall = any endpoint has n_neg > 0
    stall = (m["forward_n_neg_vib"] > 0) | (m["reverse_n_neg_vib"] > 0)
    print(f"Ridge-stall endpoint: {int(stall.sum())} "
          f"({100*stall.mean():.1f}%) "
          f"[one/both endpoints still on a saddle]")


if __name__ == "__main__":
    main()
