#!/usr/bin/env python
"""Integrate SLURM job 61087603 results once they land.

Produces:
  - analysis_2026_04_29/master_2026_05_16.csv      (refreshed headline table)
  - analysis_2026_04_29/reactant_0pm_2026_05_16.csv (refreshed bar-chart data, with hybrid)
  - analysis_2026_04_29/longbudget_2026_05_16.csv  (R4 10k-step probe — fmax<{0.05, 0.023, 0.01, 0.005, 0.001})

Run after `squeue -j 61087603` returns nothing.
"""
from __future__ import annotations

import os
import sys
import glob

import duckdb
import numpy as np
import pandas as pd

import re

# Live (log-parse) cells with fewer than MIN_N samples are flagged but
# NOT written to master_2026_05_16.csv or reactant_0pm_2026_05_16.csv —
# the existing published numbers stay until enough samples land.
MIN_N_FOR_WRITE = 50

ROOT = "/lustre06/project/6033559/memoozd/GAD_plus"
RUNS = "/lustre07/scratch/memoozd/gadplus/runs"
LOGS = "/lustre07/scratch/memoozd/gadplus/logs"
CSV  = f"{ROOT}/analysis_2026_04_29"
con  = duckdb.connect()


# ── Log-parser fallback for cells that time out before writing summary ──
SELLA_LINE = re.compile(
    r"\[\s*(?P<sid>\d+)\]\s+(?P<formula>\S+)\s*\|\s*sella=(?P<sella>\w+)\s+"
    r"fmax=(?P<fmax>[\d.eE+-]+)\s*\|\s*n_neg=(?P<nneg>-?\d+)\s+"
    r"force=(?P<fnorm>[\d.eE+-]+)\s+ours=(?P<ours>\w+)\s*\|\s*"
    r"steps=\s*(?P<steps>\d+)\s*\|\s*(?P<wall>[\d.]+)s"
)
HYBRID_LINE = re.compile(
    r"\[\s*(?P<sid>\d+)\]\s+(?P<formula>\S+)\s*\|\s*(?P<status>CONV|FAIL)\s*\|\s*"
    r"n_neg=(?P<nneg>-?\d+)\s+fmax=(?P<fmax>[\d.eE+-]+)\s+"
    r"steps=(?P<steps>\d+)\s+wall=(?P<wall>[\d.]+)s"
)
GAD_LINE = re.compile(
    r"\[\s*(?P<sid>\d+)\]\s+(?P<formula>\S+)\s*\|\s*(?P<status>CONV|FAIL)\s*\|\s*"
    r"n_neg=(?P<nneg>-?\d+)\s*\|\s*force_norm=(?P<fnorm>[\d.eE+-]+)\s*\|\s*"
    r"force_max=(?P<fmax>[\d.eE+-]+)\s*\|\s*steps=(?P<steps>\d+)\s*\|\s*"
    r"(?P<wall>[\d.]+)s"
)


def parse_sella_log(log_path):
    if not os.path.exists(log_path): return None
    rows = []
    with open(log_path) as f:
        for line in f:
            m = SELLA_LINE.search(line)
            if not m: continue
            rows.append({
                "sample_id": int(m["sid"]), "formula": m["formula"],
                "final_fmax": float(m["fmax"]), "final_n_neg": int(m["nneg"]),
                "final_force_norm": float(m["fnorm"]),
                "total_steps": int(m["steps"]), "wall_time_s": float(m["wall"]),
            })
    return pd.DataFrame(rows) if rows else None


def parse_hybrid_log(log_path):
    if not os.path.exists(log_path): return None
    rows = []
    with open(log_path) as f:
        for line in f:
            m = HYBRID_LINE.search(line)
            if not m: continue
            rows.append({
                "sample_id": int(m["sid"]), "formula": m["formula"],
                "final_force_max": float(m["fmax"]), "final_n_neg": int(m["nneg"]),
                "total_steps": int(m["steps"]), "wall_time_s": float(m["wall"]),
            })
    return pd.DataFrame(rows) if rows else None


def parse_gad_log(log_path):
    if not os.path.exists(log_path): return None
    rows = []
    with open(log_path) as f:
        for line in f:
            m = GAD_LINE.search(line)
            if not m: continue
            rows.append({
                "sample_id": int(m["sid"]), "formula": m["formula"],
                "final_force_max": float(m["fmax"]),
                "final_force_norm": float(m["fnorm"]),
                "final_n_neg": int(m["nneg"]),
                "total_steps": int(m["steps"]), "wall_time_s": float(m["wall"]),
            })
    return pd.DataFrame(rows) if rows else None


def thresholds_from_df(df, fmax_col):
    n = len(df)
    if n == 0: return None
    out = {"n": n}
    for thr, lab in [(0.05, "fmax_005"), (0.023, "fmax_023"), (0.01, "fmax_010"),
                     (0.005, "fmax_005t"), (0.001, "fmax_001")]:
        out[lab] = 100 * ((df["final_n_neg"] == 1) & (df[fmax_col] < thr)).sum() / n
    return out


def thresholds_from_summary(parquet_glob, fmax_col, nneg_col="final_n_neg",
                            log_fallback=None, parser=None):
    """Compute conv % at 5 fmax thresholds for one cell.

    If the summary parquet glob is empty and `log_fallback` is provided, try
    parsing the SLURM .out log as a partial-data fallback.
    """
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{parquet_glob}')").df()
        if len(df) > 0:
            return thresholds_from_df(df, fmax_col)
    except Exception:
        pass
    # Fallback: log-parse
    if log_fallback and parser:
        df = parser(log_fallback)
        if df is not None and len(df) > 0:
            r = thresholds_from_df(df, fmax_col)
            if r is not None:
                r["from_log"] = True
            return r
    return None


# ── R1: hybrid from reactant @ 0pm ────────────────────────────────────────
print("R1: hybrid from reactant @ 0pm")
r1_rows = []
for cfg, tag, log_idx in [
    ("damped_dt5e-3_tr0.05",   "Hybrid damped Eckart eig tr=0.05", 0),
    ("undamped_dt5e-3_tr0.05", "Hybrid undamped Eckart eig tr=0.05", 1),
]:
    paths = glob.glob(f"{RUNS}/start_reactant_hybrid/{cfg}/summary_*.parquet")
    log_path = f"{LOGS}/compr_61087603_{log_idx}.out"
    r = thresholds_from_summary(
        paths[0] if paths else "/nonexistent/*",
        "final_force_max",
        log_fallback=log_path, parser=parse_hybrid_log,
    )
    if r is None:
        print(f"  PENDING: {cfg}")
        continue
    r["config"] = tag; r["family"] = "hybrid"; r["is_partial"] = r["n"] < 287
    src = "log" if r.get("from_log") else "parquet"
    r1_rows.append(r)
    print(f"  {tag} @ 0pm  n={r['n']}/287  fmax<0.01={r['fmax_010']:.1f}  ({src})")

# Merge with existing reactant CSV — only write hybrid bars once n >= MIN_N
existing = pd.read_csv(f"{CSV}/reactant_0pm_2026_05_16.csv")
r1_rows_writeable = [r for r in r1_rows if r["n"] >= MIN_N_FOR_WRITE]
if r1_rows_writeable:
    new = pd.DataFrame(r1_rows_writeable)
    combined = pd.concat([existing[existing["family"] != "hybrid"], new], ignore_index=True)
    combined = combined[["config", "family", "n", "fmax_005", "fmax_023", "fmax_010", "is_partial"]]
    out_path = f"{CSV}/reactant_0pm_2026_05_16.csv"
    combined.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}: {len(combined)} rows (hybrid added)")
elif r1_rows:
    print(f"  Hybrid n<{MIN_N_FOR_WRITE} — NOT updating reactant CSV yet (would be too noisy)")


# ── Helpers for pooled (main + safety-net) accounting ──────────────────────
def pool_sample_rows(sources):
    """Read all source dataframes/log files, dedupe by sample_id, return pooled df."""
    pieces = []
    for src in sources:
        if "kind" in src:
            kind = src["kind"]
            if kind == "parquet":
                try:
                    df = con.execute(f"SELECT * FROM read_parquet('{src['path']}')").df()
                    if len(df) > 0:
                        pieces.append(df.assign(_src=src["path"]))
                except Exception:
                    pass
            elif kind == "log":
                df = src["parser"](src["path"])
                if df is not None and len(df) > 0:
                    pieces.append(df.assign(_src=src["path"]))
    if not pieces:
        return None
    pooled = pd.concat(pieces, ignore_index=True)
    pooled = pooled.drop_duplicates(subset=["sample_id"], keep="first")
    return pooled


# ── R2: Sella d=3 @ 200pm (POOLED: main + 3 safety nets) ──────────────────
print("\nR2: Sella d=3 @ 200pm (pooled main + safety-net)")
r2_sources = [
    {"kind": "parquet", "path": p} for p in
    glob.glob(f"{RUNS}/test_hessfreq/sella_carteck_libdef_d3/summary*200pm*.parquet")
] + [
    {"kind": "parquet", "path": p} for p in
    glob.glob(f"{RUNS}/safetynet/sella_carteck_libdef_d3/summary*200pm*.parquet")
] + [
    {"kind": "log", "path": f"{LOGS}/compr_61087603_2.out", "parser": parse_sella_log},
] + [
    {"kind": "log", "path": f"{LOGS}/safety_61088001_{i}.out", "parser": parse_sella_log}
    for i in [0, 1, 2]
]
r2_pool = pool_sample_rows(r2_sources)
if r2_pool is not None:
    r2 = thresholds_from_df(r2_pool, "final_fmax")
    print(f"  d=3 @ 200pm  n={r2['n']}/287 (pooled)  fmax<0.01={r2['fmax_010']:.1f}")
else:
    r2 = None
    print("  PENDING")


# ── R3: Sella internal @ 150pm and 200pm ──────────────────────────────────
print("\nR3: Sella internal @ 150/200pm")
r3 = {}
for npm, log_idx, sn_idxs in [(150, 3, []), (200, 4, [3, 4, 5])]:
    src = [
        {"kind": "parquet", "path": p} for p in
        glob.glob(f"{RUNS}/test_set/sella_internal_default/summary*{npm}pm*.parquet")
    ] + [
        {"kind": "parquet", "path": p} for p in
        glob.glob(f"{RUNS}/safetynet/sella_internal_default/summary*{npm}pm*.parquet")
    ] + [
        {"kind": "log", "path": f"{LOGS}/compr_61087603_{log_idx}.out", "parser": parse_sella_log},
    ] + [
        {"kind": "log", "path": f"{LOGS}/safety_61088001_{i}.out", "parser": parse_sella_log}
        for i in sn_idxs
    ]
    pool = pool_sample_rows(src)
    if pool is None:
        print(f"  PENDING: {npm}pm")
        continue
    r = thresholds_from_df(pool, "final_fmax")
    r3[npm] = r
    pooled_note = " (pooled)" if sn_idxs else ""
    print(f"  internal @ {npm}pm  n={r['n']}/287{pooled_note}  fmax<0.01={r['fmax_010']:.1f}")


# ── R4: 10k-step long budget probe ────────────────────────────────────────
print("\nR4: 10k-step long budget @ 50pm")
r4_rows = []
r4_specs = [
    (f"{RUNS}/test_longbudget/gad_dt005_10k/summary_*.parquet",            "final_force_max", "GAD dt=0.005 ×10k",  f"{LOGS}/compr_61087774_5.out", parse_gad_log),
    (f"{RUNS}/test_longbudget/sella_carteck_libdef_10k/summary_*.parquet", "final_fmax",      "Sella libdef ×10k",  f"{LOGS}/compr_61087774_6.out", parse_sella_log),
    (f"{RUNS}/test_longbudget/hybrid_damped_10k/summary_*.parquet",        "final_force_max", "Hybrid damped ×10k", f"{LOGS}/compr_61087774_7.out", parse_hybrid_log),
]
for path_glob, fcol, label, log_path, parser in r4_specs:
    r = thresholds_from_summary(path_glob, fcol, log_fallback=log_path, parser=parser)
    if r is None:
        print(f"  PENDING: {label}")
        continue
    r["config"] = label; r["noise_pm"] = 50
    src = "log" if r.get("from_log") else "parquet"
    r4_rows.append(r)
    print(f"  {label}  n={r['n']:>3}/287  fmax<0.05={r['fmax_005']:5.1f}  fmax<0.01={r['fmax_010']:5.1f}  fmax<0.005={r['fmax_005t']:5.1f}  fmax<0.001={r['fmax_001']:5.1f}  ({src})")

if r4_rows:
    df = pd.DataFrame(r4_rows)
    out_path = f"{CSV}/longbudget_2026_05_16.csv"
    df.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")


# ── R1-c: Sella from midpoint @ 0pm ───────────────────────────────────────
print("\nR1-c: Sella libdef from midpoint @ 0pm")
r1c_path = glob.glob(f"{RUNS}/start_midpoint/sella_carteck_libdef/summary*.parquet")
rmc = thresholds_from_summary(
    r1c_path[0] if r1c_path else "/nonexistent/*", "final_fmax",
    log_fallback=f"{LOGS}/compr_61087603_8.out", parser=parse_sella_log,
)
if rmc:
    src = "log" if rmc.get("from_log") else "parquet"
    print(f"  midpoint Sella libdef @ 0pm  n={rmc['n']}/287  fmax<0.01={rmc['fmax_010']:.1f}  ({src})")
else:
    print("  PENDING")


# ── Refresh master_2026_05_16.csv ─────────────────────────────────────────
print("\nRefreshing master table…")
master = pd.read_csv(f"{CSV}/master_2026_05_11.csv")

# Patch d=3 @ 200pm — only when n >= MIN_N_FOR_WRITE
if r2 is not None and r2["n"] >= MIN_N_FOR_WRITE:
    mask = ((master["config"] == "Sella cartesian Eckart untuned Hess.Freq.=3") &
            (master["noise_pm"] == 200))
    if mask.any():
        master.loc[mask, "conv_pct"] = r2["fmax_010"]
    else:
        master = pd.concat([master, pd.DataFrame([{
            "family": "Sella",
            "config": "Sella cartesian Eckart untuned Hess.Freq.=3",
            "noise_pm": 200,
            "conv_pct": r2["fmax_010"],
        }])], ignore_index=True)
elif r2 is not None:
    print(f"  d=3 @ 200pm n={r2['n']} too small — keeping prior published value")

# Patch Sella internal @ 150/200 — only when n >= MIN_N_FOR_WRITE
for npm, r in r3.items():
    if r["n"] < MIN_N_FOR_WRITE:
        print(f"  internal @ {npm}pm n={r['n']} too small — keeping prior partial value")
        continue
    mask = ((master["config"] == "Sella internal tuned Hess.Freq.=1") &
            (master["noise_pm"] == npm))
    if mask.any():
        master.loc[mask, "conv_pct"] = r["fmax_010"]
    else:
        master = pd.concat([master, pd.DataFrame([{
            "family": "Sella",
            "config": "Sella internal tuned Hess.Freq.=1",
            "noise_pm": npm,
            "conv_pct": r["fmax_010"],
        }])], ignore_index=True)

master.to_csv(f"{CSV}/master_2026_05_16.csv", index=False)
print(f"  Wrote {CSV}/master_2026_05_16.csv ({len(master)} rows)")

print("\nDone. Rerun scripts/build_pdf_2026_05_16.py + pdflatex BENCHMARK_REPORT_2026-05-16.tex to refresh.")
