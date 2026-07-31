#!/usr/bin/env python3
"""Summarize pure-GAD trajectory parquet files into failure taxonomy rows.

The report deliberately retains the raw dynamical evidence needed to diagnose
why a trajectory did not reach an index-1 TS: λ2 sign/magnitude, time in the
index-1 basin, mode continuity, force history, and geometry drift.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def transitions(values: np.ndarray) -> int:
    return int(np.count_nonzero(values[1:] != values[:-1])) if len(values) > 1 else 0


def classify(df: pd.DataFrame, threshold: float) -> str:
    last = df.iloc[-1]
    n = int(last.n_neg)
    f = float(last.force_max)
    if n == 1 and f < threshold:
        return "converged"
    if n == 0:
        return "minimum_like"
    if n > 1:
        return "multi_negative"
    if (df.n_neg == 1).any():
        return "index1_force_limited"
    return "never_index1"


def summarize(path: str, threshold: float) -> dict:
    df = pd.read_parquet(path).sort_values("step").reset_index(drop=True)
    last = df.iloc[-1]
    idx1 = df.n_neg.eq(1)
    lam2 = df.eig1.to_numpy(dtype=float)
    cont = df.eigvec_continuity.dropna().to_numpy(dtype=float)
    # A λ2 sign change corresponds to moving between the index-1 and
    # multi-negative sides (up to the explicit n_neg cutoff).
    lam2_sign_changes = transitions(np.signbit(lam2))
    row = {
        "trajectory": str(path),
        "sample_id": int(last.sample_id),
        "formula": last.formula,
        "rxn": last.rxn,
        "n_steps": len(df),
        "failure_class": classify(df, threshold),
        "initial_n_neg": int(df.n_neg.iloc[0]),
        "final_n_neg": int(last.n_neg),
        "final_fmax": float(last.force_max),
        "min_fmax": float(df.force_max.min()),
        "min_fmax_at_nneg1": float(df.loc[idx1, "force_max"].min()) if idx1.any() else np.nan,
        "frac_nneg1": float(idx1.mean()),
        "first_nneg1_step": int(df.loc[idx1, "step"].iloc[0]) if idx1.any() else -1,
        "nneg_transitions": transitions(df.n_neg.to_numpy()),
        "lambda2_initial": float(lam2[0]),
        "lambda2_final": float(lam2[-1]),
        "lambda2_min_abs": float(np.abs(lam2).min()),
        "lambda2_sign_changes": lam2_sign_changes,
        "lambda2_frac_positive": float((lam2 > 1e-4).mean()),
        "mode_continuity_min": float(cont.min()) if len(cont) else np.nan,
        "mode_continuity_mean": float(cont.mean()) if len(cont) else np.nan,
        # Number of step-to-step changes in the best-matching tracked mode.
        "mode_index_switches": transitions(df.mode_index.fillna(0).to_numpy()),
        "max_disp_from_start": float(df.disp_from_start.max()),
        "final_dist_to_labelled_ts": float(last.dist_to_known_ts),
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, dest="pattern", help="Quoted trajectory parquet glob.")
    p.add_argument("--output", required=True)
    p.add_argument("--force-threshold", type=float, default=0.01)
    args = p.parse_args()
    paths = sorted(glob.glob(args.pattern, recursive=True))
    if not paths:
        raise SystemExit(f"No trajectory files matched: {args.pattern}")
    table = pd.DataFrame([summarize(path, args.force_threshold) for path in paths])
    table = table.sort_values(["failure_class", "sample_id"]).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.groupby("failure_class").size().to_string())
    print(f"Wrote {len(table)} trajectory summaries to {out}")


if __name__ == "__main__":
    main()
