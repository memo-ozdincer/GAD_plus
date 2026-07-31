#!/usr/bin/env python
"""Summarize a paired PaiNN GAD/Sella pilot without unpaired-rate shortcuts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _two_sided_exact_binomial(k: int, n: int) -> float | None:
    """Exact two-sided sign-test p-value for discordant paired outcomes."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--terminal-summary",
        type=Path,
        help="optional IRC/TOPO validation summary for strict terminal successes",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_parquet(args.summary)
    required = {"candidate_file", "noise_pm", "seed", "method", "strict_converged"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"summary missing required fields: {missing}")
    duplicated = data.duplicated(["candidate_file", "noise_pm", "seed", "method"])
    if bool(duplicated.any()):
        raise ValueError("summary has duplicate paired method rows")

    if args.terminal_summary:
        terminals = pd.read_parquet(args.terminal_summary)
        terminal_keys = ["candidate_file", "noise_pm", "seed", "method"]
        required_terminal = set(terminal_keys) | {"terminal_accepted"}
        missing_terminal = sorted(required_terminal - set(terminals.columns))
        if missing_terminal:
            raise ValueError(f"terminal summary missing required fields: {missing_terminal}")
        if bool(terminals.duplicated(terminal_keys).any()):
            raise ValueError("terminal summary has duplicate terminal rows")
        data = data.merge(
            terminals[terminal_keys + ["terminal_accepted"]],
            how="left",
            on=terminal_keys,
        )
        data["terminal_accepted"] = data["terminal_accepted"].fillna(False).astype(bool)
        data["chemical_success"] = (
            data["strict_converged"].astype(bool) & data["terminal_accepted"]
        )

    output: dict[str, object] = {
        "n_rows": int(len(data)),
        "n_paired_starts": 0,
        "by_noise_pm": {},
        "errors": int(data["optimizer_error"].notna().sum()) if "optimizer_error" in data else 0,
    }
    for noise, subset in data.groupby("noise_pm", sort=True):
        def paired_outcomes(column: str) -> dict[str, int | float | None]:
            pivot = subset.pivot(
                index=["candidate_file", "seed"], columns="method", values=column
            )
            if set(pivot.columns) != {"gad", "sella"} or pivot.isna().any().any():
                raise ValueError(f"incomplete paired rows at noise={noise}")
            gad = pivot["gad"].astype(bool)
            sella = pivot["sella"].astype(bool)
            gad_only = int((gad & ~sella).sum())
            sella_only = int((~gad & sella).sum())
            discordant = gad_only + sella_only
            return {
                "n_starts": int(len(pivot)),
                "gad_successes": int(gad.sum()),
                "sella_successes": int(sella.sum()),
                "gad_rate": float(gad.mean()),
                "sella_rate": float(sella.mean()),
                "both_success": int((gad & sella).sum()),
                "gad_only": gad_only,
                "sella_only": sella_only,
                "neither_success": int((~gad & ~sella).sum()),
                "mcnemar_exact_two_sided_p": _two_sided_exact_binomial(
                    min(gad_only, sella_only), discordant,
                ),
            }

        row = {"strict_stationary": paired_outcomes("strict_converged")}
        if args.terminal_summary:
            row["terminal_irc_topo"] = paired_outcomes("chemical_success")
        output["by_noise_pm"][str(noise)] = row
        output["n_paired_starts"] += int(row["strict_stationary"]["n_starts"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
