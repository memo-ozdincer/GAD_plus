#!/usr/bin/env python3
"""Build the convergence/noise and successful-step figures beside the main table."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
RUNS = Path("/scratch/memoozd/gadplus/runs")
FIGURES = REPO / "docs/research/figures/all_methods_all_surfaces"
DATA = REPO / "docs/research/data/all_methods_all_surfaces"

COLORS = {
    "Ordinary GAD": "#7f8c8d", "Hard gate": "#9467bd",
    "Smooth λ2": "#17becf", "Intrinsic λ2": "#8c2d04",
    "Regular GAD": "#7f8c8d", "Competitive GAD": "#1f77b4",
    "CS²-GAD": "#d62728", "Sella": "#e69f00",
    "Plain GAD dt=.003": "#9ecae1", "Plain GAD dt=.005": "#4292c6",
    "Plain GAD dt=.007": "#08519c", "Sella Cartesian": "#fdae6b",
    "Sella + Eckart d=1": "#e6550d", "Sella + Eckart d=3": "#a63603",
    "Sella internal": "#8c564b", "Hybrid damped": "#31a354",
    "Hybrid undamped": "#74c476",
}


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _add_rate(rates, surface, method, noise, success, total, median=None):
    rates.append({
        "surface": surface, "method": method, "noise": float(noise),
        "success": int(success), "total": int(total),
        "rate_percent": 100.0 * success / total, "median_progress": median,
    })


def collect_lj(rates, steps):
    method_labels = {
        "ordinary_gad": "Ordinary GAD", "hard_gate": "Hard gate",
        "historical_lambda2": "Smooth λ2", "intrinsic": "Intrinsic λ2",
        "sella": "Sella",
    }
    rows = json.loads((RUNS / "lj-method-progression-1946071/all_results.json").read_text())
    rows += json.loads((RUNS / "lj7-sella-progression-1994169/all_results.json").read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[(method_labels[row["method"]], float(row["noise"]))].append(row)
    for (method, noise), group in sorted(groups.items()):
        successful = [row for row in group if row["converged"]]
        values = [float(row["n_evaluations"]) for row in successful]
        _add_rate(rates, "LJ7", method, noise, len(successful), len(group), _median(values))
        steps.extend({"surface": "LJ7", "method": method, "noise": noise, "progress": x}
                     for x in values)

    labels = {"intrinsic_lambda2": "Intrinsic λ2", "cs2": "CS²-GAD", "sella": "Sella"}
    rows = []
    for root in ("lj-multisize-cs2-2076554", "lj-multisize-high-noise-20260814"):
        rows += json.loads((RUNS / root / "all_results.json").read_text())
    groups.clear()
    for row in rows:
        groups[(labels[row["method"]], float(row["level_sigma"]))].append(row)
    for (method, noise), group in sorted(groups.items()):
        successful = [row for row in group if row["strict_ts"]]
        values = [float(row["evaluations"]) for row in successful]
        _add_rate(rates, "LJ13–75", method, noise, len(successful), len(group), _median(values))
        steps.extend({"surface": "LJ13–75", "method": method, "noise": noise, "progress": x}
                     for x in values)


def _gxtb_root(method: str, code: str) -> Path:
    candidates = sorted(RUNS.glob(f"t1x-gxtb-grid-{method}-{code}-*"))
    for root in reversed(candidates):
        if method in {"competitive", "competitive_subspace"}:
            count = len(list((root / "tasks").glob("task_*.json")))
        else:
            count = len(list(root.glob("task_*/summary_*.parquet")))
        if count == 287:
            return root
    raise FileNotFoundError(f"no complete g-xTB cell for {method} {code}")


def collect_gxtb(rates, steps):
    aggregate = json.loads(
        Path("/scratch/memoozd/gadplus/analysis/t1x-gxtb-matched-noise-grid/summary.json").read_text()
    )
    labels = {
        "regular_gad": "Regular GAD", "competitive": "Competitive GAD",
        "competitive_subspace": "CS²-GAD", "sella": "Sella",
    }
    codes = {0.10: "0p10", 0.20: "0p20", 0.50: "0p50", 1.00: "1p00", 2.00: "2p00"}
    expected = {(row["method"], float(row["noise_angstrom"])): row for row in aggregate}
    for method, label in labels.items():
        for noise, code in codes.items():
            root = _gxtb_root(method, code)
            if method in {"competitive", "competitive_subspace"}:
                rows = [json.loads(path.read_text()) for path in sorted((root / "tasks").glob("task_*.json"))]
                fmax_key, progress_key = "final_fmax", "total_steps"
            else:
                rows = [pq.read_table(path).to_pylist()[0]
                        for path in sorted(root.glob("task_*/summary_*.parquet"))]
                fmax_key, progress_key = "final_force_max", "total_steps"
            successful = [row for row in rows if int(row.get("final_n_neg", -1)) == 1
                          and float(row.get(fmax_key, np.inf)) < 0.03]
            target = expected[(method, noise)]["local_index1"]
            if len(rows) != 287 or len(successful) != target:
                raise RuntimeError(f"g-xTB raw/aggregate mismatch for {method} {noise}")
            values = [float(row[progress_key]) for row in successful]
            _add_rate(rates, "g-xTB", label, noise, len(successful), 287, _median(values))
            steps.extend({"surface": "g-xTB", "method": label, "noise": noise, "progress": x}
                         for x in values)


def collect_hip(rates, steps, medians_only):
    master = list(csv.DictReader(
        (REPO / "docs/research/analysis_2026_04_29/master_2026_05_16.csv").open()
    ))
    labels = {
        "GAD dt=0.003": "Plain GAD dt=.003",
        "GAD dt=0.005": "Plain GAD dt=.005",
        "GAD dt=0.007": "Plain GAD dt=.007",
        "Sella cartesian tuned Hess.Freq.=1": "Sella Cartesian",
        "Sella cartesian Eckart untuned Hess.Freq.=1": "Sella + Eckart d=1",
        "Sella cartesian Eckart untuned Hess.Freq.=3": "Sella + Eckart d=3",
        "Sella internal tuned Hess.Freq.=1": "Sella internal",
        "Hybrid damped Eckart eig tr=0.05": "Hybrid damped",
        "Hybrid undamped Eckart eig tr=0.05": "Hybrid undamped",
    }
    for row in master:
        label = labels.get(row["config"])
        if label is None:
            continue
        noise = int(row["noise_pm"]) / 1000.0
        success = round(float(row["conv_pct"]) * 287 / 100)
        median = float(row["med_step"]) if row["med_step"] else None
        _add_rate(rates, "HIP", label, noise, success, 287, median)
        if label in {"Plain GAD dt=.005", "Sella + Eckart d=3", "Hybrid damped", "Hybrid undamped"}:
            medians_only.append({"surface": "HIP", "method": label, "noise": noise,
                                 "progress": median})

    cs2_root = RUNS / "hip-cs2-h100-production-20260809-v2"
    for noise in (0.10, 0.15, 0.20):
        rows = [json.loads(path.read_text()) for path in sorted(
            (cs2_root / f"noise_{noise:.2f}A/samples").glob("sample_*.json")
        )]
        successful = [row for row in rows if row.get("strict_ts")]
        values = [float(row["n_evaluations"]) for row in successful]
        _add_rate(rates, "HIP", "CS²-GAD", noise, len(successful), 287, _median(values))
        steps.extend({"surface": "HIP", "method": "CS²-GAD", "noise": noise, "progress": x}
                     for x in values)

    gad_rows = list(csv.DictReader(
        (REPO / "docs/research/analysis_2026_04_29/gad_test_rmsd.csv").open()
    ))
    raw_gad = {
        "GAD dt=0.003 (5k)": "Plain GAD dt=.003",
        "GAD dt=0.007 (5k)": "Plain GAD dt=.007",
    }
    for row in gad_rows:
        label = raw_gad.get(row["method"])
        if label and int(row["n_neg"]) == 1 and float(row["force_max"]) < 0.01:
            steps.append({"surface": "HIP", "method": label,
                          "noise": int(row["noise_pm"]) / 1000.0,
                          "progress": float(row["total_steps"]) + 1.0})

    sella_rows = list(csv.DictReader(
        (REPO / "docs/research/analysis_2026_04_29/test_summary_full.csv").open()
    ))
    raw_sella = {
        "Sella default": "Sella Cartesian", "Sella libdef": "Sella + Eckart d=1",
        "Sella internal": "Sella internal",
    }
    for row in sella_rows:
        label = raw_sella[row["method"]]
        if int(row["final_n_neg"]) == 1 and float(row["final_fmax"]) < 0.01:
            steps.append({"surface": "HIP", "method": label,
                          "noise": int(row["noise_pm"]) / 1000.0,
                          "progress": float(row["total_steps"])})


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rates(rates):
    surfaces = ["LJ7", "LJ13–75", "g-xTB", "HIP"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, surface in zip(axes.flat, surfaces):
        subset = [row for row in rates if row["surface"] == surface]
        methods = list(dict.fromkeys(row["method"] for row in subset))
        for method in methods:
            rows = sorted((row for row in subset if row["method"] == method), key=lambda x: x["noise"])
            ax.plot([row["noise"] for row in rows], [row["rate_percent"] for row in rows],
                    marker="o", linewidth=2, markersize=5, label=method,
                    color=COLORS.get(method))
        ax.set_title(surface, loc="left", fontweight="bold")
        ax.set_xlabel("Cartesian noise (σ)" if surface.startswith("LJ") else "Cartesian noise (Å)")
        ax.set_ylabel("nneg=1 / fmax convergence (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=.22)
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    fig.suptitle("Local transition-state convergence over noise", fontsize=17, fontweight="bold")
    fig.savefig(FIGURES / "convergence_rate_over_noise.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(surface, rates, steps, medians_only):
    noises = sorted({row["noise"] for row in rates if row["surface"] == surface})
    methods = list(dict.fromkeys(row["method"] for row in rates if row["surface"] == surface))
    values = [row["progress"] for row in steps if row["surface"] == surface and row["progress"] > 0]
    lo, hi = max(1.0, min(values) * .8), max(values) * 1.25
    bins = np.geomspace(lo, hi, 18)
    fig, axes = plt.subplots(1, len(noises), figsize=(max(12, 3.25 * len(noises)), 4.5),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    handles, labels = [], []
    for ax, noise in zip(axes, noises):
        for method in methods:
            vals = [row["progress"] for row in steps
                    if row["surface"] == surface and row["method"] == method and row["noise"] == noise]
            if vals:
                weights = np.ones(len(vals)) / len(vals)
                _, _, artists = ax.hist(vals, bins=bins, weights=weights, histtype="step",
                                        linewidth=1.8, color=COLORS.get(method), label=method)
                if method not in labels:
                    handles.append(artists[0])
                    labels.append(method)
            else:
                med = next((row["progress"] for row in medians_only
                            if row["surface"] == surface and row["method"] == method
                            and row["noise"] == noise), None)
                if med is not None:
                    artist = ax.axvline(med, color=COLORS.get(method), linestyle=":", linewidth=1.6,
                                        label=f"{method} (median only)")
                    label = f"{method} (median only)"
                    if label not in labels:
                        handles.append(artist)
                        labels.append(label)
        unit = "σ" if surface.startswith("LJ") else "Å"
        ax.set_title(f"noise {noise:g} {unit}", fontsize=10)
        ax.set_xscale("log")
        ax.grid(alpha=.18)
        ax.set_xlabel("successful steps / evaluations")
    axes[0].set_ylabel("fraction of successful runs / bin")
    fig.suptitle(f"{surface}: progress to local convergence", fontsize=15, fontweight="bold")
    fig.legend(handles, labels, loc="outside lower center", ncol=min(5, len(labels)),
               frameon=False, fontsize=8)
    fig.savefig(FIGURES / f"steps_histogram_{surface.lower().replace('–', '_').replace('‑', '_').replace(' ', '_')}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rates, steps, medians_only = [], [], []
    collect_lj(rates, steps)
    collect_gxtb(rates, steps)
    collect_hip(rates, steps, medians_only)
    rates.sort(key=lambda row: (row["surface"], row["method"], row["noise"]))
    steps.sort(key=lambda row: (row["surface"], row["method"], row["noise"], row["progress"]))
    medians_only.sort(key=lambda row: (row["method"], row["noise"]))
    write_csv(DATA / "convergence_rates.csv", rates,
              ["surface", "method", "noise", "success", "total", "rate_percent", "median_progress"])
    write_csv(DATA / "successful_progress.csv", steps,
              ["surface", "method", "noise", "progress"])
    write_csv(DATA / "median_only_progress.csv", medians_only,
              ["surface", "method", "noise", "progress"])
    plot_rates(rates)
    for surface in ("LJ7", "LJ13–75", "g-xTB", "HIP"):
        plot_histograms(surface, rates, steps, medians_only)
    print(f"wrote {len(rates)} rate cells and {len(steps)} successful-run progress values")


if __name__ == "__main__":
    main()
