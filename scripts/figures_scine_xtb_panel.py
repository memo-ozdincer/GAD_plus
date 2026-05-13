"""SCINE/xTB second-calculator panel — mirrors the HIP `fig_noise_sweep_with_irc`
layout for the paper's appendix.

Left panel: HIP rows from analysis_2026_04_29/noise_sweep_with_irc.csv
            ("plain GAD dt=0.005", "Sella libdef") for context.

Right panel: SCINE/DFTB0 rows from analysis_2026_04_29/noise_sweep_scine_xtb.csv
             plus the xTB favorable-panel single-point at 10 pm (annotated).

Uses strict-conv (n_neg==1 ∧ fmax<0.01) for solid lines and topology-only
(n_neg==1, any fmax) for dashed lines on the SCINE panel, so the
"GAD plateau" is visible.

Output:
  figures/fig_noise_sweep_scine_xtb.pdf
  figures/fig_noise_sweep_scine_xtb.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from plotting_style import apply_plot_style, palette_color

apply_plot_style()

ROOT = Path("/lustre06/project/6033559/memoozd/GAD_plus")
ANL = ROOT / "analysis_2026_04_29"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True, parents=True)

NOISES = [10, 30, 50, 100, 150, 200]

C_GAD = palette_color(0)
C_GAD_NECK = palette_color(1)
C_SELLA = palette_color(2)
C_XTB = palette_color(3)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def lookup(rows, family_substr, key="conv_pct"):
    """Return {noise_pm: value} dict for the first row whose family field
    contains `family_substr`. Missing noises -> not in the dict.
    """
    out = {}
    for r in rows:
        if family_substr not in r["family"]:
            continue
        try:
            pm = int(r["noise_pm"])
            out[pm] = float(r[key])
        except (ValueError, KeyError):
            continue
    return out


def main():
    hip = load_csv(ANL / "noise_sweep_with_irc.csv")
    scine = load_csv(ANL / "noise_sweep_scine_xtb.csv")

    fig, (ax_hip, ax_scine) = plt.subplots(
        1, 2, figsize=(11, 4.5), sharey=True,
        gridspec_kw=dict(wspace=0.06),
    )

    # ---- HIP panel (left, reference) ----
    hip_gad = lookup(hip, "plain GAD dt=0.005", "conv_pct")
    hip_sella = lookup(hip, "Sella libdef", "conv_pct")

    ax_hip.plot([n for n in NOISES if n in hip_gad],
                [hip_gad[n] for n in NOISES if n in hip_gad],
                marker="o", color=C_GAD, lw=2.0, label="plain GAD dt=0.005")
    ax_hip.plot([n for n in NOISES if n in hip_sella],
                [hip_sella[n] for n in NOISES if n in hip_sella],
                marker="s", color=C_SELLA, lw=2.0, label="Sella libdef")
    ax_hip.set_title("HIP (reference)", fontsize=12)
    ax_hip.set_xlabel("Noise (pm)")
    ax_hip.set_ylabel("Convergence rate (%)\nstrict: n_neg==1 ∧ fmax<0.01")
    ax_hip.set_xticks(NOISES)
    ax_hip.set_ylim(-2, 100)
    ax_hip.legend(loc="upper right", fontsize=9, frameon=False)
    ax_hip.grid(alpha=0.3)

    # ---- SCINE / xTB panel (right) ----
    # Use the matched 15k-budget rows. Legacy 2k rows are kept in the CSV for
    # comparison but not plotted here.
    gad_strict = {}
    sella_strict = {}
    gad_irc = {}
    sella_irc = {}
    for r in scine:
        fam = r["family"]
        if "15k" not in fam:
            continue
        try:
            pm = int(r["noise_pm"])
        except (ValueError, TypeError):
            continue
        irc_v = r.get("irc_topo_pct_over_all", "")
        irc_f = float(irc_v) if irc_v not in ("", "nan", "None") else None
        if "plain GAD" in fam:
            gad_strict[pm] = float(r["conv_pct"])
            if irc_f is not None:
                gad_irc[pm] = irc_f
        elif "Sella libdef" in fam:
            sella_strict[pm] = float(r["conv_pct"])
            if irc_f is not None:
                sella_irc[pm] = irc_f

    ns_strict = [n for n in NOISES if n in gad_strict]
    ax_scine.plot(ns_strict, [gad_strict[n] for n in ns_strict],
                  marker="o", color=C_GAD, lw=2.0, label="GAD strict conv")
    ax_scine.plot([n for n in NOISES if n in gad_irc],
                  [gad_irc[n] for n in NOISES if n in gad_irc],
                  marker="o", color=C_GAD, lw=2.5, ls="--",
                  mfc="white", mec=C_GAD, mew=2,
                  label="GAD IRC TOPO-intended")
    ax_scine.plot([n for n in NOISES if n in sella_strict],
                  [sella_strict[n] for n in NOISES if n in sella_strict],
                  marker="s", color=C_SELLA, lw=2.0, label="Sella strict conv")
    ax_scine.plot([n for n in NOISES if n in sella_irc],
                  [sella_irc[n] for n in NOISES if n in sella_irc],
                  marker="s", color=C_SELLA, lw=2.5, ls="--",
                  mfc="white", mec=C_SELLA, mew=2,
                  label="Sella IRC TOPO-intended")

    # xTB single point + annotation
    # Both rows are 0% — annotate at 10 pm
    ax_scine.scatter([10], [0], marker="v", s=80, color=C_XTB, zorder=5)
    ax_scine.annotate(
        "xTB/GFN1 GAD+Sella\n(top-30 favorable)\n0/30 reach n_neg=1",
        xy=(10, 0), xytext=(40, 14),
        fontsize=8, color=C_XTB,
        arrowprops=dict(arrowstyle="-", color=C_XTB, lw=0.7),
    )

    ax_scine.set_title(
        "SCINE/DFTB0 (dt=0.007, 15k steps): GAD matches HIP strict-conv (89.9% @10pm); "
        "GAD IRC TOPO > Sella's (+12.9pp @10pm)",
        fontsize=10,
    )
    ax_scine.set_xlabel("Noise (pm)")
    ax_scine.set_xticks(NOISES)
    ax_scine.legend(loc="upper right", fontsize=8, frameon=False)
    ax_scine.grid(alpha=0.3)

    fig.suptitle(
        "Noise sweep, T1x test (n=287): HIP reference (left) vs SCINE/DFTB0 + xTB/GFN1 (right)",
        fontsize=12, y=1.02,
    )

    out_pdf = OUT / "fig_noise_sweep_scine_xtb.pdf"
    out_png = OUT / "fig_noise_sweep_scine_xtb.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
