# SCINE / xTB Second-Calculator Findings — 2026-05-11

Companion to `analysis_2026_04_29/noise_sweep_with_irc.csv` (HIP headline).
Question answered: do GAD and Sella's noise-sweep behaviors transfer to
non-ML calculators? Where and why do they diverge?

---

## 1. Bottom line

| Finding | Where to look |
|---|---|
| GAD strict-conv plateau-at-fmax≈0.01 is **intrinsic to GAD dynamics**, not the Eckart projection (Eckart vs no-Eckart within ±3 pp at every noise) | `noise_sweep_scine_xtb.csv` rows 1–12 |
| Sella's wrong-saddle failure mode is **dramatically worse on DFTB0** than on HIP. Sella IRC TOPO = **0.0%** at every noise level despite 87.5% strict-conv at 10 pm | `noise_sweep_scine_xtb.csv` rows 13–18 (irc_topo_pct_over_all column) |
| **GAD's IRC TOPO ≥ Sella's at every noise level on SCINE/DFTB0** (+9.4 pp at 10 pm). On HIP this only happens at high noise; on DFTB0 it's universal | `fig_noise_sweep_scine_xtb.pdf` right panel |
| GAD-on-SCINE TOPO successes are 93% subset of HIP-GAD successes — i.e. they are the "calculator-invariant" easy reactions | `cross_calc_topo_compare.csv` rows 1–6 |
| **xTB/GFN1 and GFN2 cannot be used as benchmark calculators** for T1x: forces at the HIP TS geometry are 4-15 eV/Å (vs DFTB0's ~3-9 eV/Å) and final IRC topologies have n_neg=7-22. Even the 30 most-favorable samples yield 0% TOPO | `noise_sweep_scine_xtb.csv` rows 19–20 |

---

## 2. Headline figure

`figures/fig_noise_sweep_scine_xtb.{pdf,png}` — two-panel: HIP reference
(left) vs SCINE/DFTB0 + xTB negative result (right), with both strict-conv
and IRC TOPO-intended lines for GAD and Sella.

The right-panel finding worth emphasizing: Sella's strict-conv curve
(green solid) looks correct in shape but lies entirely above its TOPO
curve (green dashed at zero). Sella reports "converged" on real n_neg=1
saddles — but they're the wrong saddles. GAD's TOPO curve (blue dashed)
sits above Sella's at every noise level.

---

## 3. Experimental setup

- Dataset: Transition1x **test split** (n=287). Paper hygiene per
  `feedback_test_split_only` memory.
- Calculators tested: SCINE/DFTB0 (full grid), xTB-GFN1/GFN2 (smoke +
  favorable-sample panel only).
- Search budgets matched to HIP canonical: 2000 steps × 287 samples
  × {10, 30, 50, 100, 150, 200} pm noise.
- Convergence criterion: strict `n_neg==1 ∧ fmax<0.01` (matches HIP).
- IRC validation: Sella IRC forward+reverse from each strict-converged
  TS, scored by bond-graph isomorphism vs known T1x R/P. Same procedure
  as `analysis_2026_04_29/noise_sweep_with_irc.csv` HIP rows.

Scripts (deliberately not refactors of HIP scripts):

- `scripts/gad_smoke.py` — parallel GAD runner accepting `--backend {scine,xtb}`.
- `scripts/sella_smoke.py` — parallel Sella runner, saves `final_coords_flat`
  in summary parquets.
- `scripts/scine_irc_validate.py` — IRC validator that pulls TS coords
  from Sella summary parquets (or GAD trajectory parquets), runs IRC,
  scores TOPO.
- `scripts/aggregate_scine_xtb_main.py` — produces the 5-row CSV.
- `scripts/cross_calc_topo_compare.py` — HIP vs SCINE sample-level join.
- `scripts/figures_scine_xtb_panel.py` — two-panel figure.

SLURM wrappers: `scripts/main_scine_{gad,sella}.slurm`,
`scripts/main_scine_sella_with_coords.slurm` (the rerun with coords),
`scripts/main_scine_gad_no_eckart.slurm`, `scripts/main_xtb_favorable.slurm`,
`scripts/scine_irc.slurm`, `scripts/scine_sella_irc.slurm`.

---

## 4. The four big findings, with mechanism

### 4.1 GAD's plateau-at-fmax≈0.01 is GAD's, not Eckart's

`noise_sweep_scine_xtb.csv` shows the Eckart and no-Eckart SCINE GAD
rows within ±3 pp at every noise level. At 10 pm: 32.4% vs 35.2% strict
conv. The plateau is a property of the Euler GAD dynamics meeting a
gradient noise floor in v_1 — projection out the TR modes only changes
the n_neg counting, not the dynamics' fixed point.

This is consistent with the HIP-side paper-narrative claim that the
plateau "is not a dt artifact." Now also: "is not a projection
artifact." Mechanism is the GAD vector field itself.

### 4.2 Sella's wrong-saddle problem dominates on DFTB0

Sella on SCINE/DFTB0 finds n_neg=1 saddles at 87.5% rate at 10 pm noise,
but **none** of them are the intended R↔P TS. IRC traces from those
saddles converge to the same minimum from both directions (forward and
reverse both match product). The saddles Sella locates are real and
near-stationary, just not the right ones.

The paper narrative memory predicts exactly this: "Sella's bimodal RMSD
at high noise: trust-region Newton step jumps basins; at 200 pm, 63% of
Sella failures end at n_neg≥2 (wrong saddle), not 'saddle but loose.'"
On HIP this only flares at high noise. On DFTB0 — where the Hessian is
not paper-grade — it dominates at every noise level.

Verified by manually checking sample 11 at 10 pm: Sella's reported TS is
geometrically much closer to the product than to the reactant; the
n_neg=1 mode points away from R. IRC integrates downhill to product in
both directions.

### 4.3 GAD's TOPO ≥ Sella's TOPO at every SCINE noise level

| Noise | GAD strict / IRC TOPO | Sella strict / IRC TOPO |
|------:|----------------------:|------------------------:|
| 10 pm | 32.4% / **9.4%** | 87.5% / **0.0%** |
| 30 pm | 12.2% / 0.7% | 73.2% / 0.0% |
| 50 pm | 3.1% / 0.3% | 55.7% / 0.0% |
| 100 pm | 0.7% / 0.0% | 31.7% / 0.0% |
| 150 pm | 0.0% / 0.0% | 14.3% / 0.0% |
| 200 pm | 0.0% / 0.0% | 3.8% / 0.0% |

GAD's lead at 10 pm: +9.4 pp. This is the SCINE analog of HIP's +21 pp
at 200 pm — same direction, plays out at lower absolute level because
DFTB0 has worse Hessian quality than HIP across the board.

The headline narrative strengthens: GAD's edge is about **avoiding
wrong-saddle convergence**, and it scales with how noisy/inexact the
Hessian is. HIP-exact Hessian is the best case for Sella; anything less
hurts Sella more than it hurts GAD.

### 4.4 SCINE-GAD TOPO successes are 93% within HIP-GAD successes

| Method | HIP TOPO@10pm | SCINE TOPO@10pm | both | P(SCINE\|HIP) | P(HIP\|SCINE) |
|---|---:|---:|---:|---:|---:|
| GAD   | 255 | 27 | 25 | 0.10 | 0.93 |
| Sella | 256 |  0 |  0 | 0.00 |  —   |

When GAD on SCINE/DFTB0 does find the right saddle (27 samples at 10
pm), 25 of those 27 are also found by GAD on HIP. Only 2 are SCINE-only.

Interpretation: the 27 SCINE-GAD successes are samples whose DFTB0
saddle happens to lie near HIP's. The 230 HIP-only samples are reactions
where DFTB0's PES disagrees with HIP's enough that GAD's plateau-aware
trajectory still lands on a calculator-specific (wrong-for-T1x) saddle.

This pins down that the cross-calculator decay is a PES-disagreement
effect, not a GAD bug.

---

## 5. xTB negative result

At HIP-TS with zero noise (the smoke probe at job 60772097):
- DFTB0:    fmax ~3-9 eV/Å, Hessian eig range ~[-65, +170]
- GFN1:     fmax ~4-10 eV/Å, eig range ~[-20, +190] (similar magnitude
            to DFTB0)
- GFN2:     fmax ~6-15 eV/Å, eig range ~[-625, +315] (4× stiffer)

For both GFN1 and GFN2, neither GAD nor Sella locates a meaningful TS
near HIP-TS+10pm noise. Final geometries are at n_neg=7-22 on every
sample. Even on the 30 samples where GFN1's fmax-at-HIP-TS is lowest
(3.1-4.2 eV/Å), TOPO = 0/30 on both methods.

Documented in `noise_sweep_scine_xtb.csv` rows 19-20 and the panel
annotation in `fig_noise_sweep_scine_xtb.pdf` right panel.

What would unblock xTB for benchmarking: starting from xTB-relaxed
reactant/product (not HIP-noised TS), so the search is on the xTB
reaction graph. Not in scope here.

---

## 6. Data files (ground-truth pointers per `feedback_ground_truth_pointers`)

- Summary table:           `analysis_2026_04_29/noise_sweep_scine_xtb.csv`
- Cross-calculator table:  `analysis_2026_04_29/cross_calc_topo_compare.csv`
- xTB fmax @ HIP-TS:       `analysis_2026_04_29/xtb_gfn1_fmax_at_hipts.csv`
- Top-30 xTB indices:      `analysis_2026_04_29/xtb_favorable_top30.txt`
- Figure:                  `figures/fig_noise_sweep_scine_xtb.{pdf,png}`

Raw parquets:
- `runs/main_scine_gad_60772085/noise{10,30,50,100,150,200}pm/summary_*.parquet`
- `runs/main_scine_gad_neck_60774250/...` (no-Eckart ablation)
- `runs/main_scine_sella_c_60776830/...` (Sella with final_coords_flat)
- `runs/main_xtb_favorable_60774467/{gad,sella}/summary_*.parquet`
- `runs/scine_irc_60776605/gad/irc_validation_*.parquet` (GAD TOPO)
- `runs/scine_sella_irc_60777076/sella/irc_validation_*.parquet` (Sella TOPO)

Memory: `project_scine_xtb_main_findings_2026_05_11.md`,
`project_scine_xtb_install.md`, `project_scine_xtb_smoke_findings.md`.

---

## 7. What's NOT closed (and why that's fine)

- **xTB with xTB-relaxed starting points** — out of scope per `feedback_compute_strategy`
  (deep-not-wide). The negative result with HIP-noised TS is itself the
  finding; reframing the benchmark for xTB is a separate piece of work.
- **SCINE other functionals (PM6, AM1, DFTB2, DFTB3)** — DFTB0 is the
  closest to HIP per the smoke comparison; the others would mostly
  add nuance to "less-accurate-Hessian → more wrong-saddle." If the
  paper needs that nuance, run them; otherwise DFTB0 carries the
  point.
- **IRC sensitivity** (bond-cutoff 1.2 → 1.3, rmsd_threshold 0.3 → 0.5)
  — cheap re-score from existing parquets if reviewers ask.
