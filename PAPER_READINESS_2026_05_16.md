# Paper-readiness checklist — 2026-05-16

Status of all findings, figures, and tables in `BENCHMARK_REPORT_2026-05-16.pdf`
as of last cron refresh. Use this when planning paper sections / decide what
needs more compute vs. what's ready to write.

## Solid (ready to write up as-is)

| Finding | Source | Confidence |
|---|---|---|
| **Headline 4-axis** (best-of-family vs noise, noised TS) | `master_2026_05_11.csv` + live pooled refinements | High — full $n=287$ on canonical cells |
| **Sella naming convention** (3-axis, with historical map) | TeX §2 | Documented, internally consistent |
| **Convergence-criterion family table** (ASE / Gaussian / project / Sella) | TeX §3 | Factual, sourced |
| **Gaussian threshold conversions** ($E_h/a_0 \to$ eV/Å) | TeX §3 | Authoritative, sourced |
| **Five swappable 4-axis figures** under each fmax threshold | `fig_main_4axis_fmax0p{05,023,01,005,001}.pdf` | Reproducible from `master_2026_05_16.csv` + `threshold_sweep_2026_05_16.csv` |
| **The fmax plateau in GAD** | TeX §5 | Now backed by **10k-step probe** (see below) |
| **Long-budget probe** (`fig_longbudget.pdf`) | $n=86$ for GAD, less for hybrid/Sella | **Strong** — GAD 0% at fmax<0.005 even with 5$\times$ budget |
| **Pareto plane per noise** | `fig_pareto_per_noise.pdf` | High |
| **Wall-time rankings (lollipop low/high)** | `fig_ranking_lollipop_{low,high}.pdf` | High |
| **IRC TOPO recovery** (Sella catches wrong saddles) | `fig_topo_recovery.pdf` | High, prior data |
| **RMSD-to-known-TS** (hybrid wins p95 at high noise) | `fig_rmsd_to_ts.pdf` + Table 6 | High |
| **Sella d=3 vs d=1**: d=3 wins TS conv low/mid; d=1 wins IRC TOPO + high noise | TeX §7 + `fig_d1_vs_d3.pdf` | Trajectory documented; **d=3 200pm continues to settle near 22%** vs d=1's 27.2% |

## Partial — confidence "n+" but acceptable for paper

| Cell | Latest | n / 287 | Notes |
|---|---|---|---|
| Sella d=3 @ 200 pm (TS conv) | 23.0% | 204 (pooled) | Trajectory monotonically settling; final likely 22–24% |
| Sella internal @ 150 pm | 41.8% | 79 (solo) | Replaces sample-biased partial 29.3%; +12 pp correction |
| Sella internal @ 200 pm | 16.1% | 161 (pooled) | Below original partial 19.9% |
| Sella libdef midpoint @ 0 pm (NEW) | 52.4% | 126 | New starting-condition data point |
| Hybrid damped reactant @ 0 pm (NEW) | 0.0% | 70 | Mechanism: eig-switch never fires from reactant in 2000 steps |
| Hybrid undamped reactant @ 0 pm (NEW) | 0.0% | 70 | Same mechanism |
| GAD dt=0.005 ×10k @ 50 pm (longbudget) | 0% fmax<0.005 | 86 | Plateau confirmed at $5\times$ budget |
| Sella libdef ×10k @ 50 pm | 81.8% fmax<0.005 | 11 | Newton scales correctly; small n still |
| Hybrid damped ×10k @ 50 pm | 38.5% fmax<0.005 | 13 | Newton scales correctly; small n still |

## In flight (will land in 0–6 h)

| Job | Cells | What's expected |
|---|---|---|
| 61087603 / 61087774 (wave 1) | 9 main + 3 longbudget | Full $n=287$ on midpoint, internal 150pm; partial on d=3 / internal 200pm (10k @ Sella libdef will time out at ~$n=50$) |
| 61088001 (safety net) | 6 partitions | Each finishes its $\sim 96$-sample slice; pool with main fills d=3 200pm and internal 200pm to $\sim 287$ |
| 61091399 (wave 2: 10k from reactant) | 2 cells | Tests "hybrid 0% from reactant is budget-limited" prediction. 12h walltime. Will produce $\sim 50$ samples each. |

## Pre-wired but not yet submitted

| Script | Submit when | Closes |
|---|---|---|
| `scripts/build_pooled_summaries_2026_05_16.py` | Wave 1 finishes (sees both main + safety-net parquets) | Canonical pooled summary parquets at fixed paths |
| `scripts/run_irc_comprehensive_2026_05_16.slurm` (4 cells) | After `build_pooled_summaries_2026_05_16.py` | (a) d=1 vs d=3 chemistry verdict at 200pm; (b) internal 150/200pm IRC TOPO; (c) midpoint @ 0pm IRC TOPO — real-basin vs wrong-saddle |

## Open algorithm work (R5; **not started**)

Mode-overlap-aware switching for the hybrid. Currently the eig-switch only
checks $n_\mathrm{neg}=1$; an additional gate on eigenvector continuity
across steps (overlap $> 0.9$) would prevent Newton from firing on a
spurious mode swap. Expected to reduce hybrid wrong-saddle rate from 44% to
$\lesssim 30$% at 200 pm and close the −5.9 pp IRC TOPO gap vs plain GAD.

Files to touch:
  - `src/gadplus/search/hybrid_gad_damped_eigfollownewton_eckart.py` line 479
    (the `if switch_based_on_hessian_eigval:` block — add eigvec-continuity
    check)
  - `scripts/hybrid_gad_newton_runner.py` (cache previous eigvec, pass as
    arg, log overlap)

Risk: invasive change touching the canonical hybrid step. Recommend an
isolated worktree + smoke test on $\le 10$ samples before committing.

## Suggested paper-section mapping

1. **Methods**: Sella naming convention, convergence-criterion family
   (Gaussian/ASE/project), Eckart projection details.
2. **Headline result**: §1 of report — 4-axis figure at fmax<0.01; tables 1-2.
3. **Starting condition matters**: §2 — reactant bar chart + midpoint
   companion + the hybrid-from-reactant mechanism.
4. **The fmax plateau (mechanistic argument for hybrid Newton)**: §5 of report
   — fmax-plateau figure + the 10k-budget probe demonstrating intrinsicness.
5. **Pareto + rankings**: §4 (wall vs TOPO).
6. **Chemistry validation**: §6 — IRC TOPO recovery + RMSD-to-known-TS.
7. **Discussion**: d=3 surprise as a methodological cautionary tale; the
   partial-data bias correction story (29.3% → 41.4% on Sella internal 150pm
   is a 12 pp correction from sample-biased SLURM-log extraction).

## What's *not* worth more compute

- Tighter than fmax<0.001 — unreachable on HIP within reasonable budgets.
- More noise levels — 6-level grid is already saturating the noise-axis story.
- Per-molecule deep-dive — would be a separate paper.
- More dt values — `test_dtgrid` covers dt ∈ {0.003, 0.004, 0.005, 0.006, 0.007, 0.008}; the 0.005-0.007 plateau is well-mapped.
