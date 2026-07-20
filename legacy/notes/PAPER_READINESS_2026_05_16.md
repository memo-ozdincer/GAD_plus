# Paper-readiness checklist — 2026-05-16 (updated end-of-day)

Status of all findings, figures, and tables in `BENCHMARK_REPORT_2026-05-16.pdf`.
Use this when planning paper sections / decide what needs more compute vs. what's
ready to write.

## Solid (ready to write up as-is)

| Finding | Source | Confidence |
|---|---|---|
| **Headline 4-axis** (best-of-family vs noise, noised TS) | `master_2026_05_16.csv` | High — full $n=287$ on canonical cells |
| **Sella naming convention** (3-axis, with historical map) | TeX §2 | Documented, internally consistent |
| **Convergence-criterion family table** (ASE / Gaussian / project / Sella) | TeX §3 | Factual, sourced |
| **Gaussian threshold conversions** ($E_h/a_0 \to$ eV/Å) | TeX §3 | Authoritative, sourced |
| **Five swappable 4-axis figures** under each fmax threshold | `fig_main_4axis_fmax0p{05,023,01,005,001}.pdf` | Reproducible from `master_2026_05_16.csv` + `threshold_sweep_2026_05_16.csv` |
| **Comprehensive fmax table** (9 methods × 6 noises × 5 thresholds, per-cell best bolded) | TeX §3 | Full $n=287$ on canonical cells |
| **GAD plateau is intrinsic** (10k-step probe @ 50pm) | `fig_longbudget.pdf` + TeX §5 | **n=287 final for GAD: 0% at fmax<0.005** |
| **fmax-decay summary figure** (GAD plateau, hybrid rescues) | `fig_fmax_decay_gad_vs_hybrid.pdf` | High |
| **Pareto plane per noise** | `fig_pareto_per_noise.pdf` | High |
| **Wall-time rankings (lollipop low/high)** | `fig_ranking_lollipop_{low,high}.pdf` | High |
| **IRC TOPO recovery** (Sella catches wrong saddles) | `fig_topo_recovery.pdf` + Table 2 | High |
| **RMSD-to-known-TS** (hybrid wins p95 at high noise) | `fig_rmsd_to_ts.pdf` + Table 6 | High |
| **Sella d=3 vs d=1**: d=3 wins TS conv low/mid; d=1 wins IRC TOPO + high noise | TeX §7 + `fig_d1_vs_d3.pdf` | **Now n=287 at 200pm**: d=3 23.3% raw / 22.0% IRC; gap −3.9 raw / −1.3 IRC |
| **Hybrid from reactant = 2.1% (n=287)** + 10k-step disproof of budget hypothesis | TeX §"Hybrid from reactant" + `FINDING_HYBRID_REACTANT` | Geometric, not budget-limited |
| **Sella midpoint @ 0pm: 46.7% raw / 46.7% IRC TOPO** (n=287) | TeX §"Companion: midpoint" + `irc_followup_2026_05_16.csv` | Recovery −0.0 pp; per-conv TOPO 91% — clean starts |
| **Sella internal d=1 @ 200pm: 13.9% raw / 16.0% IRC** | Table 1 + Table 2 | **+2.1 pp recovery** — unique to internal coords |

## Partial — acceptable for paper with $n$ noted

| Cell | Latest | n / 287 | Notes |
|---|---|---|---|
| Sella internal @ 150 pm | 41.7% (solo) | 172 | Original cell timed out at n=172 with no parquet; refill in flight |
| Sella libdef ×10k @ 50 pm | 77.3% fmax<0.005 | 66 | Newton scales correctly; SLURM timed out at 12h |
| Hybrid damped ×10k @ 50 pm | 47.2% fmax<0.005 | 72 | Newton scales correctly; SLURM timed out at 12h |

## In flight

| Job | Cells | ETA |
|---|---|---|
| 61166201 (Sella internal 150pm refill, 4-way partitioned) | 4 cells | ~4–5 h remaining (started 16:38, 41 min in) |

## Pre-wired but not yet submitted

| Script | Submit when | Closes |
|---|---|---|
| Sella 150pm IRC follow-up (will mirror `run_irc_parallel_2026_05_16.slurm`) | After 61166201 lands + `build_pooled_summaries` for 150pm | The last `---` in Table 2 (Sella internal d=1 @ 150pm IRC TOPO) |

## Open algorithm work (R5; **not started**)

Mode-overlap-aware switching for the hybrid. Currently the eig-switch only
checks $n_\mathrm{neg}=1$; an additional gate on eigenvector continuity
across steps (overlap $> 0.9$) would prevent Newton from firing on a
spurious mode swap. Expected to reduce hybrid wrong-saddle rate from 44% to
$\lesssim 30$% at 200 pm and close the $-5.9$ pp IRC TOPO gap vs plain GAD.

Files to touch:
  - `src/gadplus/search/hybrid_gad_damped_eigfollownewton_eckart.py` line 479
  - `scripts/hybrid_gad_newton_runner.py` (cache previous eigvec, pass as
    arg, log overlap)

Risk: invasive change touching the canonical hybrid step. Recommend an
isolated worktree + smoke test on $\le 10$ samples before committing.

## Suggested paper-section mapping

1. **Methods**: Sella naming convention, convergence-criterion family
   (Gaussian/ASE/project), Eckart projection details.
2. **Headline result**: §1 of report — 4-axis figure at fmax<0.01; Tables 1-2.
3. **Starting condition matters**: §2 — reactant bar chart + midpoint companion
   (incl. IRC TOPO 46.7%) + the hybrid-from-reactant geometric finding.
4. **The fmax plateau (mechanistic argument for hybrid Newton)**: §5 of report
   — fmax-plateau figure + fmax-decay figure + 10k-budget probe demonstrating
   intrinsicness.
5. **Comprehensive fmax table** (all methods × noises × thresholds) — supports
   the plateau claim numerically.
6. **Pareto + rankings**: §4 (wall vs TOPO).
7. **Chemistry validation**: §6 — IRC TOPO recovery + RMSD-to-known-TS.
8. **Discussion**: d=3 surprise as a methodological cautionary tale; the
   partial-data bias correction story (Sella internal 150pm 29.3% → 41.7%);
   Sella internal's unique +2.1 pp IRC TOPO recovery at 200pm.

## What's *not* worth more compute

- Tighter than fmax<0.001 — unreachable on HIP within reasonable budgets.
- More noise levels — 6-level grid is already saturating the noise-axis story.
- Per-molecule deep-dive — would be a separate paper.
- More dt values — `test_dtgrid` covers dt ∈ {0.003, 0.004, 0.005, 0.006, 0.007, 0.008}; the 0.005-0.007 plateau is well-mapped.
- Bigger 10k-budget refill — partial data already confirmed the qualitative finding (hybrid + Sella reach fmax<0.005, GAD doesn't).

## Late-day commit lineage (for traceability)

| Commit | What |
|---|---|
| `9d96c12` | End of wave 1: final n=287 numbers for the main cells |
| `f78382c` | Post-wave cleanup of stale "in flight" language |
| `ce7a458` | Comprehensive fmax table + fmax-decay figure (user-requested re-add) |
| `8f9acec` | IRC parallel batch SLURM + merger + 150pm refill SLURM |
| `bd0d526` | merge_irc_parallel: per-conv vs full-N reporting |
| `b7d218b` | Midpoint IRC TOPO landed: 46.7% / recovery -0.0 pp |
| `32f27f7` | All 3 IRC follow-up cells landed: d=3 200pm 22.0, internal 200pm 16.0, midpoint 46.7 |
