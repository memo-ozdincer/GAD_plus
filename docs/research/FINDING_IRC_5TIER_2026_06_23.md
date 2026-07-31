# Finding: IRC 5-tier outcomes (test-287) — "unintended saddle" claim is wrong

**Date:** 2026-06-23. Source: fresh IRC re-validation on the test split (n=287),
`--method sella_hip --all-endpoints`, for the three paper methods.

## What was run
- GAD: `runs/irc_gad_test287/` (source TSs `gad_eckart_fmax_cart_test`, dt=0.003 Cartesian)
- Hybrid (damped Eckart eig tr=0.05): `runs/irc_hybrid_test287/{N}pm/`
- Sella (cart+Eckart untuned d=1): `runs/irc_sella_test287/` (stage-1 TS re-run `runs/sella_libdef_test287/`, since the test-287 Sella TS coords were not on disk)
- Per-sample tiers: `analysis_2026_04_29/irc_outcomes_5tier_test287.csv` (5166 rows).

## Tier logic (paper-style, ungated)
intended = `topology_intended`; partial = `topology_half_intended` (not intended);
unintended = converged ∧ ¬intended ∧ ¬partial ∧ ¬`topology_error`;
ts_error = ¬converged ∧ ¬intended ∧ ¬partial. (intended is **not** gated on
convergence — `--all-endpoints` — which is why intended% can exceed conv%.)

## Result: unintended ≈ 0 for ALL methods
| method | metric | 10 | 30 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|
| GAD    | intended | 253 | 255 | 250 | 220 | 176 | 130 |
| GAD    | partial  | 33 | 30 | 35 | 49 | 74 | 89 |
| GAD    | unintended | 0 | 0 | 0 | 0 | 0 | 0 |
| GAD    | ts_error | 1 | 2 | 2 | 18 | 37 | 68 |
| Hybrid | intended | 256 | 256 | 255 | 221 | 163 | 110 |
| Hybrid | partial  | 30 | 29 | 28 | 46 | 73 | 91 |
| Hybrid | unintended | 0 | 0 | 0 | 0 | 0 | 0 |
| Hybrid | ts_error | 1 | 2 | 4 | 20 | 51 | 86 |
| Sella  | intended | 254 | 256 | 250 | 209 | 143 | 68 |
| Sella  | partial  | 33 | 30 | 32 | 54 | 88 | 101 |
| Sella  | unintended | 0 | 0 | 0 | 0 | 2 | 2 |
| Sella  | ts_error | 0 | 1 | 5 | 24 | 54 | 116 |

Total unintended across all 5166 samples = **4**. Validated: Sella intended%
(88.5/89.2/87.1/72.8/49.8/23.7) matches the report (89.2/…/23.3).

## Implication for the manuscript (MUST FIX)
The draft's **abstract** ("Newton/RFO jumps cross basin boundaries and converge
to unintended saddles") and **Results II** ("the baseline's converged geometries
include a large unintended fraction that the IRC rejects, so its intended rate
falls below its convergence rate") are **not supported**: unintended ≈ 0 for Sella.

Corrected mechanism (data-backed): no method confidently finds the wrong saddle.
At 200 pm GAD beats Sella because it (1) recovers the full intended chemistry ~2×
as often (130 vs 68) and (2) fails to converge less (ts_error 68 vs 116); Sella's
loss is dominated by **non-convergence** and **one-sided (partial) IRC** (101 vs 89),
not wrong-basin saddles. The +21 pp intended headline is unchanged; only the
"why" changes. Rewrite Results II around partial-vs-failure, drop "unintended."

The `fig_outcome_stacked` third band should be labelled **"failed to converge"**
(plus partial as its own band), not "unintended or failed."
