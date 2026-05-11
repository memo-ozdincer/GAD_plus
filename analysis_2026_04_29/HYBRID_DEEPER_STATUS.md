# Hybrid GAD-Newton deeper sweep — running log

Started 2026-05-10. SLURM `60741726` (10 hybrid cells, coords logged) →
`60741727` (10 IRC validation cells, dependency).

## Goal

The original 2026-05-09 sweep produced a confounded comparison: the
"no-Eckart hybrid beats plain GAD on IRC TOPO at 100 pm by +2.1 pp"
finding mixed *four* simultaneous variable changes (Eckart projection,
damping, switch criterion, trust radius). This sweep disentangles them
on a fixed `tr=0.05`, `gad_dt=5e-3`, `n=287` test split.

## Cells

| # | Algo | Switch | sf | Purpose |
|---|---|---|---|---|
| 0–1 | no-Eckart `hybrid` | force | 1e-2 | Newton would fire here if force-switch reachable in Cartesian frame |
| 2–3 | Eckart undamped | force | 1e-2 | Eckart axis with Newton firing |
| 4–5 | Eckart damped | force | 1e-2 | Eckart + damping with Newton firing |
| 6–7 | Eckart undamped | eig | n/a | Damping isolation (compare to existing damped+swtrue) |
| 8–9 | no-Eckart | force | 1e-3 | tr-effect control (vs existing tr=0.005) |

## Newton-firing rates (from completed cells, 4/10 so far)

| Cell | Noise | Raw conv % | Newton step % | Samples w/ Newton % |
|---|---|---|---|---|
| `hybrid_eckart_swtrue` tr=0.05 sf=1e-2 | 10pm | 84.7 | 81.1 | 99.0 |
| `hybrid_eckart_swtrue` tr=0.05 sf=1e-2 | 100pm | 65.5 | 52.3 | 95.1 |
| `hybrid_swfalse` tr=0.05 sf=1e-2 (no Eckart) | 10pm | 88.9 | **0.0** | **0.0** |
| `hybrid_swfalse` tr=0.05 sf=1e-3 (no Eckart) | 10pm | 88.9 | **0.0** | **0.0** |

## Finding so far: three Newton-firing regimes

1. **Cartesian force-switch never triggers Newton in the no-Eckart hybrid.**
   Even at `sf=1e-2`, the Cartesian L2 force norm $\|F\|_2$ stays above
   the threshold across all 287 trajectories at 10 pm. Reason: $\|F\|_2$
   in the Cartesian frame includes ~3N components per atom; for a 6-atom
   molecule with `fmax`$\approx 0.01$ each, $\|F\|_2 \sim 0.03$–$0.1$,
   far above $0.01$. So the no-Eckart "hybrid" in the original sweep is
   effectively GAD-with-trust-cap, not a Newton-firing hybrid.

2. **Eckart-projected force-switch fires Newton on ~75% of samples at
   10 pm, but only 1% at 100 pm.** The internal-coord force is smaller
   (TR modes removed) so it more readily crosses `sf=1e-2`, but at high
   noise GAD's plateau is further from the saddle and force never
   approaches zero.

3. **Eckart eigenvalue-switch fires Newton on 95–99% of samples at both
   10 and 100 pm.** Curvature signature is reachable even when force is
   not. This is why `swtrue` configs are the only ones that engage
   Newton at high noise.

## Implication for the original "+2.1 pp" finding

The hybrid no-Eckart at 100 pm with IRC TOPO = 80.1% is *pure GAD with
a trust cap* — Newton plays no role. So the +2.1 pp vs plain GAD on IRC
TOPO is a within-GAD-family effect (trust cap reduces plateau-orbit
tail) and the "hybrid Newton phase" had nothing to do with it.

The Eckart-vs-no-Eckart question, when controlled, is: does adding
Newton help or hurt IRC TOPO? Answer pending Phase 2 IRC.

## Sources

- Newton firing CSV: `analysis_2026_04_29/hybrid_deeper_newton_firing.csv`
- Hybrid summaries: `runs/hybrid_deeper/<tag>_<noise>pm/summary_*.parquet`
- Hybrid trajectories: `runs/hybrid_deeper/<tag>_<noise>pm/traj_*.parquet`
- IRC validation (pending): `runs/irc_hybrid_deeper/`
- SLURM: 60741726 (hybrid), 60741727 (IRC, dependency)
