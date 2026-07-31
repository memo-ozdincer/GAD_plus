# HEADLINE — GAD vs Sella with HIP analytic Hessian (2026-05-01)

One-page summary. For the full report see
`/lustre06/project/6033559/memoozd/GAD_plus/IRC_TEST_2026-04-29.pdf`.

## The number

**At 200pm TS-noise on T1x test (n=287), GAD outperforms best Sella
by +21.3pp on IRC TOPO-intended (44.6% vs 23.3%).**

Full IRC TOPO-intended table:

| method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| GAD dt=0.005 (5k steps)        | 88.9 | 89.2 | 88.9 | 78.0 | **61.7** | **44.6** |
| Sella libdef (cart, every-step H) | **89.2** | **89.2** | 87.5 | 72.5 | 49.8 | 23.3 |
| **best-GAD − best-Sella**     | −0.7 | 0.0 | +1.4 | **+5.9** | **+11.9** | **+21.3** |

Source: `runs/test_irc/<method>/irc_validation_sella_hip_allendpoints_<noise>pm.parquet`.

## What this means

1. **At low noise (≤50pm)** GAD and Sella tie — both methods are near
   the saddle and both work.
2. **As noise grows** Sella degrades faster than GAD. By 200pm, Sella
   converges only half as often as GAD on the chemistry ground truth.
3. **The IRC gap is wider than the raw-conv gap** (+21.3 vs +17.4pp at
   200pm). Reason: IRC catches Sella's wrong-saddle failures (Sella
   "converges" at higher-order saddles); GAD's plateau-orbit failures
   stay in the right basin and are partly TOPO-validated anyway.

## Why GAD wins (mechanism)

| mechanism | Sella | GAD |
|---|---|---|
| Step character | Newton, $\mathcal{O}(0.1\,\text{\AA})$ jump | Euler, $\mathcal{O}(dt)$ walk |
| Failure mode at 200pm | 63% wrong-saddle ($n_{\text{neg}}\ge 2$, fail $f_{\max}{=}0.37$) | 62% plateau-orbit ($n_{\text{neg}}=1$, fail $f_{\max}{=}0.046$) |
| Recovery from far init | Trust-region jumps may land in wrong basin | Smooth descent stays in basin |

GAD's Euler step doesn't jump basins — it walks through them. The
trade-off is GAD plateaus at $f_{\max}{\approx}0.01$ (eigenvector noise
floor) and never goes below; Sella's Newton step crushes forces to ~0
when it works at all. This is why GAD gets higher RMSDs at convergence
(within ~0.15Å of saddle) but Sella gets bimodal RMSDs (either 0Å
"nailed" or 0.5Å "wrong saddle").

## Robustness — handicap arguments preempted

| concern | resolution |
|---|---|
| "You gave Sella only 2000 steps" | Sella libdef 5000 steps: within ±2pp of 2000 (`runs/test_sella_extended/`) |
| "Sella library default has nsteps_per_diag=3" | d=3 vs d=1: only −3pp at 100pm, −6pp at 150pm. Same trend (`runs/test_hessfreq/`) |
| "You tuned Sella sub-optimally" | We tested {default, libdef, internal, lson} — libdef is best, that's what we report |
| "Sella w/o Hessian gets 5%" | Yes — that's the point. Both methods need HIP H. The paper is about MLIP-armed optimizers |
| "GAD's high RMSD" | Plateau orbit in right basin. NR polish (60314225 in flight) tests if it can be tightened |
| "Larger noise" | 300pm: GAD 22 vs Sella 20 (strict); 500pm: GAD 8 vs Sella 6. Both saturate beyond 1000pm. 200pm is the right stress test |

## Two reporting framings

**Framing A — most-faithful** (both tuned, both armed with HIP H every step,
same coord system, same step budget): see headline table above.

**Framing B — most-out-of-the-box** (GAD canonical dt=0.003, Sella library
defaults d=3 + fmax<0.05 + internal coords): GAD 88.2/87.1/84.7/69.3/51.6/34.8
vs Sella default 94.4/90.6/84.7/63.8/--/-- (fmax<0.05 criterion). Same shape;
GAD wins at high noise.

## Key sources

- `IRC_TEST_2026-04-29.pdf` — current report (27 pages, 2026-05-01)
- `analysis_2026_04_29/FINDINGS.md` — narrative findings, every number sourced
  - §15: IRC headline result
  - §16: Sella step-budget experiment
  - §17: Sella Hessian-frequency sweep
  - §18: huge-noise probe
- `analysis_2026_04_29/DATA_INDEX.md` — query guide for every CSV/parquet
- `LIVE_STATUS.md` — current job state
- `~/.claude/.../memory/project_paper_narrative_2026_05_01.md` — strategic context

## Still pending

- NR-GAD polish (60314225, n=80, 18h budget) — tests if NR breaks plateau
- Midpoint single-ended (60314226, 96G mem, 18h) — tests "from R+P midpoint" claim
- Sella matched-budget 10k cells at 150/200pm — landing
- IRC sensitivity checks (0.3 → 0.5 RMSD threshold; 1.2 → 1.3 bond cutoff) — cheap re-score from existing parquets
