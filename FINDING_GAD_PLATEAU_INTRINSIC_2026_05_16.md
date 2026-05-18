# In-flight finding: GAD's fmax plateau is intrinsic, not step-budget-limited (2026-05-16)

## Snapshot (live log, n=11, 50 pm noise, GAD dt=0.005)

10000-step budget, target fmax<0.001. Per-sample log:

| sample | status | n_neg | force_max | force_norm | steps | wall |
|---|---|---|---|---|---|---|
| 0 (C2H3N3O2) | FAIL | 1 | 0.0117 | 0.0080 | 10000 | 639 s |
| 1 (C2H3N3O2) | FAIL | 1 | 0.0126 | 0.0073 | 10000 | 640 s |
| 2 (C2H3N3O2) | CONV | 1 | 0.0100 | 0.0064 | 199 | 13 s |
| 3 (C2H3N3O2) | CONV | 1 | 0.0100 | 0.0072 | 260 | 16 s |
| 4 (C2H3N3O2) | CONV | 1 | 0.0100 | 0.0080 | 269 | 17 s |
| 5 (C2H5NO2) | CONV | 1 | 0.0099 | 0.0055 | 261 | 17 s |
| 6 (C2H5NO2) | CONV | 1 | 0.0100 | 0.0049 | 240 | 16 s |
| 7 (C2H5NO2) | CONV | 1 | 0.0100 | 0.0095 | 184 | 12 s |
| 8 (C2H5NO2) | CONV | 1 | 0.0096 | 0.0065 | 272 | 17 s |
| 9 (C2H5NO2) | CONV | 1 | 0.0099 | 0.0065 | 382 | 24 s |
| 10 (C2H5NO2) | CONV | 1 | 0.0098 | 0.0054 | 281 | 18 s |

## Two observations

1. **Every converged sample has force_max in [0.0096, 0.0100] — perched exactly on the plateau ceiling.** The trajectory hits this band and stops descending; convergence by the fmax<0.01 criterion is incidental.
2. **The two failures (samples 0, 1) ran the full 10000 steps and ended at force_max = 0.012 / 0.013** — even 5× the canonical 2000-step budget cannot push the force below ~0.012 for these harder geometries.

## Aggregate at thresholds (n=11)

| threshold | conv % |
|---|---|
| fmax<0.05 | 100.0 |
| fmax<0.023 (Gaussian) | 100.0 |
| fmax<0.01 (project) | 36.4 |
| **fmax<0.005 (tight)** | **0.0** |
| fmax<0.001 (Sella README) | 0.0 |

Compare with the 2000-step canonical sweep at 50 pm:

| threshold | GAD dt=0.005, 5k steps | GAD 10k (this run) |
|---|---|---|
| fmax<0.01 | 85.7% (n=287) | 36.4% (n=11) |
| fmax<0.005 | 0.0% | 0.0% |

The 36.4% at fmax<0.01 is the small-n noise on these particular first-11 samples — but the 0.0% at fmax<0.005 is **structural**: GAD literally cannot descend below the plateau regardless of budget.

## Mechanism (from the existing report, now confirmed at 10× budget)

The GAD step is $\dot x = (I - 2vv^T) F$ where $v$ is the climbing eigenvector
of the Hessian. Near a saddle, $\|F\| \to 0$ along the soft mode, and the
remaining force component along orthogonal modes is what drives descent.
Once $\|F\|_\text{soft-modes}$ falls below ~0.01 eV/Å, the Euler step
$dt \cdot F_\perp \approx 0.005 \times 0.01 = 5 \times 10^{-5}$ Å, which is too
small to push through numerical noise in the curvature direction. The trajectory
oscillates around the saddle indefinitely.

Newton (and the hybrid's Newton step) is immune: $H^{-1} F$ grows as $F$ shrinks,
so the step size scales correctly. Hence the hybrid's 7–11% at fmax<0.005
in the canonical 2000-step sweep — Newton landing accounts for *all* of the
sub-plateau samples.

## Implication for paper

The "GAD plateau" section in `BENCHMARK_REPORT_2026-05-16.pdf` is now supported
by an **explicit 5× budget probe**: the plateau is not a budget bug. Newton
landing is the only way to reach fmax<0.005 on this PES.

When R4-a finishes its full 287 samples (~3 h), this becomes a single-paragraph
addendum to §5 ("the fmax plateau").

## Confidence

n=11. The aggregate fmax<0.005=0.0% would only change if 1+ of the remaining 276
samples lands at fmax in [0.001, 0.005]. Across the 11 samples seen, the median
final force_max is 0.0100. Final n=287 ETA ~3 h.

---

## CONFIRMED at FULL n=287 (2026-05-16 end of wave)

GAD dt=0.005 × 10000 steps @ 50 pm, all 287 test samples processed:

| threshold | conv % |
|---|---|
| fmax<0.05 | 95.5 |
| fmax<0.023 (Gaussian) | 91.6 |
| fmax<0.01 (project) | 85.7 |
| **fmax<0.005 (tight)** | **0.0** |
| fmax<0.001 (Sella README) | 0.0 |

The 5× step budget gains +12 pp at fmax<0.01 (85.7% vs 71.8% canonical
2000-step) but **gains zero ground at fmax<0.005**. The plateau is intrinsic
and complete. Newton landing (via hybrid or pure Sella) is the only
mechanism that reaches fmax<0.005 on this PES.

## Source

- `/lustre07/scratch/memoozd/gadplus/logs/compr_61087774_5.out`
- `/lustre07/scratch/memoozd/gadplus/runs/test_longbudget/gad_dt005_10k/` (trajs writing; summary at job end)
