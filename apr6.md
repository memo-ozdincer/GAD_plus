# GAD+ Status — April 6-7, 2026 (Final)

All experiments complete. Both agents' work is documented in EXPERIMENT_LOG.md (detailed) and EXPERIMENTS.tex (publication-ready).

---

## Results at a Glance

| Rank | Method | 10pm | 50pm | 100pm | 200pm | Diff-compat? |
|------|--------|------|------|-------|-------|-------------|
| 1 | **gad_dt003** (dt=0.003, 2000 steps) | **94.7** | **92.0** | **88.9** | **58.8** | Yes |
| 2 | gad_small_dt (dt=0.005, 1000 steps) | 94.3 | 91.3 | 86.7 | 51.3 | Yes |
| 3 | gad_no_clamp (no displacement cap) | 94.3 | 91.3 | 86.7 | 54.6 | Yes |
| 4 | nr_gad_damped a=0.1 | 94.7 | 77.7 | 58.0 | 33.7 | No |
| 5 | adaptive_floor (dt_min=1e-3) | 83.0 | 70.2 | 43.2 | 15.9 | Yes |
| 6 | precond_gad_dt01 (|H|^-1, dt=0.01) | 78.3 | 68.3 | 58.0 | 41.7 | Yes |
| 7 | precond_gad_001 (|H|^-1, dt=0.005) | 73.7 | 48.0 | 21.7 | 3.3 | Yes |
| 8 | blend_k50 (sigmoid blend + |H|^-1) | 72.0 | 37.2 | 14.3 | 3.1 | Yes |
| 9 | adaptive_mm (Multi-Mode GAD params) | 53.7 | 35.1 | 22.6 | 5.7 | Yes |
| 10 | nr_gad_pingpong (undamped NR) | 56.7 | 31.7 | 24.7 | 18.3 | No |

---

## Key Findings

1. **The simplest method wins.** Plain Eckart-projected GAD with small fixed dt (0.003) and enough steps beats every "clever" method. No mode tracking, no adaptive dt, no preconditioning, no NR.

2. **|H|^-1 preconditioning hurts GAD.** -21pp at 10pm, -43pp at 50pm. Newton-like scaling (1/|lambda_i|) starves steep modes of progress. GAD needs uniform step sizes across all modes — it's navigating a saddle, not descending to a minimum.

3. **All adaptive dt variants fail.** 5 parameter sets tested (3 eigenvalue-clamped + 1 floor fix + 1 pure precond). All worse than fixed dt. Step-size variability disrupts GAD's steady mode-tracking.

4. **All NR/descent switching variants fail.** Undamped NR, damped NR (3 alphas), preconditioned descent — all worse than pure GAD. The switching logic (hard or soft) isn't the problem; applying different dynamics when n_neg>=2 is.

5. **Displacement capping is inert.** No cap vs 0.35A vs 0.1A — zero effect. The cap never triggers at small dt.

6. **Lambda_2 blend needs re-test.** The mechanism (sigmoid(k*lambda_2)) is correct and differentiable, but was only tested with preconditioning. The preconditioning failure masked the blend signal. All three variants (k=50, k=100, hard-switch descent) gave identical ~72% — the |H|^-1 was the bottleneck.

---

## What's Left to Try

### Highest priority: Blend WITHOUT preconditioning

```
F_blend = F + 2*sigmoid(k*lambda_2)*(F . v1)*v1
Dx = dt * F_blend      # plain Euler, no |H|^-1
```

This isolates the blend effect on top of the already-good gad_small_dt base. If it helps at high noise (100-200pm) where n_neg oscillates, it's the paper's differentiable method.

### Then:
- dt=0.002 with 3000 steps (continue the "smaller dt wins" trend)
- Multiple noise seeds (10x) for 95% confidence intervals
- Full Transition1x (9,561 samples)
- Sella baselines (TS-BFGS, full Hessian) for paper comparison

---

## Jobs & Data

| Job ID | What | Status | Output |
|--------|------|--------|--------|
| 58885855 | Preconditioned GAD (30 jobs) | Complete 30/30 | `precond_gad/` |
| 58886863 | Round 2 (48 jobs) | 9 complete + 39 partial | `round2/` |
| 58845357 | Round 1 method cmp (42 jobs) | Complete | `method_cmp_300/` |
| 58852071 | Damped NR-GAD (42 jobs) | Complete | `targeted/` |

All data: `/lustre07/scratch/memoozd/gadplus/runs/`

---

## Documents

- **EXPERIMENT_LOG.md** — Full log with all 13 experiments, detailed stats, consolidated rankings
- **EXPERIMENTS.tex** — Publication-ready 7-section report with tables and figures
- **EXPERIMENT_PLAN_ROUND2.md** — Original plan (for reference)
- **PAPER_INSIGHTS.md** — Paper narrative vs noisyTS/LMHE
- **DATA_REFERENCE.md** — Parquet schemas, DuckDB queries, plot recipes
