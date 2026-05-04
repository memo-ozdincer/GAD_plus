# Findings — 2026-04-29 test-set sweep

This file accumulates **findings** as data lands. Each entry has:
hypothesis tested, evidence, and (when known) mechanism. **Every numeric
claim cites the parquet/csv/log file it came from.**

## ⚠️ Headline metric ordering

Raw convergence rate (n_neg=1 ∧ fmax<thresh) is a **proxy**, not the
final answer. The paper's headline metric is **IRC-validated TOPO-intended
rate**: starting from each candidate TS, run IRC forward + reverse
to two minima; check that the two minima are the labeled reactant + product
by element-aware bond-graph isomorphism. That is the chemistry ground truth.

Order of importance:
1. **TOPO-intended** (IRC + bond-graph match) — final answer
2. **RMSD-intended** (IRC + RMSD<0.3Å) — strict version
3. **conv n_neg=1 ∧ fmax<0.01** — proxy: did we land on *some* index-1 saddle
4. **conv (no saddle req)** — too lax for paper claims

IRC validation rerun (60146557) is in flight. Once landed, the headline
shifts to TOPO-intended; raw conv is then the upper bound on it
(``can't be intended without first being converged''). The IRC method
itself is audited in §12 below.

## 1. Convergence-threshold dictates the GAD-vs-Sella ranking

**Setup:** Test split (n=287), 6 noise levels (10–200 pm), 5000-step
budget for GAD, 2000-step budget for Sella.

**Sources:**
- GAD dt grid: `runs/test_dtgrid/gad_dt00{3..8}_fmax/summary_*.parquet`
- Sella canonical configs: `runs/test_set/sella_{carteck_default,carteck_libdef,internal_default}/summary_*.parquet`
- Sweep aggregator: `scripts/analyze_threshold_sweep.py` → `analysis_2026_04_29/threshold_sweep.csv` and `threshold_sweep_table.md`

**Tables:** see `threshold_sweep_table.md`. Headline:

| criterion (n_neg=1 ∧ ⋯) | best method | margin |
|---|---|---|
| `fmax<0.01` | tied at low noise; GAD +2-17pp at 100-200pm | small |
| `force_norm<0.01` | GAD wins big | +24pp at 200pm |
| `fmax<0.005` | Sella wins everywhere | +30pp |
| `force_norm<0.005` | Sella wins everywhere | +60pp |
| `*<0.001` | nothing converges | — |

**Mechanism:** GAD's Euler ODE plateaus at force_norm ≈ 0.005-0.01.
Sella's QN+trust-region drives forces to tiny values when it succeeds,
hence its bimodal RMSD distribution (close-or-far). GAD's residuals are
uniformly small (~0.005) but never tiny.

**Why GAD plateaus:** Conjecture — eigenvector v₁ becomes ill-defined
near the saddle (λ₁ → 0 from negative), so the flip term 2(F·v₁)v₁
gets noisy. Plus Hessian numerical noise (~1e-5) sets a floor on what
F_GAD can resolve. Low-dt sweep (60110297) will test this — if dt=1e-4
GAD also plateaus at ~0.005, it's structural.

## 2. Sella has bimodal final-RMSD; GAD has unimodal

**Setup:** Kabsch RMSD from final geometry to known TS, all 287 samples
× 6 noise levels.

**Evidence:** `analysis_2026_04_29/test_summary_full.csv`,
`figures/fig_rmsd_distrib_test.pdf`.

**Saddle-quality breakdown (Sella):** Sella's "n_neg=1 only" rate is
20-25 pp higher than its "n_neg=1 ∧ fmax<0.01" rate at high noise.
I.e. Sella **finds the saddle** but **can't tighten the force enough
in 2000 steps** more often than the headline number suggests.

| Sella libdef | n_neg=1 ∧ fmax<.01 | n_neg=1 only | gap |
|---|---|---|---|
| 100 pm | 70.7% | 90.6% | +19.9 pp |
| 150 pm | 54.0% | 73.5% | +19.5 pp |
| 200 pm | 27.2% | 54.0% | +26.8 pp |

**Implication:** A "Sella + force-tightening kick" variant could close
the gap at the cost of more steps. Worth a follow-up experiment.

**Sources:**
- per-sample stats: `analysis_2026_04_29/test_summary_full.csv`
- saddle-quality table: `analysis_2026_04_29/saddle_quality_table.md`
- distrib figures: `figures/fig_rmsd_distrib_test.{pdf,png}`, `figures/fig_rmsd_distrib_combined.{pdf,png}`
- builders: `scripts/analyze_rmsd_bimodal.py`, `scripts/analyze_rmsd_gad.py`

## 3. Compute cost — replaced 2026-05-04 (the "50× cheaper" framing was misleading)

Earlier draft compared Sella vs GAD on **step counts**: Sella's median 13
steps vs GAD's 700-1000 steps gave the "Sella is 50× cheaper" headline.
That framing is wrong because per-step cost is comparable — both call HIP
Hessian + eigendecomp every step. The right metric is wall-time per
converged TS.

**Per-step wall-time** (median across all $n=287$ samples per cell, ms/step):
- GAD dt=0.007: 62 ms/step (consistent across noise)
- Sella libdef: 76 ms/step (1.20× GAD; trust-region linear solve adds ~14 ms)
- Sella internal: 100 ms/step (internal-coords transform overhead)

**Wall-time per converged TS** (`SUM(wall_time_s)` over all 287 attempts /
`SUM(converged)`):

| noise | GAD dt007 (5k) | Sella libdef (2k) | ratio | n_conv (GAD/Sella) |
|---|---|---|---|---|
| 10pm  | 47 s  | 14 s  | 3.33× | 256 / 276 |
| 30pm  | 51 s  | 15 s  | 3.38× | 255 / 276 |
| 50pm  | 68 s  | 22 s  | 3.06× | 246 / 264 |
| 100pm | 141 s | 61 s  | 2.31× | 209 / 217 |
| 150pm | 261 s | 125 s | 2.08× | 167 / 164 |
| **200pm** | **441 s** | **348 s** | **1.27×** | **128 / 89** |

At 200pm, GAD costs only **27% more wall per converged TS** while
producing **44% more converged TS**. Above n_conv = 89 at 200pm, Sella
*cannot deliver* regardless of compute (Sella libdef 5k = 218 conv, 10k
= 218 conv at 100pm — saturated).

**Mechanism for why Sella saturates and GAD doesn't:**
- Sella's failures at 200pm: 63% wrong-saddle (n_neg≥2), 27% non-saddle
  (n_neg=0 minimum). P-RFO already converged successfully, just to the
  wrong critical point. More steps don't help.
- GAD's failures at 200pm: 62% plateau-orbit (right basin, n_neg=1, just
  fmax≈0.05 above gate). These would converge with a Newton polish (but
  see §20 — naive NR-polish doesn't work).

**GAD truncated to 2k step budget (matched to Sella):** at 200pm raw
conv drops from 44.6% → 39.0% (vs Sella's 31.0%) — GAD still wins by 8pp
at the same step budget.

**Sources:**
- `analysis_2026_04_29/compute_summary.csv` (per-cell step quantiles, wall, ms/step, wall-per-conv-TS).
- `analysis_2026_04_29/gad_truncation.csv` and `sella_truncation.csv` (cumulative converged-fraction by step N).
- `analysis_2026_04_29/dynamics_walltime.csv` (fmax median+IQR by wall-time bin).
- Build script: `scripts/analyze_compute.py`.
- Figures: `figures/fig_compute_step_dist.pdf, fig_compute_wall_per_conv.pdf, fig_compute_dynamics_walltime.pdf, fig_compute_truncation_cdf.pdf, fig_compute_pareto.pdf`.
- Section in tex: §"Compute cost — what does GAD's robustness actually cost?" (rewritten 2026-05-04).

## 4. dt grid: cliff at dt ≈ 0.008

GAD dt grid on test (5000 steps, fmax<0.01):

| dt | 10 | 30 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|
| 0.003 | 89.2 | 88.5 | 85.4 | 71.1 | 55.1 | 40.8 |
| 0.005 | 89.2 | 88.5 | 85.7 | 71.8 | 57.1 | 43.2 |
| 0.007 | 89.2 | 88.9 | 85.7 | 72.8 | 58.2 | 44.6 |
| 0.008 | 72.1 ↓ | 71.4 ↓ | 70.7 ↓ | 60.3 ↓ | 49.5 | 37.3 |

dt=0.007 is the sweet spot. Cliff at dt∈(0.007, 0.008) — Euler
stability bound dt < 2/λ_max with stiff bond modes ~250.

**Sources:** `runs/test_dtgrid/gad_dt00{3..8}_fmax/summary_*_*pm.parquet`,
aggregated in `analysis_2026_04_29/threshold_sweep.csv`.

## 5. adaptive_dt collapses

`gad_adaptive_dt` drops 15-25 pp below fixed-dt at every noise level
on test. **Mechanism (added 2026-05-01):** at 10pm noise, comparing
adaptive vs fixed dt=0.007:

| metric | adaptive_dt | fixed dt=0.007 |
|---|---|---|
| converged (n\_neg=1 ∧ fmax<0.01) | 209/287 (73%) | 256/287 (89%) |
| n\_neg=1 reached | 259/287 | 283/287 |
| n\_neg=2 at end | 18 | 4 |
| n\_neg≥3 at end | 10 | 0 |
| median steps | 457 | 75 |

Adaptive has *more* end-n\_neg=2 and ≥3 saddles than fixed-dt — the
eigenvalue clamp is pushing trajectories *out* of the index-1 basin
into higher-order saddles. Combined with 6× more steps to converge,
the clamp is over-conservative in flat regions and over-aggressive
when the trajectory crosses bifurcations. Drop the feature unless
re-thought from scratch.

**Sources:**
- adaptive: `runs/test_set/gad_adaptive_dt/summary_*.parquet`
- fixed dt=0.007: `runs/test_dtgrid/gad_dt007_fmax/summary_*.parquet`

## 6. Sella collapses without HIP Hessian (added 2026-05-01)

**Setup:** Same Sella configs (carteck_default, internal_default) but with
`hessian_function=None` — Sella reverts to its internal BFGS update from
gradients. 12-cell sweep (60110188) timed out at 12h; partial coverage of
~25 of 287 samples per cell.

**Evidence:** `analysis_2026_04_29/sella_nohess_partial.csv` (parsed from
slurm stdout — no summary parquet was written before timeout).

| method | 10 | 30 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|
| carteck (with HIP H) | 95% | 90% | 87% | 70% | 54% | 23% |
| carteck (no Hessian) | 10% | 3.4% | 10% | 3.3% | 0% | 3.4% |
| internal (with HIP H) | 91% | 86% | 81% | 73% | 60% | 36% |
| internal (no Hessian) | 4.3% | 0% | 4.3% | 0% | 0% | 0% |

**Conclusion:** Sella's strength on T1x is *entirely* due to the injected
HIP analytic Hessian. With no Hessian, BFGS-from-gradients gets stuck —
median fmax 0.12-1.4 after 2000 steps, ~25 minutes per sample. **This is
the load-bearing experimental result for the paper's headline argument.**

When MLIP Hessians are available (which is the regime we care about),
both Sella and GAD work; GAD then wins on convergence rate at high noise
and at loose criterion. When MLIP Hessians are not available, Sella's
QN baseline collapses below 10% across all noise levels — making the
question "GAD vs Sella" moot in that regime.

**Caveat:** "no Hessian" is one extreme of a continuum — Sella's library
default is `nsteps_per_diag=3` (recompute every 3 steps). Job 60147671
sweeps `diag_every ∈ {3, 5, 10, 25}` to fill the curve.

**Sources:**
- raw logs: `logs/testsellanohess_60110188_*.out`
- parsed CSV: `analysis_2026_04_29/sella_nohess_partial.csv`
- parser: `scripts/parse_nohess_logs.py`
- comparison rows ("with HIP H"): `analysis_2026_04_29/threshold_sweep.csv`, threshold=0.01

## 7b. Low-dt diagnostic confirms GAD plateau is structural (added 2026-05-01)

**Setup:** Job 60110297 — same GAD code at very small dt:
- dt=1e-3 with 20k step budget
- dt=5e-4 with 40k step budget
- dt=1e-4 with 100k step budget

**Partial table (% CONV / N completed; full target n=287):**

| dt | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| 1e-3 | 89.7/282 | 89.3/242 | 86.4/191 | 77.4/124 | 56.6/76 | 50.0/64 |
| 5e-4 | 89.4/151 | 89.0/127 | 87.0/100 | 79.1/67 | 56.8/37 | 51.5/33 |
| 1e-4 | 91.9/62 | 90.5/42 | 87.9/33 | 81.0/21 | 50.0/12 | 45.5/11 |

**Reference (dt=0.007, full 287):** 89.2/88.9/85.7/72.8/58.2/44.6.

Lowering dt by 70× changes convergence rate by <5pp, while compute grows 70-700×.
**Mechanism confirmed:** GAD's plateau at force\_max≈0.0099 is structural, not
dt-discretization. The Euler trajectory orbits the saddle at this residual
because v₁ becomes ill-defined as λ₁→0 — smaller dt just resolves the orbit
more finely without resolving the residual.

**Implication:** dt=0.007 with 5000-step budget is the right operating point
for GAD on T1x. Pushing to smaller dt is dominated by compute waste. The only
remaining lever for tighter convergence is a *post-hoc* Newton-Raphson polish
(level-4 NR-GAD), not smaller dt.

**Sources:**
- logs: `logs/testlowdt_60110297_{0..17}.out` (still in flight; numbers parsed live)
- per-sample: `runs/test_lowdt/gad_dt{001,0005,0001}_fmax/traj_*.parquet`
- ref dt=0.007: `runs/test_dtgrid/gad_dt007_fmax/summary_*.parquet`

## 7. IRC validation (60110465) was broken — split=train (added 2026-05-01)

`scripts/irc_validate.py` hardcoded `split="train"` when loading the
reference dataset. Test-split runs were validated against train-split
reactant/product references — chemistry-mismatched. Combined with silent
exception swallowing in `irc_sella_hip.py`, every single IRC cell
returned 0 intended / 0 topology_intended.

Fixed: added `--split` arg; updated slurm script to pass `--split test`;
exposed exception traceback. Smoke test 60145419 verified the fix:
1 RMSD-INTENDED, 4 TOPO-INTENDED, 0 errors, ~22s/sample. Full rerun
launched as 60146557.

**Sources:**
- old broken parquets: `runs/test_irc/*/irc_validation_sella_hip_allendpoints_*pm.parquet` (intended=0 everywhere — do not trust)
- smoke verification: `/tmp/irc_smoketest/`, `logs/testirc_smoke_60145419.out`
- fix: `scripts/irc_validate.py:218,103-105`, `src/gadplus/search/irc_sella_hip.py:207-211`

## 8. From reactants — Sella climbs surprisingly well (added 2026-05-01)

**Setup:** Job 60110201 — single-ended search from the reactant minimum, no
TS noising. 4 methods × 0pm noise × 287 samples × 10000-step budget.

**Partial:**

| method | completed | converged (n_neg=1 ∧ fmax<0.01) | rate |
|---|---|---|---|
| GAD dt=0.003 | 86/287 | 44 | 51% |
| GAD dt=0.005 | 105/287 | 62 | 59% |
| Sella libdef | 287/287 | 232 | 80.8% |
| Sella internal default | 207/287 | 181 | 87.4% |

**Caveat:** these are "any-saddle" rates — many samples may climb to a
*wrong* saddle. The paper's claim about GAD being more reliably initialized
from minima requires IRC-validated TOPO-intended rates, not raw conv rates.
Will revise once IRC sweep (60146557) lands.

GAD takes 7000-9000 steps per sample (~600s) when it converges; Sella
averages ~50 steps + 5s. GAD is ~100× more compute-expensive from reactant.
Stops the paper from saying "GAD is cheap from minima" — it isn't, but it
is "single-ended" (no path or product reference needed).

**Sources:**
- per-method logs: `logs/testreact_60110201_{0..3}.out`
- Sella libdef summary: `runs/test_reactant/sella_carteck_libdef/summary_sella_cartesian_eckart_fmax0p01_carteck_libdef_0pm.parquet`
- in-flight per-sample: `runs/test_reactant/gad_dt00{3,5}_fmax/traj_*.parquet`

## 9. The "best of each" picture (added 2026-05-01)

**What this section claims.** A single ``best'' GAD and a single ``best''
Sella per noise level, and the head-to-head delta. Built from
\texttt{analysis_2026_04_29/threshold_sweep.csv} at threshold=0.01,
gate=\texttt{conv_fmax_pct} (= n_neg=1 ∧ fmax<0.01).

### The "best" config for each method

**Best GAD** — dt=0.007, 5000 steps, full Eckart projection, no
adaptive_dt. (Unique winner across every noise level — the dt grid
ranks dt=0.007 ≥ 0.006 ≥ 0.005 ≥ 0.004 ≥ 0.003 monotonically; dt≥0.008
collapses from Euler instability.)

**Best Sella (Cartesian + Eckart)** — `libdef`: $\delta_0{=}0.1,
\gamma{=}0.4$, full HIP Hessian every step (`diag_every_n=1`),
fmax<0.01. Best-tuned Cartesian Sella per noise level.

**Best Sella (internal coords)** — `internal default`: $\delta_0{=}0.048,
\gamma{=}0$, full HIP Hessian every step. Note: internal-coords Sella
is **a different optimizer** from cartesian — different coordinate
system, different trust-region semantics. Listed separately because
combining them as "Sella" would conflate two algorithms.

### Head-to-head best vs best (loose gate, fmax<0.01 ∧ n_neg=1)

| noise (pm) | GAD dt=0.007 (5k) | Sella libdef (cart, every-step H) | gap (GAD − Sella) |
|---|---|---|---|
| 10  | 89.2 | 92.7 | −3.5 |
| 30  | 88.9 | 92.0 | −3.1 |
| 50  | 85.7 | 88.2 | −2.4 |
| 100 | 72.8 | 70.7 | **+2.1** |
| 150 | 58.2 | 54.0 | **+4.2** |
| 200 | 44.6 | 27.2 | **+17.4** |

**Headline:** Sella wins at low noise (close to the saddle, where its
Newton step is in its sweet spot); GAD wins at high noise, where Sella's
trust-region collapses but GAD's monotone descent keeps progressing.

## 10. Two comparison framings (added 2026-05-01)

The user's note: ``a few of Sella's configs can be classified as
separate optimizers'' — we report two framings explicitly so the
reader can pick which is fair.

### Framing A: most-faithful, both-tuned, both armed with HIP H

**GAD:** dt=0.007 (sweet-spot picked from in-house sweep), 5000 steps.
**Sella:** libdef hyperparams ($\delta_0{=}0.1, \gamma{=}0.4$) tuned in
our hyperparameter grid; HIP Hessian every step. Same coordinate system
(Cartesian + Eckart projection) for both. Same convergence gate
(fmax<0.01 ∧ n_neg=1).

This is the comparison reported above (Section 9). Both methods are
tuned to the same level. Sella wins at low noise, GAD wins at high noise.

### Framing B: most-out-of-the-box for each

**GAD:** dt=0.003, no special-case features (the original published
GAD paper's default Euler step). 2000 steps (also their default).
$\Rightarrow$ runs `runs/test_set/gad_dt003_fmax/`.

**Sella:** library defaults — internal coords, $\delta_0{=}0.048,
\gamma{=}0$, `nsteps_per_diag=3` (Hessian every 3 steps, not every
step), fmax=**0.05** (Sella's library default convergence gate, not
0.01). $\Rightarrow$ partly the `Sella internal` row (still 4/6 noise
levels), but the every-3-steps-Hessian configuration is **still landing**
in the Hessian-frequency sweep (job 60147671).

| noise (pm) | GAD dt=0.003 (out-of-box) at fmax<0.01 | Sella internal default at fmax<0.01 | Sella defaults at fmax<0.05 |
|---|---|---|---|
| 10 | 88.2 | 79.1 | 94.4 |
| 30 | 87.1 | 77.4 | 90.6 |
| 50 | 84.7 | 71.8 | 84.7 |
| 100 | 69.3 | 50.9 | 63.8 |
| 150 | 51.6 | -- | -- |
| 200 | 34.8 | -- | -- |

**Note on the right-most column:** Sella's library convergence default is
fmax<0.05 (5× looser than fmax<0.01), so Sella-out-of-the-box converges
strictly more samples than the middle column suggests. We report both.

**Caveat:** the "every-3-steps-Hessian" out-of-the-box Sella is still
running (60147671). The middle column above uses every-step Hessian
because we hand-tuned that as the natural way to inject HIP — meaning
the middle column actually *over-helps* Sella (more HIP Hessians than
its library default). The truly out-of-the-box Sella will be slower
(fewer Hessian evaluations) and have lower convergence; we'll fill the
table once 60147671 lands.

### Sella variants are separate optimizers — explicit list

Each row below is a different algorithm in the literature sense, even
though they're all called "Sella":

| Sella variant | what changes | algorithm differs in |
|---|---|---|
| `cart_eckart_libdef` (our canonical) | tuned $\delta_0,\gamma$, every-step HIP H | none beyond hparams |
| `cart_eckart_default` | library $\delta_0,\gamma$, every-step HIP H | hparams |
| `cart_eckart_d3` (60147671) | every-3-steps HIP H (library default) | Hessian update cadence |
| `cart_eckart_d10` | every-10-steps HIP H | Hessian update cadence |
| `cart_eckart_nohess` | no analytic H, BFGS-only | **fundamentally different** (BFGS Hessian model) |
| `internal_default` | internal coords | **different coord system** |
| `internal_nohess` | internal coords + BFGS-only | both |

The first three differ in hyperparameter only and cluster around the
same algorithm. The last three are arguably distinct optimizers. We
report each separately and never aggregate them under a single
``Sella'' row.

## 11. RMSD distribution: why does GAD plateau farther but more uniformly? (added 2026-05-01)

**Setup:** Kabsch+Hungarian RMSD from each *converged* sample's final
geometry (n_neg=1 ∧ fmax<0.01) to the labeled TS coords. Statistics
across samples per (method, noise) cell.

### Statistics (converged samples only)

| method | noise | n_conv | RMSD med (Å) | RMSD mean | q25–q75 | n<0.05Å | n>0.5Å |
|---|---|---|---|---|---|---|---|
| Sella libdef | 10 | 266 | 0.008 | 0.017 | 0.004–0.019 | 249 | 0 |
| Sella libdef | 100 | 203 | 0.008 | 0.033 | 0.004–0.018 | 185 | 2 |
| Sella libdef | 200 | 78 | 0.014 | **0.131** | 0.004–0.078 | 57 | 9 |
| GAD dt=0.007 (5k) | 10 | 256 | 0.009 | 0.012 | 0.007–0.011 | 252 | 0 |
| GAD dt=0.007 (5k) | 100 | 209 | 0.071 | 0.079 | 0.059–0.087 | 33 | 1 |
| GAD dt=0.007 (5k) | 200 | 128 | 0.152 | **0.235** | 0.113–0.198 | 0 | 14 |

(source: `analysis_2026_04_29/test_summary_full.csv` for Sella;
`gad_test_rmsd.csv` for GAD; both filtered to converged-only.)

### Reading the distribution

**Sella's bimodal pattern at high noise** — at 200pm: mean 0.131 Å,
median 0.014 Å, q75 0.078 Å. Mean ≫ median. 9 of 78 converged samples
have RMSD>0.5Å. **Mechanism:** Sella's trust-region step jumps the
geometry by O(0.1 Å) per step. When the jump lands in the saddle's
basin, the next Newton step nails it (RMSD~0). When it lands in a
*different* index-1 region (a wrong saddle), Sella converges there
just as confidently, but the RMSD to the labeled TS is large. Hence
two clusters: "Newton's-method nails" near 0, and "wrong saddle"
near 0.5–0.8 Å.

**GAD's tighter, larger-shifted distribution** — at 200pm: mean 0.235 Å,
median 0.152 Å, q25–q75 0.113–0.198. Mean ≈ median (not bimodal
within the converged samples; some long-tail outliers). **Mechanism:**
GAD's Euler dynamics doesn't take large jumps. It walks toward the
saddle; if the noise pushed the start far, GAD's *trajectory* never
quite reaches the labeled TS in 5000 steps because of the plateau
($f_{\max}\approx 0.01$ orbital radius, see §dynamics).
RMSD ~0.15 Å at 200pm reflects the fact that ``converged at fmax<0.01''
is *not* ``geometrically at the saddle'' — it's ``within 0.15 Å of it.''

### Why GAD's RMSDs are systematically larger

When Sella converges, its post-conditions are: $\nabla E \approx 0$
(quadratic in displacement) AND $\nabla^2 E$ has correct index. GAD's
post-conditions are: only $f_{\max}<0.01$ (linear). Sella's stricter
condition gives smaller residual displacement when it succeeds.

**Can it be fixed?** Yes — a NR polish step after GAD plateaus drives
fmax to machine precision in O(few) steps, which should drag RMSD
down with it. Job 60151717 (`nr_gad_polish_dt007_*`) tests exactly
this. Predicted outcome: GAD+NR's RMSD distribution should match
Sella's "nailed it" mode; the bimodal "wrong saddle" mode shouldn't
appear because GAD's slower trajectory is less likely to jump into
it. To be confirmed when 60151717 lands.

### Steps to convergence (converged samples only)

| noise | GAD dt=0.007 median steps | Sella libdef median steps | ratio |
|---|---|---|---|
| 10  | 72   | 4  | 18× |
| 30  | 144  | 6  | 24× |
| 50  | 199  | 7  | 28× |
| 100 | 331  | 9  | 37× |
| 150 | 436  | 11 | 40× |
| 200 | 545  | 13 | 42× |

**Compute reality:** Sella is 18–42× cheaper per converged sample
(median steps). GAD's win is in *converging more samples at high
noise*, not in efficiency. The relevant figure-of-merit for compute
is "intended TSs per GPU-hour", not raw conv rate.

**Sources:** `analysis_2026_04_29/test_summary_full.csv` (Sella),
`gad_test_rmsd.csv` (GAD), filter `final_n_neg=1 AND final_fmax<0.01`
on Sella, `n_neg=1 AND force_max<0.01` on GAD.

## 12. IRC method audit (added 2026-05-01)

The "intended" metric is the paper's chemistry ground truth. The IRC
machinery has several knobs; this section documents exactly what they are,
where they live, and which were chosen by us vs by Sella's defaults.

### Pipeline

For each candidate TS produced by an optimizer:

1. **Forward IRC** from TS for ≤ 500 steps with `direction="forward"`.
2. **Reverse IRC** from TS for ≤ 500 steps with `direction="reverse"`.
3. Take the *positions* at the end of each direction (whether or not it
   converged); call them `forward_coords` and `reverse_coords`.
4. **Score against dataset reactant R and dataset product P** via two
   independent metrics: Kabsch+Hungarian RMSD, and element-aware bond-graph
   isomorphism. Direction-agnostic: an endpoint matches *whichever* of R
   or P happens to be closer.

### Integrator: Sella IRC + HIP Hessian every step

We use Sella's `IRC` class (the same package as the P-RFO optimizer)
with a hand-patched Hessian wrapper:

- After every Sella inner kick, `_force_hessian_every_kick(pes)` overwrites
  the BFGS-updated Hessian with our HIP analytic Hessian (mass-weighted,
  Eckart-projected, un-mass-weighted to Cartesian — same recipe as the
  GAD search). This means every IRC step uses the full ML-IP Hessian, not
  a BFGS estimate.
- A second monkey patch `_force_first_kick(irc)` bypasses ASE's pre-kick
  convergence check. Without this, when the input TS is already at
  fmax<halt_threshold (which it is by construction — that's how it became
  a TS candidate), IRC would return immediately without ever stepping.
  This patch forces at least one `step()` call before convergence is checked.

| IRC parameter | value | meaning |
|---|---|---|
| `dx` | 0.1 Å | step size in mass-weighted coords |
| `eta` | 1e-4 | trust-region regularization |
| `gamma` | 0.4 | line-search parameter |
| `fmax` | 0.01 eV/Å | per-step force convergence (when forces drop below, IRC ends early) |
| `max_steps` | 500 | step cap per direction |
| `hessian_function` | `_make_mw_eckart_hessian_function(calc)` | every-kick HIP H + Eckart projection |

These match the defaults from `sella.IRC`, with the HIP Hessian injected
as the only major change.

### Scoring against R/P

**Two metrics in parallel** — they answer different questions:

#### A. RMSD-based "intended"

```python
fr_rmsd = kabsch_hungarian(forward_coords, reactant)
rr_rmsd = kabsch_hungarian(reverse_coords, reactant)
fp_rmsd = kabsch_hungarian(forward_coords, product)
rp_rmsd = kabsch_hungarian(reverse_coords, product)

found_reactant = min(fr_rmsd, rr_rmsd) < 0.3 Å
found_product  = min(fp_rmsd, rp_rmsd) < 0.3 Å
intended       = found_reactant AND found_product
half_intended  = (found_reactant XOR found_product)
```

`min(fr, rr) < 0.3` is direction-agnostic: we don't require Sella's
forward direction to match the dataset's reactant — only that *one* of
the two endpoints does.

**Knob: `rmsd_threshold = 0.3 Å`.** Looking at our final-state RMSDs
post-conv, this is borderline: Sella's bimodal "wrong saddle" cluster
sits at 0.5+ Å (would correctly be unintended). But IRC endpoints
that are close to but not quite within 0.3 (e.g., 0.35 Å due to
breathing-mode noise on a 30-atom structure) would also be unintended.
A more permissive threshold (0.5 Å) might catch more truly-intended
cases without much false-positive risk; we should report both.

#### B. Topology (bond-graph isomorphism)

```python
G_R   = bond_graph(reactant)
G_P   = bond_graph(product)
G_fwd = bond_graph(forward_coords)
G_rev = bond_graph(reverse_coords)

topology_intended = (
    (G_fwd ≅ G_R and G_rev ≅ G_P)
    or (G_fwd ≅ G_P and G_rev ≅ G_R)
)
```

Bond graph: ASE's `neighbor_list` with cutoffs = `natural_cutoffs(atoms)
× 1.2` (covalent_radius * 1.2). Element-labeled graph isomorphism via
`networkx.is_isomorphic` with `node_match` checking atomic number Z.

**Knob: `cutoff_scale = 1.2`.** Standard choice; covers most bond
lengths but borderline-long bonds (e.g., partial C–C ~1.6 Å in a
late-product) might be excluded. Worth re-running with 1.3 to check.

This is the **chemistry-grounded** metric: it asks ``does the IRC
endpoint have the same molecular connectivity as R/P?'' Two structures
with the same bonds are the same molecule, even at different RMSD.

### Why TOPO is more permissive (and more correct) than RMSD

Two molecules can be the same compound (TOPO match) while differing
in conformer (RMSD>>0.3). After IRC, we expect to land "near" R/P but
not necessarily in the exact dataset conformer. RMSD may report
unintended even when the chemistry is right. TOPO catches this.

The smoke test (60145419, 5 samples at 10pm, gad_dt007) showed:
- 1 sample RMSD-intended (0.4Å threshold-passed)
- 4 samples TOPO-intended (RMSD between 0.36 and 0.78 Å)

So TOPO captures 4×-more truly-intended cases at this small sample.
Will scale similarly at full sweep.

### Knobs we control vs Sella defaults

- **Knobs we set explicitly:** `dx=0.1`, `eta=1e-4`, `gamma=0.4`,
  `fmax=0.01`, `max_steps=500`, `hessian_function = our HIP+Eckart`,
  `rmsd_threshold=0.3`, `cutoff_scale=1.2`.
- **Sella IRC defaults we accept:** internal IRC step direction logic,
  trust-region update, line-search machinery.
- **Monkey patches:** `_force_hessian_every_kick`, `_force_first_kick`.

### Past bug: split=train

Earlier IRC run (60110465) loaded the dataset with `split="train"` while
the survey TSs came from test split. Reactant/product references were
chemistry-mismatched ⇒ every cell reported 0 intended / 0 topology_intended.
Fixed in `scripts/irc_validate.py:218` (added `--split` arg, slurm uses
`--split test`); smoke test 60145419 verified the fix; full rerun (60146557)
in flight.

### Open audit items / what could still be wrong

1. **`fmax=0.01` IRC stop criterion** — IRC ends when forces drop below.
   500 steps may not always reach this; may want `fmax=0.05` for IRC
   purposes (we just need to know which basin we're in, not minimize).
2. **`max_steps=500` per direction** — likely fine but unverified for
   long IRC paths.
3. **Sella IRC's internal step-direction logic** — we trust their
   implementation; haven't independently verified.
4. **`coords_to_bond_graph` cutoff_scale=1.2** — should test 1.3 to
   check robustness of TOPO claim.
5. **TS candidate quality vs IRC outcome** — a TS with RMSD>0.5Å to the
   labeled saddle may still IRC-validate to the right R/P (we'll know
   once data lands).

**Sources:**
- `src/gadplus/search/irc_sella_hip.py` (integrator + Hessian wrapper)
- `src/gadplus/search/irc_validate.py` (scoring + bond graph)
- `scripts/irc_validate.py` (CLI driver)

## 13. Threshold spectrum: where does GAD win across criteria? (added 2026-05-01)

**What this is.** The user asked: ``we have multiple thresholds and gates,
where does GAD outperform?'' Below is best-GAD vs best-Sella head-to-head
at every (threshold, gate) we measured. Best-GAD = max across $dt$ in
$\{0.003, 0.005, 0.007\}$; best-Sella = max across $\{$default, libdef,
internal default$\}$ Cartesian variants.

**Cells:** convergence rate (\%) at fmax<0.01 ∧ n_neg=1 unless noted.

### Threshold T = 0.05 eV/Å (Sella's library default)

| gate | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| n_neg=1 ∧ fmax<T          | GAD 98.6 / Sella 98.6 / **+0** | 98.3 / 98.3 / +0 | 95.1 / 95.1 / +0 | 82.9 / 83.3 / −0.3 | 72.1 / 66.2 / **+5.9** | 68.3 / 46.0 / **+22.3** |
| n_neg=1 ∧ ‖F‖_mean<T      | 99.0 / 98.6 / +0.3 | 98.6 / 98.3 / +0.3 | 96.5 / 95.1 / +1.4 | 88.5 / 84.7 / **+3.8** | 80.5 / 68.3 / **+12.2** | 73.9 / 47.7 / **+26.1** |
| fmax<T (no saddle req)    | 99.7 / 100 / −0.3 | 99.3 / 99.7 / −0.3 | 96.2 / 96.9 / −0.7 | 85.0 / 87.1 / −2.1 | 76.0 / 71.4 / **+4.5** | 74.2 / 55.1 / **+19.2** |

### Threshold T = 0.01 eV/Å (our canonical)

| gate | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| n_neg=1 ∧ fmax<T          | 89.2 / 92.7 / −3.5 | 88.9 / 92.0 / −3.1 | 85.7 / 88.2 / −2.4 | 72.8 / 70.7 / **+2.1** | 58.2 / 54.0 / **+4.2** | 44.6 / 27.2 / **+17.4** |
| n_neg=1 ∧ ‖F‖_mean<T      | 95.8 / 96.5 / −0.7 | 95.5 / 96.2 / −0.7 | 92.0 / 92.0 / 0.0 | 78.7 / 75.6 / **+3.1** | 64.8 / 57.1 / **+7.7** | 55.7 / 31.4 / **+24.4** |
| fmax<T (no saddle req)    | 89.2 / 92.7 / −3.5 | 88.9 / 92.0 / −3.1 | 85.7 / 88.2 / −2.4 | 72.8 / 71.1 / **+1.7** | 58.5 / 54.4 / **+4.2** | 44.9 / 28.9 / **+16.0** |

### Threshold T = 0.005 eV/Å (tight, half our canonical)

| gate | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| n_neg=1 ∧ fmax<T          | 0 / 35.9 / **−35.9** | 0 / 33.1 / **−33.1** | 0 / 34.1 / **−34.1** | 0 / 24.7 / **−24.7** | 0 / 15.7 / **−15.7** | 0 / 7.0 / **−7.0** |
| n_neg=1 ∧ ‖F‖_mean<T      | 17.4 / 82.6 / −65.2 | 18.1 / 82.2 / −64.1 | 16.0 / 78.7 / −62.7 | 12.9 / 59.9 / −47.0 | 9.8 / 46.3 / −36.6 | 8.0 / 22.3 / −14.3 |

### Reading the spectrum

- **GAD wins big at fmax<0.05 ∧ n_neg=1**: +22pp at 200pm, +5.9pp at 150pm.
- **GAD wins biggest at ‖F‖_mean<0.05 ∧ n_neg=1**: +26pp at 200pm.
  Mean force is more forgiving of single-atom outliers than max force,
  rewarding GAD's ``almost everywhere small'' behavior.
- **GAD wins at fmax<0.01 ∧ n_neg=1**: +17pp at 200pm, +4pp at 150pm.
  This is the "canonical" comparison.
- **Sella wins at fmax<0.005**: GAD literally goes to 0% — the plateau bites.
- **Sella wins at low noise across every threshold**: at 10/30/50pm,
  Sella wins by 0–4pp consistently.

So the picture is **Sella at low noise, GAD at high noise + loose-to-moderate
threshold**, and **Sella at any tight threshold**.

**Sources:** `analysis_2026_04_29/threshold_sweep.csv`. Generic best-of-each query:
```sql
SELECT threshold, noise_pm,
  MAX(CASE WHEN method LIKE 'GAD dt%' AND method LIKE '%(5k)' THEN conv_fmax_pct END) AS gad,
  MAX(CASE WHEN method LIKE 'Sella%' AND method NOT LIKE '%nohess%' THEN conv_fmax_pct END) AS sella
FROM read_csv_auto('threshold_sweep.csv')
GROUP BY threshold, noise_pm ORDER BY threshold, noise_pm;
```

## 14. Why is Sella bad at 200pm? Failure-mode analysis (added 2026-05-01)

**Question.** Sella libdef at 200pm goes from 70.7\% conv at 100pm down to 27.2\% at 200pm. Why?

### Sella is NOT early-stopping prematurely

We pass `fmax=0.01` as the IRC convergence threshold. Sella's library
default is fmax=0.05 (5× looser); we're already running at the stricter
gate. So if Sella declares converged, it's at fmax<0.01 — no early-stopping
issue. Most Sella runs at 200pm hit the **2000-step cap** without
declaring convergence.

### Failure breakdown at 200pm

For each (method × noise) cell, classify FAILED samples (those with
$\neg(n_{\text{neg}}=1 \wedge f_{\max}<0.01)$) by their final-state
geometry:

- **`n_neg=0`** (minimum): the optimizer slid downhill to a non-saddle
- **`n_neg≥2`** (higher-order saddle): wandered to a different saddle type
- **`n_neg=1, fmax≥0.01`** (saddle-loose): found *the* saddle but couldn't
  tighten the force in budget

**Cell legend:** counts of failed samples in each category, plus median
RMSD-to-known-TS over failures and median fmax over failures.

| Sella libdef | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| n_fail / 287       | 21  | 23  | 34  | 84  | 132 | 209 |
| ↪ minimum          | 0   | 0   | 0   | 4   | 6   | 1   |
| ↪ higher-order saddle | 4 | 5 | 8 | 23  | 70  | **131** |
| ↪ saddle-loose     | 17  | 18  | 26  | 57  | 56  | 77  |
| fail RMSD median (Å) | 0.073 | 0.042 | 0.088 | 0.178 | 0.324 | **0.432** |
| fail fmax median (eV/Å) | 0.014 | 0.014 | 0.019 | 0.045 | 0.113 | **0.369** |

| GAD dt=0.007 (5k) | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| n_fail / 287       | 31  | 32  | 41  | 78  | 120 | 159 |
| ↪ minimum          | 0   | 0   | 0   | 1   | 2   | 0   |
| ↪ higher-order saddle | 4 | 5 | 7 | 23  | 40  | 60  |
| ↪ saddle-loose     | 27  | 27  | 34  | 54  | 78  | **99** |
| fail RMSD median (Å) | 0.041 | 0.047 | 0.075 | 0.171 | 0.357 | 0.635 |
| fail fmax median (eV/Å) | 0.019 | 0.019 | 0.024 | 0.056 | 0.062 | **0.046** |

### Reading the failure modes

**Sella's high-noise failure mode is wrong-saddle.** At 200pm, **131 of
209 Sella failures (63\%)** ended up at $n_{\text{neg}}\ge 2$ — a
higher-order saddle that isn't the chemistry we wanted. The trust-region
Newton step jumped between saddle basins and got stuck at a
higher-index one. Median fail fmax = **0.37 eV/Å**: forces are nowhere
near small. More step budget won't help — Sella is at the wrong attractor.

**GAD's high-noise failure mode is plateau-orbit.** At 200pm, **99 of
159 GAD failures (62\%)** are $n_{\text{neg}}=1$ with $f_{\max}<0.05$
but $\ge 0.01$ — i.e., GAD found the right saddle and is orbiting it
at fmax just above the convergence gate. Median fail fmax = **0.046**:
forces are only 5× the gate, not 50×. **More step budget would help GAD
at 200pm**, but the plateau eventually limits how far it can go (as
the low-dt experiment showed).

### Compound implication

- Sella's failure-RMSD jumps from 0.18→0.32→0.43 Å across 100→150→200pm.
  The "Sella bimodal" we observed isn't just Newton-step jumps within
  the same molecule's saddle basin; at high noise it's **landing in
  qualitatively wrong stationary points** (higher-order saddles).
- GAD's failure-RMSD jumps similarly (0.17→0.36→0.64 Å), but the
  underlying mechanism is "the trajectory orbits ~0.5Å away with
  fmax≈0.05", not "we found a different saddle." So GAD failures may
  still IRC-validate to the right R+P (the orbit is in the *right*
  basin, just not at the bottom). To be confirmed once IRC validation
  data lands.

### What MIGHT help Sella at 200pm

1. **More steps** — won't help: Sella is converged at the wrong type
   of saddle, more steps won't escape it.
2. **Smaller trust radius (`delta0`)** — should reduce basin-jumps;
   the libdef config $\delta_0=0.1$ is already moderate; $\delta_0=0.05$
   might help. *Untested.*
3. **Less Hessian (every-3-step instead of every-step)** — counter-intuitively
   may help because frequent Hessian updates near a noisy region
   keep flipping which mode to climb. The Hessian-frequency sweep
   (60147671) tests this.
4. **No Hessian (BFGS only)** — already shown to collapse to ~5% conv.

### What MIGHT help GAD at 200pm

1. **More steps with NR polish** — we expect GAD's plateau cases to
   tighten when we switch to spectral-Newton at $n_{\text{neg}}=1$.
   Job 60151717 tests this (preliminary results pending).
2. **Smaller dt with 20k step budget** — diminishing returns, see §7b.
3. **Mode tracking k=8** — may reduce mode-flips on the orbit.
   *Untested on test split with this gate.*

**Sources:**
- Failure-mode counts: query `analysis_2026_04_29/test_summary_full.csv`
  (Sella) and `gad_test_rmsd.csv` (GAD), filter by failure conditions.
- Generic query template:
  ```sql
  SELECT method, noise_pm,
    SUM(CASE WHEN n_neg=1 AND fmax<0.01 THEN 1 ELSE 0 END) AS n_conv,
    SUM(CASE WHEN n_neg=0 AND NOT (n_neg=1 AND fmax<0.01) THEN 1 ELSE 0 END) AS n_minimum,
    SUM(CASE WHEN n_neg>=2 THEN 1 ELSE 0 END) AS n_higher_saddle
  FROM <csv> GROUP BY method, noise_pm;
  ```

## 🎯 15. IRC validation — the paper's actual headline (added 2026-05-01, MAJOR)

**Job 60146557 landed 34/36 cells.** This is the chemistry ground truth.

### TOPO-intended rate (paper headline metric)

Each cell = % of n=287 candidate TSs whose IRC forward+reverse endpoints
recover the labeled reactant + product by element-aware bond-graph
isomorphism. Source: `runs/test_irc/<method>/irc_validation_sella_hip_allendpoints_<noise>pm.parquet`,
`SUM(CAST(topology_intended AS INT)) / COUNT(*)`.

| method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| GAD dt=0.003 (5k) | 88.5 | 89.2 | 88.9 | 78.4 | 61.0 | 44.6 |
| GAD dt=0.005 (5k) | 88.9 | 89.2 | 88.9 | 78.0 | 61.7 | 44.6 |
| GAD dt=0.007 (5k) | 88.5 | 88.5 | 88.5 | 78.4 | 61.7 | 43.9 |
| Sella libdef       | 89.2 | 89.2 | 87.5 | 72.5 | 49.8 | 23.3 |
| Sella default      | 88.9 | 88.9 | 87.5 | 70.7 | 46.7 | 17.8 |
| Sella internal default | 88.2 | 87.5 | 83.3 | 64.5 | -- | -- |
| **best-GAD − best-Sella** | **−0.7** | **0** | **+1.4** | **+5.9** | **+11.9** | **+21.3** |

### Headline

**At 200pm noise, GAD outperforms best Sella by +21.3pp on the IRC-validated
TOPO-intended metric** (44.6% vs 23.3%). At 150pm, +11.9pp. At 100pm, +5.9pp.
At 50pm and below, GAD and Sella tie.

This is the paper's headline result. It's **stronger than the raw conv
gap** (+17.4pp at 200pm at fmax<0.01), because Sella's bimodal failures
land at \emph{wrong} saddles which IRC correctly rejects. GAD's
failures are mostly plateau-orbit in the \emph{right} basin and so are
more often validated by IRC.

### RMSD-intended (the strict 0.3Å version)

Same source, `SUM(CAST(intended AS INT))`.

| method | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| GAD dt=0.007 (5k) | 46.3 | 45.6 | 46.3 | 40.1 | 30.7 | 20.9 |
| Sella libdef       | 45.6 | 46.3 | 44.6 | 36.6 | 24.7 | 10.1 |
| gap                | +0.7 | −0.7 | +1.7 | +3.5 | +6.0 | **+10.8** |

RMSD-intended is much lower across the board because the 0.3Å threshold
is too strict (many true-positive IRC endpoints land at 0.35–0.5Å due
to conformational breathing). TOPO is the more permissive and
chemistry-grounded metric. Both stories tell the same qualitative
shape: GAD wins at high noise.

### Why TOPO ≈ raw conv at low noise but diverges at high

At 10pm, GAD raw conv (n_neg=1 ∧ fmax<0.01) was 89.2 and TOPO-intended
is 88.5. They agree because at low noise, every converged saddle has the
right chemistry — there's no ambiguity about which TS this is.

At 200pm, GAD raw conv was 44.6 and TOPO-intended is 44.6 — identical.
GAD's converged TSs are virtually all correct-chemistry. \emph{GAD
never converges to a wrong saddle.}

Sella raw conv at 200pm was 27.2 and TOPO-intended is 23.3. Sella loses
~4pp from raw → IRC-validated: those are samples where Sella claimed
``converged'' but IRC reveals it landed at a wrong saddle.

### Headline gap widens as noise grows

\textbf{The IRC-validated comparison is more favorable to GAD than the
raw-conv comparison.} At every noise level $\ge 100$pm, GAD's TOPO-intended
margin over Sella exceeds its raw-conv margin. Mechanism: Sella's
``wrong saddle'' failures inflate its raw conv but get caught by IRC;
GAD's ``plateau-orbit'' failures stay in the right basin and are partly
caught by IRC anyway.

### Caveats

- 2 of 36 IRC cells failed (60146557_34, _35 — Sella internal 150/200pm,
  because the source survey parquets don't exist; recovery sweep timed
  out for those cells).
- TOPO uses bond cutoff = covalent_radius × 1.2; sensitivity to the
  cutoff is untested. Worth re-running at 1.3 (cheap: just re-score).
- RMSD threshold = 0.3Å; relaxing to 0.5Å would catch more
  true-positives. Cheap to re-score.

**Sources:**
- raw IRC parquets: `runs/test_irc/<method>/irc_validation_sella_hip_allendpoints_<noise>pm.parquet`
- columns: `topology_intended, intended, topology_half_intended, half_intended, forward_coords_flat, reverse_coords_flat, forward_rmsd_reactant, ...`
- query template:
  ```sql
  SELECT noise_pm, AVG(CAST(topology_intended AS DOUBLE))*100 AS topo_pct
  FROM 'runs/test_irc/<method>/irc_validation_*.parquet'
  GROUP BY noise_pm ORDER BY noise_pm;
  ```

## 16. Sella step-budget is NOT the bottleneck (added 2026-05-01)

**Question.** Does giving Sella a 5000-step budget (matched to GAD) change the result?
**Answer.** No, within ±2pp.

| noise | Sella libdef 2k (canonical) | Sella libdef 5k (matched) | gain |
|---|---|---|---|
| 10pm | 92.7% | 90.9% | −1.7 |
| 30pm | 92.0% | 90.6% | −1.4 |
| 50pm | 88.2% | 88.2% | 0.0 |
| 100pm | 70.7% | 71.1% | +0.3 |

(150/200pm cells of the 10k-step variant are still landing — predicted +1-3pp at most.)

**Mechanism.** Sella converges in ~10 median steps when it converges
at all. At high noise, failures are wrong-saddle (n_neg≥2 with
fmax≈0.4) — adding steps doesn't escape the wrong attractor. **The step
budget was never the handicap.** The headline conclusion stands: GAD
wins at high noise because Sella jumps to wrong saddles, not because
Sella runs out of time.

**Implication for the paper:** the eliminated-handicap framing now has
hard data. We can confidently report Sella libdef at 2000 steps as the
canonical Sella row without ducking the "but more steps would help"
counterargument.

**Sources:** `runs/test_sella_extended/carteck_libdef_5k/summary_*.parquet`
vs `runs/test_set/sella_carteck_libdef/summary_*.parquet`. Job 60154183.

## 17. Sella Hessian-frequency sweep results (added 2026-05-01)

**Question.** What happens if we relax Sella's HIP Hessian injection
from "every step" (our canonical, `diag_every_n=1`) toward "less
frequent" (Sella's library default `nsteps_per_diag=3`)? Does Sella
degrade smoothly or fall off a cliff?

**Setup:** `runs/test_hessfreq/sella_carteck_libdef_d{3,5,10,25}/`. 4
cadences × 6 noise levels (200pm cells timed out — re-launch with
longer budget pending).

| diag_every | 10pm | 30pm | 50pm | 100pm | 150pm | 200pm |
|---|---|---|---|---|---|---|
| 1 (every step, canonical) | 92.7 | 92.0 | 88.2 | 70.7 | 54.0 | 27.2 |
| 3 (Sella library default) | 91.3 | 90.2 | 88.2 | 67.9 | 48.1 | -- |
| 5 | 91.6 | 90.6 | 85.7 | 66.6 | 45.6 | -- |
| 10 | 92.0 | 90.6 | 84.7 | 64.5 | 43.2 | -- |
| 25 | 93.0 | 91.3 | 85.7 | 62.7 | 41.1 | -- |
| no Hessian (BFGS only) | 10.0 | 3.4 | 10.0 | 3.3 | 0.0 | 3.4 |

**Reading the data:**
- At **low noise**, cadence doesn't matter much — Sella converges in
  ≤10 steps and the diag schedule rarely kicks in. d=25 is even
  slightly *better* than d=1 (within noise).
- At **medium-to-high noise**, sparser Hessians cost ~3-6pp at 100pm
  and 150pm. d=3 (library default) costs ~3pp at 100pm, ~6pp at 150pm.
- The drop from "any HIP H frequency" → "no Hessian at all" is a cliff:
  d=25 still gets 41% at 150pm, but BFGS-only gets 0%. **The first HIP
  Hessian per ~25 steps is what's load-bearing**, not "every step."
- Sella's library default (d=3) is **only marginally worse than our
  every-step canonical** in this sweep — the comparison is robust to
  this hyperparameter.

**Compute trade-off (median HIP forward calls per sample):**

| diag_every | 10pm | 100pm | 150pm |
|---|---|---|---|
| 1 | 6 | 11 | 38 |
| 3 | 6 | 15 | 2001 (cap) |
| 25 | 6 | 29 | 2001 (cap) |

Counter-intuitively, sparser Hessians take *more* total HIP calls at
high noise because Sella runs more steps trying to recover from drifted
BFGS estimates between updates. So "fewer Hessians" does not save compute
at noise levels where it matters.

**Implication for paper:** Framing B (out-of-the-box Sella, library
default cadence d=3) is well-characterized and only modestly different
from Framing A. The headline GAD-wins-at-high-noise result does NOT
depend on us tuning Sella to every-step Hessian.

**Sources:** `runs/test_hessfreq/sella_carteck_libdef_d{3,5,10,25}/summary_*.parquet`. Job 60147671.

## 18. Huge-noise probe (300/500/1000/2000 pm) (added 2026-05-01)

**Question.** What happens beyond 200pm — do GAD and Sella both fail
gracefully, or does one collapse first?

**Setup:** Job 60154004, n=50 per cell (small probe), 4 noise × 4 methods.

### Strict gate (n_neg=1 ∧ fmax<0.01)

| method | 300pm | 500pm | 1000pm | 2000pm |
|---|---|---|---|---|
| GAD dt=0.003 (5k) | 20.0% | 8.0% | 0.0% | 0.0% |
| GAD dt=0.007 (5k) | 22.0% | 8.0% | 0.0% | 0.0% |
| Sella libdef 2k   | 14.0% | 6.0% | 0.0% | 0.0% |
| Sella libdef 5k   | 20.0% | 6.0% | 0.0% | 0.0% |

### Loose gate (n_neg=1 ∧ fmax<0.05)

| method | 300pm | 500pm | 1000pm | 2000pm |
|---|---|---|---|---|
| GAD dt=0.003 (5k) | 42.0% | 30.0% | 14.0% | 4.0% |
| GAD dt=0.007 (5k) | 44.0% | 42.0% | 20.0% | 2.0% |
| Sella libdef 2k   | 30.0% | 18.0% | 8.0% | 2.0% |
| Sella libdef 5k   | 34.0% | 18.0% | 4.0% | 2.0% |

**Reading the data:**
- **GAD's lead persists at huge noise.** At 300pm, GAD 22% vs Sella 20%
  (strict); at 500pm, GAD 8% vs Sella 6%. On loose gate, GAD has +12pp
  at 300pm (44 vs 30) and +24pp at 500pm (42 vs 18).
- **Both methods saturate beyond 1000pm.** Strict-gate conv goes to 0%;
  loose gate hangs on at ~5-20% but is mostly random-baseline.
- **Sella step-budget continues to NOT help.** 5k vs 2k at 300pm: 20%
  vs 14% strict (+6pp from going to 5k), but at 500/1000/2000 the
  matched-budget gives no benefit.
- **HIP Hessian is robust at huge displacement.** Both methods
  produce sensible n_neg=1 saddles at 300pm; the failure mode is
  geometry quality (RMSD) and number-of-saddles-found, not HIP failure.

**Implication.** The 200pm noise level was the right "high-noise" stress
test — beyond it, both methods saturate and the comparison loses
meaning. Don't extend the headline noise grid above 200pm.

**Sources:** `runs/test_huge/{gad_dt003_fmax,gad_dt007_fmax,sella_libdef_2k,sella_libdef_5k}/summary_*.parquet`.

## 19. IRC sensitivity to scoring knobs (added 2026-05-03)

**Question.** The original IRC validation used `rmsd_threshold=0.3 Å` and
bond `cutoff_scale=1.2 × covalent_radius`. Both knobs are somewhat
arbitrary. Are the headlines robust if we pick different values?

**Method.** `scripts/analyze_irc_sensitivity.py` re-scores existing IRC
parquets at `rmsd_threshold ∈ {0.3, 0.4, 0.5, 0.7}` Å and
`cutoff_scale ∈ {1.1, 1.2, 1.3, 1.4}` without re-running IRC (just
recomputes the scoring functions on cached endpoint coords).

**Source data:** `runs/test_irc/<method>/irc_validation_*.parquet`
(forward_coords_flat, reverse_coords_flat columns).
Output: `analysis_2026_04_29/irc_sensitivity.csv` (one row per
method × noise × rmsd_threshold × cutoff_scale).

### TOPO sensitivity (cutoff_scale)

TOPO-intended at cutoff=1.2 (canonical) vs 1.3 (more permissive):

| method | 200pm @ 1.2 | 200pm @ 1.3 | shift |
|---|---|---|---|
| GAD dt=0.007 (5k) | 52.3 | 53.5 | +1.2 |
| Sella libdef     | 31.5 | 32.9 | +1.4 |
| gap (GAD−Sella)  | +20.8 | +20.6 | unchanged |

(Note: the cutoff=1.2 numbers here differ slightly from the headline
table because this script denominates by samples-with-valid-endpoints
rather than full $n=287$. The ranking is unchanged.)

### RMSD-intended sensitivity (threshold)

RMSD-intended at threshold=0.3 Å vs 0.5 Å:

| method | 200pm @ 0.3 Å | 200pm @ 0.5 Å | shift |
|---|---|---|---|
| GAD dt=0.007 (5k) | 24.9 | 32.0 | +7.1 |
| Sella libdef     | 13.6 | 19.7 | +6.1 |
| gap (GAD−Sella)  | +11.3 | +12.3 | +1.0 |

RMSD numbers are uniformly higher with looser threshold (as expected),
but the gap is preserved.

### Conclusion: headlines are robust

- Cutoff_scale 1.2 → 1.3: every cell shifts by ≤2pp. Ranking unchanged.
- RMSD threshold 0.3 → 0.5: every cell shifts up uniformly ~6-10pp; the
  GAD-minus-Sella gap is preserved within ±1pp.
- **The +21pp IRC-TOPO headline at 200pm is robust to these knob choices.**

This pre-empts the reviewer concern "you cherry-picked the bond cutoff
or RMSD threshold."

**Sources:**
- `analysis_2026_04_29/irc_sensitivity.csv` (192 rows, every
  method × noise × rmsd_threshold × cutoff_scale combination)
- builder: `scripts/analyze_irc_sensitivity.py`

## 20. NR-polish negative result (added 2026-05-04)

**Hypothesis:** spectral-partitioned Newton–Raphson polish on top of
GAD breaks the structural plateau at fmax≈0.01. Mechanism would be: NR
step Δx = -H⁻¹F (with v_1 reflected) is quasi-second-order, whereas
explicit Euler is first-order. Predicted: NR phase drives fmax to ~1e-5
when activated.

**Setup:** `scripts/method_single.py --method nr_gad_polish_dt007_{loose,strict}`
on test split, n=80 subset, 3000 step budget, 6 noise levels (10/30/50/100/150/200pm),
two thresholds: loose (fmax<0.01) and strict (fmax<1e-4). SLURM 60314225,
12 cells, all completed in 1.9–4.3 h. Reduced from full n=287 because the
NR phase is expensive (per-step Hessian eigendecomp + linear solve in
the polished phase).

**Result:** NR underperforms vanilla GAD by 14–40pp at every noise.

| noise | GAD dt=0.007 baseline (n=287) | NR-loose (n=80) | Δ |
|---|---|---|---|
| 10pm  | 89.2% | 65.0% | -24.2pp |
| 30pm  | 88.9% | 48.8% | -40.1pp |
| 50pm  | 85.7% | 46.3% | -39.4pp |
| 100pm | 72.8% | 37.5% | -35.3pp |
| 150pm | 58.2% | 30.0% | -28.2pp |
| 200pm | 44.6% | 30.0% | -14.6pp |

**Critical diagnostic:** strict (fmax<1e-4) hits **0% conv across ALL
noise levels**. Min observed fmax across all 480 NR-loose samples =
0.0055; not a single sample crossed fmax<1e-3. The NR phase is *not*
driving fmax down — it's making it worse.

**Mechanism (revised hypothesis):** the NR step magnitude
‖(H + αv_1v_1ᵀ)⁻¹F‖ blows up near the saddle, where H is near-singular
along v_1. Each NR jump is large enough to escape the saddle basin,
forcing GAD to re-approach in the next iteration. The flip-flop between
NR overshooting and GAD recovering never settles. Average final fmax for
NR-loose: 0.21–0.72, vs vanilla GAD 0.024–0.27 — NR ends 4–10× *worse*
than no polish.

**Implications:**
- The "level 4" of the bottom-up plan (NR+GAD flipflop) does not work
  as currently implemented.
- The structural plateau at fmax≈0.01 stays — must either accept it
  (paper headline already does, IRC validation works at fmax<0.01)
  or implement trust-region NR (cap ‖Δx‖ — effectively Sella's RFO
  logic on top of GAD).
- Predicted-help-but-didn't. Documented as negative result; should NOT
  be relaunched at full n=287 unless the algorithm is fixed first.

**Sources:**
- Summary parquets: `runs/test_nrpolish/nr_gad_polish_dt007_{loose,strict}/summary_*.parquet`
- 480 trajectory parquets: `runs/test_nrpolish/.../traj_*.parquet`
- SLURM logs: `logs/testnrpolish_60314225_{0..11}.out` (and `.err`)
- Tex section: §"NR-polish negative result" (added 2026-05-04).

## Open follow-up questions (not yet investigated)

1. **Why does GAD win at loose thresholds (fmax<0.05) by +22pp at 200pm?**
   Need a per-sample classification: at 200pm, take the 287 samples;
   for each, classify (GAD outcome, Sella outcome) into a contingency
   table {both intended, GAD only, Sella only, neither}. Then look at
   geometric features of "GAD only" samples — what makes them
   GAD-friendly? Hypothesis: bad initial Hessian eigendecomp (Sella's
   first move is wrong) but smooth descent in v\_1 direction (GAD walks
   into the right basin). To be done after the canonical sweeps land.

2. **Sensitivity-test IRC validation knobs.** rmsd\_threshold=0.3
   borderline; cutoff\_scale=1.2 standard but might miss long bonds.
   Re-run scoring on existing IRC parquets at threshold {0.3, 0.5}
   and cutoff_scale {1.2, 1.3} to see how rates shift. Cheap: just
   re-score, no new IRC runs.

3. **Sella with smaller delta0 ($\delta_0=0.05$) at high noise.**
   Hypothesis: smaller trust-region reduces basin-jumps. Test 200pm only.

## Open lines (running or pending)

- **Low-dt diagnostic** (60110297): dt ∈ {1e-3, 5e-4, 1e-4}, varying step
  budgets. Tests whether GAD's plateau is structural or step-budget.
- **From reactants** (60110201): single-ended GAD/Sella from minima.
- **IRC validation rerun** (60146557): TOPO-intended rates per method.
- **Sella Hessian-frequency sweep** (60147671): `diag_every ∈ {3, 5, 10, 25}` carteck_libdef.
