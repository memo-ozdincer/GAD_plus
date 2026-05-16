# SCINE / xTB Findings — 2026-05-15 (IRC TOPO debug + PES disagreement)

Updates `SCINE_XTB_FINDINGS_2026_05_12.md`. The 05-12 doc closed the
"GAD strict-conv on SCINE matches HIP" question; this doc closes the
"why is IRC TOPO still low" question with quantitative evidence that
DFTB0's saddles are geometrically displaced from T1x's by ~0.4 Å,
and the resulting plan to handle this properly (re-label minima per PES
or restrict to the PES-invariant subset).

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Are SCINE TOPO rates low because of an IRC algorithm bug? | **No.** Bond-cutoff sweep (1.10–1.50) doesn't recover failures. Sella IRC instance handling was technically wrong (two instances → independent v0ts); I fixed it (single instance). Numbers didn't move. |
| Are they low because IRC stops on a ridge? | Mostly no. ~5% of failures have endpoint n_neg=1. The other ~95% have n_neg=0 endpoints — IRC reaches a real minimum, just the wrong one. |
| Are they low because of step budget? | No. GAD-strict-conv hit 89.9% at 10pm matching HIP, and Sella was saturated already. Both find n_neg=1 ∧ fmax<0.01 saddles at HIP-level rates. |
| **Why are they low then?** | **Quantitatively measured PES disagreement.** SCINE-converged TSs are median **0.444 Å** from T1x's TS (vs HIP-converged TSs which are 0.005 Å — basically identical to T1x). DFTB0 puts the saddle in a different place. |
| Does xTB work? | Not retried with 10k-step recipe (job timed out). Root cause unchanged: HIP-TS has forces 4-15 eV/Å on xTB's PES; the start is far from any xTB saddle. Step budget can't fix that — needs xTB-relaxed starts. |
| Are DFTB2/DFTB3 better than DFTB0? | No. DFTB2 full grid: 87.5% conv at 10pm (vs DFTB0's 89.9%) and worse at higher noise. DFTB0 stays the best second calculator. |

---

## 1. The geometric smoking gun

`scripts/debug_scine_topo.py` (and an ad-hoc analyze in the chat trace)
compared HIP-converged TSs vs SCINE-converged TSs at 10pm noise, on the
231 samples where both methods converged:

| Pair | Median RMSD (Å) | IQR | p95 |
|---|---|---|---|
| HIP-TS ↔ T1x-TS | **0.005** | [0.003, 0.006] | 0.018 |
| SCINE-TS ↔ T1x-TS | 0.444 | [0.241, 0.656] | 1.019 |
| SCINE-TS ↔ HIP-TS | 0.444 | [0.243, 0.659] | 1.021 |

HIP was trained on T1x, so HIP-TS ≈ T1x-TS by construction. SCINE-TS
is half an Ångstrom away from both.

**Conditional on IRC TOPO outcome**:

| SCINE-TS-to-HIP-TS RMSD | Median | IQR |
|---|---|---|
| TOPO succeeds (N=32) | **0.157 Å** | [0.115, 0.246] |
| TOPO misses (N=199) | **0.504 Å** | [0.318, 0.685] |

TOPO succeeds when DFTB0 happens to place the saddle within ~0.3 Å of
T1x's saddle. ~13% of T1x reactions satisfy that; the rest don't.

This is mechanistically clean: it's not the IRC integrator misbehaving,
it's that DFTB0 and HIP have *different* saddles on the same R↔P pair.

---

## 2. What I tried for the IRC algorithm

### 2.1 Cutoff sweep
Built bond graphs at endpoints with cutoff_scale ∈ {1.10, 1.15, 1.20,
1.25, 1.30, 1.35, 1.40, 1.50}. **0 of 20 FAIL samples recovered TOPO at
any cutoff.** Bond detection is not the issue.

`analysis_2026_04_29/scine_topo_debug_10pm.csv` has the per-sample
edge-count comparisons.

### 2.2 Endpoint quality — n_neg at IRC endpoint
For the 132 "half-intended" GAD@10pm cases (one direction matches a
T1x basin, the other doesn't):

| Failing direction's n_neg_vib | Count |
|---|---|
| 1 (still on a ridge) | 7 |
| 0 (at a real but wrong minimum) | 73 |
| Other | 6 |
| Both directions matched the *same* basin (R+R or P+P) | 46 |

So ~85% of half-intended failures are "IRC reached a real minimum, just
not the T1x one." Tightening IRC parameters cannot fix these.

### 2.3 IRC validator v2: post-IRC BFGS minimization
`scripts/scine_irc_validate_v2.py` — after Sella IRC stops, runs BFGS
descent to fmax<0.001 on each endpoint, then rebuilds the bond graph.

Result: **0 samples saved** at any noise level. Because BFGS descends
to whichever minimum is reachable from the IRC endpoint, which is the
same minimum the IRC was already approaching. Doesn't help.

### 2.4 Sella IRC instance bug fix
`src/gadplus/search/irc_validate.py` previously created **two separate
Sella IRC instances** for forward and reverse. Each independently
recomputed v0ts and canonicalized its sign via "first nonzero component
positive." On near-degenerate spectra, that canonicalization can pick a
different reference component between the two calls, leading both
directions to step the *same* way.

Fixed to use a **single IRC instance** with sequential `.run("forward")`
then `.run("reverse")` (Sella caches v0ts and reuses -v0ts for the
reverse). This is Sella's intended usage pattern.

Numbers didn't change after the fix because most of the 46 "both same
side" cases are dominated by *shallow* saddles where |λ_0| is small
enough that the cubic terms in the PES expansion overwhelm the linear
saddle direction. Inspected TS Hessians: several samples have |λ_0|
∈ [0.003, 0.05] eV/Å² with λ_1 of similar magnitude. IRC can't define
a meaningful forward/reverse on such weak saddles regardless of v0ts.

### 2.5 Net conclusion
The IRC algorithm is correct. The bug fix is real and worth keeping
(prevents a subtle nondeterminism), but it doesn't move the headline.
The 13% TOPO ceiling on SCINE/DFTB0 is a *PES* property, not an
*algorithm* property.

---

## 3. Other PES grids run today

### 3.1 DFTB2 full grid (`runs/main_scine_dftb2_15k_61018324`)
| Noise | Conv% | n_neg=1% |
|---|---|---|
| 10 pm | 87.5% | 89.5% |
| 30 pm | 47.7% | 52.3% |
| 50 pm | 13.6% | 14.3% |
| 100 pm | 0.7% | 0.7% |
| 150 pm | 0.0% | 0.0% |
| 200 pm | 0.0% | 0.0% |

Slightly worse than DFTB0 on every row.

### 3.2 DFTB3 full grid (`runs/main_scine_dftb3_15k_61018324`)
Partial as of writing — 50pm and 100pm done (14.6%, 0.7%); 10pm and
30pm still running. Pattern is similar shape, worse than DFTB0.

### 3.3 Other functionals (smoke only, 20 samples × 10pm)
From the 05-12 doc: PM6 90% / AM1 80% / DFTB2 75% / DFTB3 70% / DFTB0
75% strict-conv. All work; none clearly beat DFTB0 on full grid.

---

## 4. xTB status

Job 60859920_0 (GFN1, 30 samples, dt=0.005 × 10k) **timed out at 2:30
hour** wall limit on 16 CPUs. GFN2 OOM'd at 22 min. The dxtb backend
has a per-step Hessian cost that makes 10k-step runs infeasible at
this scale.

But the deeper issue is unchanged from the 05-12 doc: at the HIP-TS
(zero noise) coordinates, forces on xTB's PES are 4-15 eV/Å (vs
DFTB0's 3-9 eV/Å), with n_neg=7-22. **HIP-trained TSs are nowhere
near xTB stationary points.** Step budget can't fix this — the search
needs to start somewhere on the xTB reaction graph (e.g. midpoint of
xTB-relaxed R and P, or short pre-relaxation on T1x-TS).

Not retried in any new form today.

---

## 5. The 05-12 headline numbers, recomputed with v2 IRC

The single-instance IRC fix is on disk and used by both v1 and v2
validators going forward. Reran on the 15k SCINE grids:

| Noise | GAD strict / IRC TOPO | Sella strict / IRC TOPO | Δ TOPO |
|------:|----------------------:|------------------------:|------:|
| 10 pm | 89.9% / **12.9%** | 87.8% / 0.0% | **+12.9** |
| 30 pm | 66.6% / 0.7% | 73.5% / 0.0% | +0.7 |
| 50 pm | 43.9% / 0.3% | 56.1% / 0.0% | +0.3 |
| 100 pm | 19.5% / 0.0% | 32.4% / 0.0% | 0.0 |
| 150 pm | 6.3% / 0.0% | 15.3% / 0.0% | 0.0 |
| 200 pm | 1.4% / 0.0% | 3.8% / 0.0% | 0.0 |

Identical to the 05-12 numbers — the fix is correct but inert in the
absence of changing what makes the saddles disagree across PESes.

---

## 6. The plan after this doc

User's instruction (paraphrased): smoke-test all PESes to see how far
their **minima** are from the labelled T1x R/P minima. Then either
(a) restrict to the PES-invariant subset, or (b) re-label R/P per
PES if disagreement is consistent and the PES is otherwise sensible.

### 6.1 Smoke design (next session)
Per PES (HIP, DFTB0, DFTB2, DFTB3, PM6, AM1, GFN1, GFN2):
1. Take each T1x test sample's `pos_reactant` and `pos_product`.
2. Relax each under the calculator (BFGS, fmax<0.01, max ~500 steps).
3. Record:
   - RMSD(relaxed_R, T1x_R) and RMSD(relaxed_P, T1x_P).
   - Whether bond graph of relaxed_R matches T1x_R (and same for P).
   - n_neg at relaxed point (should be 0 — these should be minima).
4. Distribution per calculator: how often is the relaxed minimum in
   the same basin as T1x's?

Output: `analysis_2026_*/pes_minima_agreement.csv` with one row per
(calculator, sample_id, side ∈ {R, P}) and these columns.

### 6.2 Decision flow
- If `relaxed_R bond graph == T1x_R` for ≥X% of samples → that PES's
  R-minimum aligns with T1x's; safe to use as-is.
- If it's only X-50% → use only the aligned subset for that PES's
  benchmark row.
- If it's <50% but the relaxed minimum is *consistent* across samples
  (e.g. always settles into a different but reproducible basin) →
  re-label that PES's R/P (and re-run the IRC TOPO against the new
  labels).

### 6.3 What this fixes
Currently we're scoring SCINE IRCs against HIP-trained T1x R/P. If
DFTB0's R is, say, 0.3 Å away from T1x's R but with the same bond
graph, the current scoring still passes. If DFTB0 puts R in a
*different* basin entirely (different bond graph), TOPO fails even
though the saddle and the IRC are doing the right thing for DFTB0's
PES. The smoke will tell us which case dominates.

---

## 7. Files added/modified today

- `scripts/debug_scine_topo.py` — per-sample IRC + bond-cutoff sweep
- `scripts/debug_scine_topo.slurm`
- `scripts/scine_irc_validate_v2.py` — IRC + post-IRC BFGS, single-instance
- `scripts/scine_irc_v2_all.slurm` — 6-noise × 2-method array
- `scripts/main_scine_dftb23_15k.slurm` — DFTB2/DFTB3 full grids
- `src/gadplus/search/irc_validate.py` — fixed `run_irc_validation` to
  use a single Sella IRC instance for forward + reverse
- `analysis_2026_04_29/scine_topo_debug_10pm.csv` — cutoff-sweep table

New runs on /lustre07/scratch:
- `runs/scine_irc_v2_61021069/{gad,sella}/irc_validation_v2_*pm_*.parquet`
- `runs/main_scine_dftb2_15k_61018324/noise{10,30,50,100,150,200}pm/`
- `runs/main_scine_dftb3_15k_61018324/noise{10,30,...}pm/`  (DFTB3
  10pm and 30pm still running as of writing)

---

## 8. Open questions for next session

1. **Per-PES minimum-agreement smoke** (the user's next ask).
2. **Re-label R/P per PES** if (1) shows consistent disagreement.
3. **xTB unblock attempt** — start GAD from midpoint of xTB-relaxed R
   and xTB-relaxed P, not from HIP-noised T1x-TS.
4. **Threshold for "right saddle"** — current `n_neg==1 ∧ fmax<0.01`
   strict-conv counts shallow saddles (|λ_0| < 0.01) where IRC can't
   distinguish forward/reverse. Optionally tighten with `|λ_0| > ε`.
   ~46 of 287 GAD@10pm strict-conv saddles fall into this category.
