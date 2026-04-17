# Backburner

Deferred items. Move into `EXPERIMENT_LOG.md` / `HANDOFF_*.md` when picked up.

---

## 1. Re-run Sella baselines with coords logged

**Why:** The Sella TS-optimization baselines (Exp 14/15, jobs 58932967 /
58937673) write a summary parquet with all the stats — but not the final
TS coordinates. That means we cannot run our `sella_hip` IRC validator
on the Sella-found TSs without re-deriving coords.

**What to do:**
- Modify `scripts/sella_baseline.py` to append `coords_flat` (and maybe
  `atomic_nums`) to each result row before writing the parquet.
- Re-run the best config only: `sella_cartesian_eckart_fmax0p01`,
  1000 steps, noise levels 10/30/50/100/150/200 pm × 300 samples.
  Output to `/lustre07/scratch/memoozd/gadplus/runs/sella_1000_coords/`.
- Then run the existing `scripts/irc_validate.py --method sella_hip`
  against those summaries, same as the gad_dt003 IRC sweep that produced
  `IRC_RESULTS_2026-04-16.tex`.
- Deliverable: head-to-head table & bar charts — sella_hip IRC outcomes
  on Sella-optimized TSs vs. gad_dt003 TSs, same 300-sample universe,
  same 6 noise levels, same three criteria (TOPO / RMSD / endpoint vib).

**Compute:** 6 × ~1 hr tasks under rrg-aspuru; parallel dispatch.

**Precedent concern:** the original Sella runs used seed 42 with
`torch.manual_seed` + sequential `torch.randn_like` to generate noise.
Any rerun must match that seeding exactly for the TS set to be the same
TSs that experiment 15 scored. Simplest: don't try to match bit-exactly —
accept that we're re-running Sella with the same hyperparameters and
same noise distribution, which is the statistically meaningful thing.

---

## 2. IRC-as-convergence-criterion (run IRC on *every* sample endpoint)

**Why:** The current `n_neg == 1 ∧ force_norm < 0.01` convergence gate is a
local check — it certifies a saddle but says nothing about which
reaction the saddle connects. An IRC-based criterion is stronger:
``the TS is valid iff IRC from it connects two real minima that match
the labeled (or at least two chemically sensible) endpoints.''

Before running IRC on every sample (converged or not), we need to lock
down the precise criterion. Open questions:

- **Which IRC outcome counts as "converged"?**
  - TOPO-intended (bond graph on both sides matches label)?
  - TOPO-half (one side matches label, other is still a real minimum)?
  - "both endpoints are vibrational minima and have different bond graphs"?
    (chemically-meaningful saddle even if labels don't match)
  - Some weighted combination?
- **Do we still require the local gate?**
  I.e. is the new criterion `IRC-pass` alone, or
  `IRC-pass ∧ (n_neg == 1 ∧ force_norm < threshold)`?
- **What IRC integrator is definitive?**
  `sella_hip` is much faster (~12 s / sample) and matches topology
  at 92.1% on gad_dt003 TSs. `rigorous` is 10× slower and currently
  scoring lower (see item 3 below). Probably `sella_hip` is the
  practical choice, but confirm the rigorous investigation first.
- **Ridge-stall handling.** If IRC ends with `n_neg_vib > 0` on either
  endpoint (0.5-1.8% of runs), that's neither a pass nor a fail under the
  current labels. Define explicitly.
- **Force threshold for each IRC endpoint.** Currently we just check
  `n_neg_vib`; should we also require `|F|_endpoint < ε` to call it a
  real minimum?

**Compute scale when running on every sample.** Currently we run IRC on
~95% × 300 samples × 6 noise = ~1700 runs. Running on *all* 300 × 6
= 1800 samples × 2 IRC directions × 500 steps. sella_hip at ~12 s/sample
fwd+rev → 6 hr on a single GPU, ~1 hr across 6 parallel MIGs. Tractable.

The harder part is scaling to the full Transition1x train split
(9,561 samples) if we ever want IRC-as-criterion at production scale.
That's 9561/300 ≈ 32× the cost — ~32 hr across 6 MIGs, or ~5 hr if we
scale to 32 MIGs. Still very doable under rrg-aspuru priority.

**Pre-work before launching.** Draft a short spec (1 page, could live
next to `IRC_OVERVIEW.md`) covering: definition of the criterion, edge
cases, acceptance thresholds, and what metric to report at the study
level. Run the spec by the user before any big sweep.

---

## 3. Investigate why `rigorous` underperforms `sella_hip`

**Status (as of 2026-04-17):** Full rigorous sweep is running
(job 59464202, ETA mid-morning). Mid-sweep telemetry at 51-69% complete
per task shows rigorous TOPO-intended at ~55-61% vs sella_hip's 93-94%
on the same TS set — a ~35 pp gap. This is too large to dismiss as
``different integrator, slightly different answer.''

**Hypotheses to check when the sweep finishes:**
- **Step-size too aggressive.** `rigorous` arc-length defaults are
  `s_min=0.01`, `s_max=0.15`, `alpha_clamp=0.3`. If it's overshooting
  past the correct minimum basin, we'd see TOPO failures but not
  `neither at minimum`. Check endpoint-quality breakdown.
- **Eckart-projected gradient too aggressive near minima.** The
  rigorous integrator projects both gradient and Hessian every step;
  near a minimum the residual mass-weighted gradient may be tiny and
  the remaining numerical noise may push toward the wrong basin.
- **Initial kick direction.** rigorous uses the lowest \emph{vibrational}
  eigenvector, skipping any residual TR-like mode. If the TR projection
  is imperfect at the TS (e.g. because `gad_dt003` TSs aren't perfectly
  Eckart-projected), the initial direction might be slightly wrong.
  sella_hip doesn't do this — it just lets Sella handle the first step.
- **K-step hold for convergence.** rigorous requires 2 consecutive
  ``converged'' flags before stopping. If the criterion is a bit
  permissive, it may stop in a shoulder region rather than the true
  minimum.
- **Hydrogen-transfer reactions.** Several of the gad_dt003 TSs that
  sella_hip handles well involve H migration. Check whether rigorous
  failures concentrate in formulas with mobile H.

**What to produce:**
- Same 6 bar charts as for sella_hip, same three criteria.
- Per-(method, noise) head-to-head table (4 outcome buckets).
- Per-sample diff: which samples did sella_hip get right but rigorous
  got wrong? If the set is small and has a common structural feature,
  that's the smoking gun.
- If a hyperparameter is obviously at fault, try one tightened rerun
  (e.g. `alpha_clamp=0.15`, or `s_max=0.08`) on a subset and see if
  it closes the gap. Don't bulk-sweep hyperparameters — pick one
  candidate based on the failure pattern.

**Decision if still far behind after investigation:** drop `rigorous`
as a production IRC. The predictor-corrector machinery was built to
be a ``more principled'' alternative to sella_hip; if it scores worse
on the same TS set, that's a negative result worth recording but not a
method we use downstream.

---

## Housekeeping

- Dates are absolute (2026-04-NN), not relative.
- When an item moves out of backburner, delete it from this file and
  add a line to `EXPERIMENT_LOG.md` pointing to the new entry.
