# In-flight finding: hybrid struggles starting from reactant (2026-05-16)

## REVISION 2026-05-16 (cron iteration 9)

The "0% conv" claim was correct **only for the early sample range (0-74)**.
At $n \approx 78$, samples 75-78 (all C3H5NO2) start converging in ~250 steps.
The GAD walking phase reaches the saddle manifold faster for some molecular
species than others; the early test-set samples (C2H3N3O2) happen to be in
the slow-convergence regime.

| Cell | Updated conv % |
|---|---|
| Hybrid damped reactant | **1.3% (n=78)** — was 0% at n<75 |
| Hybrid undamped reactant | **5.1% (n=79)** — was 0% at n<75 |

**Corrected mechanism:** the hybrid is **case-dependent** from reactant, not
categorically broken. The 4 converged C3H5NO2 samples confirm the
eig-switch DOES fire when the GAD walk reaches $n_\mathrm{neg}=1$. The 0%
finding for sample range 0-74 reflects that C2H3N3O2 reactants are
geometrically far from their saddles and the 2000-step GAD walk isn't
sufficient to bridge that distance.

**Implication:** the predicted "5000+ step budget" fix is on track. SLURM
61091399 (10000-step budget) should rescue many of the currently-failing
samples. ETA $\sim$5 h.

---

## REVISION 2 (FINAL, 2026-05-16 end of wave)

**Wave 1 final n=287:** Both hybrid damped and undamped converged to
**2.1% conv (6 of 287)**. The "1.3 / 5.1%" mid-run numbers were small-n
noise that resolved with more samples.

**Wave 2 (10000-step budget, n=72 partial — SLURM timed out):**
**0 of 72 samples converged.** Notably, a C3H5NO2 cluster (samples 60-69)
ended at $n_\mathrm{neg}=0$ and $F_\mathrm{max}=0.155$ — i.e., trajectory
was still descending toward a minimum, not climbing toward a saddle. The
"5000+ step fix" prediction is **wrong**.

**Corrected mechanism (third revision, final):** the issue is not step
budget. It is the geometric distance from the reactant to the saddle in
coordinate space. For most reactants in T1x, GAD-from-reactant starts in
a basin of attraction that pulls *toward a minimum*, not *toward the
nearest saddle*. No GAD-only walking budget can overcome this — the
trajectory just plateaus at a minimum.

The hybrid is therefore **fundamentally not a basin-finder when started
from a minimum**. Useful predicted fixes (untested):
1. Combine GAD with an outward force perturbation when $n_\mathrm{neg}=0$
   persists for $>$N steps (jiggling the trajectory off the minimum basin)
2. Start from a midpoint or noised TS, not a true reactant minimum
3. Replace the GAD walking phase with a more aggressive saddle-search
   (e.g.\ NEB or Dimer) until $n_\mathrm{neg}=1$, then switch to Newton

The honest paper framing: \textbf{the hybrid (and plain GAD) require a
near-saddle starting condition.} Their advantage is on noised-TS sweeps,
not on reactant starts. Sella is the right tool for reactant starts.

---

## Original snapshot (n=11/287, log-parsed mid-run)

| Method | Starting condition | Conv % (fmax<0.01) |
|---|---|---|
| Sella cart+Eckart untuned d=1 | Reactant | 80.8% (n=287, full) |
| Plain GAD dt=0.003           | Reactant | 40.1% (n=152, partial) |
| Plain GAD dt=0.005           | Reactant | 54.2% (n=179, partial) |
| **Hybrid damped Eckart eig tr=0.05** | **Reactant** | **0.0% (n=11)** |
| **Hybrid undamped Eckart eig tr=0.05** | **Reactant** | **0.0% (n=11)** |

## Hypothesised mechanism

The hybrid's eig-switch trigger only fires when the Hessian's vibrational
spectrum has $n_\text{neg}=1$ — i.e.\ when the trajectory has reached the
saddle manifold. The Newton step is then well-defined and lands at a nearby
saddle.

Starting from the reactant, the trajectory begins at $n_\text{neg}=0$
(a local minimum, by definition). The GAD walking phase must drag the
geometry up onto the saddle manifold before Newton can fire. Within the
2000-step budget, the GAD walk apparently never reaches $n_\text{neg}=1$
for any of the first 11 samples.

Plain GAD has the same walking mechanism but logs the full trajectory and
counts $n_\text{neg}=1 \wedge F_\text{max}<0.01$ at any step — so when GAD
*does* reach the saddle manifold, it can converge there. The hybrid's
"GAD-then-Newton" sequence is held hostage by the eig-switch never firing.

## Predicted fixes (not yet tested)

1. **Increase step budget to 5000+**: GAD from reactant typically needs
   $\gtrsim$2000 steps to reach the saddle manifold. Hybrid inherits the
   same slow walk.
2. **Force-switch instead of eig-switch**: trigger Newton when
   $\|F\|_\text{internal} < $ threshold, regardless of $n_\text{neg}$.
   This fires earlier but on possibly-wrong-sign curvature, so Newton
   may climb instead of descend on the wrong eigenvector.
3. **Composite criterion**: $n_\text{neg}=1 \vee \|F\| < $ threshold.

## What this changes about the headline story

The reactant-start probe shows hybrid's wall/conv lead **does not transfer
to reactant starts**. Sella's quadratic model + trust-region step locates
nearby saddles directly; GAD walks there slowly; hybrid waits for GAD to
hand it the saddle manifold.

Conclusion: **the hybrid is a Newton accelerator for the saddle-finding
mechanism, not an alternative basin-finder.** Its advantage shows up when
the starting geometry is already near a saddle (noised-TS sweep), not when
it has to find one from scratch.

## Confidence

- Live data only (log-parsed), n=11 per hybrid cell.
- Even if 1/11 = 9% of the remaining 276 samples converge, the cell finishes at $\sim$9%, which would only marginally change the qualitative story.
- Final n=287 will arrive when SLURM 61087603_{0,1} land (~9.5 h ETA).

## Source

- `/lustre07/scratch/memoozd/gadplus/logs/compr_61087603_0.out`
- `/lustre07/scratch/memoozd/gadplus/logs/compr_61087603_1.out`
- `/lustre07/scratch/memoozd/gadplus/runs/start_reactant_hybrid/{damped,undamped}_dt5e-3_tr0.05/traj_*.parquet`
