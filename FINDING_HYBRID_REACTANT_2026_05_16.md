# In-flight finding: hybrid struggles starting from reactant (2026-05-16)

## Snapshot (n=11/287, log-parsed mid-run)

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
