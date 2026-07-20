# HIP vs LJ vs SCINE: Why GAD Behaves Differently

Short exploratory note for the paper narrative. This compares existing HIP/SCINE artifacts against the recent LJ diagnostics.

## Main Hypothesis

HIP, LJ, and SCINE stress different failure modes:

- **HIP/T1x:** GAD usually starts in a smooth, trained local TS basin. Failures are mostly near-threshold or IRC/topology questions, not violent optimizer instability.
- **LJ7:** Noisy starts can sit on the steep repulsive wall. Pure Euler GAD then sees enormous forces/curvatures, so fixed `dt`/cap rules become a bad trust-region surrogate.
- **SCINE/DFTB0:** GAD can often satisfy `n_neg=1` and `fmax`, but IRC/topology is low because DFTB0's PES places many saddles away from the T1x/HIP saddle. That is a PES disagreement problem, not primarily an optimizer instability problem.

## HIP vs LJ Step-Scale Evidence

Canonical HIP `gad_dt003_fmax` at 200 pm:

- Initial median `fmax`: about `9.7` eV/A.
- Initial median `n_neg`: `4`.
- By the first 100 logged steps, median `fmax` drops to about `1.4` and median `n_neg` is already `1`.
- HIP failures at 200 pm usually remain near saddle quality: failed-run median final `fmax` about `0.076`, median final `n_neg = 1`.

LJ7 at noise `0.20`, hydrogen mass, cap `0.005`:

- Full run median initial `fmax`: about `1969`.
- Hard-failure replay-panel initial median `fmax`: about `35660`.
- Hard replay-panel first-step median raw atom displacement: about `213`, with p90 about `2292`, before capping.
- Even after 100 replay steps, median `n_neg` stayed around `7` in the selected hard panel.
- Failed full-run median final `fmax` at `noise=0.20`, `dt=0.005`: about `0.406`, median final `n_neg = 3`.

This is the core distinction: HIP enters a smooth contraction regime quickly; LJ spends many steps in a singularly stiff, multi-negative-mode regime where every fixed-step rule is either too large or too slow.

## Static Cap Trial Result

On the same LJ hard-failure panel (`104, 109, 112, 139, 167, 219`) plus controls, 800-step smokes gave:

| Trial | Hard failures rescued | Median hard-failure final fmax | Median hard-failure final n_neg |
|---|---:|---:|---:|
| baseline cap `0.005` | 0/6 | 1.65 | 4.5 |
| fixed cap `0.001` | 0/6 | 1.68 | 4.5 |
| ramp `1e-4 -> 0.005` | 0/6 | 1.60 | 4.5 |
| ramp `2e-4 -> 0.005` | 0/6 | 1.61 | 4.5 |
| lower `dt=0.001` | 0/6 | 1.65 | 4.5 |

Interpretation: capping is too naive. It changes speed and cap-hit counts but does not supply a per-step acceptance criterion.

## Why HIP Worked So Well

Likely reasons:

- The HIP potential is trained on the same Transition1x distribution and TS labels, so noised starts are generally close to the intended saddle basin.
- Force/curvature scales are moderate compared with LJ's repulsive wall.
- The unstable mode is reasonably persistent; logged HIP trajectories show high eigenvector continuity and quickly settle to `n_neg=1`.
- HIP failures often look like slow force cleanup near a valid saddle, not high-index trapping.

This makes fixed-step Euler GAD surprisingly effective: `dt=0.003` or `0.005` is small enough to avoid catastrophic oversteps but large enough to contract in the local basin.

## Why LJ Failed

Likely reasons:

- The LJ repulsive core creates enormous local force and Hessian scales after noise.
- Noise can create near-collisions that are not analogous to chemically plausible T1x perturbations.
- The GAD direction may initially point toward the safe reference, but the trajectory remains capped and high-index for too long.
- A global cap or global `dt` cannot choose between "take tiny damage-control steps" and "grow once the local model is trustworthy."

The LJ result is therefore a good stress test for optimizer globalization. Pure Euler GAD lacks trust-region acceptance.

## SCINE DFTB0 Comparator

Existing SCINE notes show a different failure mode:

- With a longer budget (`dt=0.007`, 15k steps), SCINE/DFTB0 GAD reaches HIP-like strict convergence at low noise.
- IRC/topology remains low because SCINE-converged TSs are geometrically displaced from the T1x/HIP TS.
- Prior analysis found SCINE-TS to HIP/T1x-TS median RMSD around `0.44 A`, while HIP-TS to T1x-TS is about `0.005 A`.
- Sella on DFTB0 can report high strict convergence but `0%` IRC topology in those runs, meaning it finds real but wrong saddles.

So SCINE supports a useful distinction:

- **TS-quality convergence** asks whether the optimizer can find some first-order stationary point on the chosen PES.
- **IRC/topology recovery** asks whether that saddle is the intended T1x reaction saddle.

LJ is currently failing mostly at the first level for hard starts; SCINE often passes the first level but fails the second because the PES differs.

## Paper Angle

A clean paper framing would be:

1. HIP demonstrates that pure projected GAD is effective when the PES is smooth and the start lies inside a realistic TS basin.
2. SCINE shows that strict saddle convergence is not enough: IRC/topology can fail when the calculator PES disagrees with the reference reaction.
3. LJ shows that pure Euler GAD has a globalization weakness on singular/stiff surfaces; the missing ingredient is adaptive trust-region/acceptance logic.

This makes the LJ result a feature, not just a failure: it identifies where the method needs a real optimizer globalization layer rather than another fixed step-size tune.
