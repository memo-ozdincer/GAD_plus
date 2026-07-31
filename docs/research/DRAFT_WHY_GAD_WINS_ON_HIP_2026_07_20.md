# Why Does Plain GAD Outperform Sella on HIP at High Noise, but Not on Other Surfaces?

## A focused cross-surface results draft

**Status:** Internal working draft, 2026-07-20.  
**Scope:** This document isolates the cross-surface optimizer result. It does
not attempt to reproduce the full experiment history or claim that the causal
mechanism has been proved.

## Abstract

We compared projected, single-mode gentlest-ascent-like dynamics (GAD) with
Sella's full-Hessian restricted-step partitioned rational-function optimizer
(RS-P-RFO) on the Hessian Interatomic Potential (HIP) and several control
potential-energy surfaces. On 287 noised Transition1x test structures
evaluated with HIP, Sella gives higher strict saddle recovery close to the
labelled transition state, the methods tie near a Cartesian noise standard
deviation of 0.10 A, and plain GAD is more robust at 0.15-0.20 A. At 0.20 A,
the best completed plain-GAD and Sella strict rates are 44.6% and 27.2%;
intended IRC-topology recovery is 44.6% and 23.3% for fixed principal
configurations.

This ranking does not transfer to the control surfaces. On SCINE DFTB0, GAD
leads strict convergence by 2.1 percentage points at 0.01 A, but Sella leads
at every larger perturbation. On exact analytic LJ7, Sella strongly
outperforms pure GAD. A smooth index-aware gate raises LJ strict recovery to
99.7-100%, but that method first suppresses ascent while the second
vibrational eigenvalue is negative and is therefore not pure GAD. A small
conservative PaiNN pilot gives 0/12 strict successes for GAD and 5/12 for
Sella, although neither method recovers the predeclared intended chemistry.
The HORM and MACE screens are not sufficiently native or matched to support
an optimizer ranking.

The results support a HIP-specific, high-noise optimizer crossover rather
than general GAD superiority. The leading explanation is that HIP's
separately learned energy, direct force, and directly supervised Hessian may
preserve a useful lowest-mode projector beyond the region where all three
products form a reliable finite-step quadratic model. Plain GAD uses the
force and that rank-one projector; Sella uses the full curvature model and an
energy-model trust update. HIP shows much larger finite-difference
energy-force-Hessian incompatibility than conservative PaiNN, and failed
Sella trajectories on HIP exhibit departure followed by trust-radius
collapse. These observations make the mechanism plausible but do not prove
it. No outcome-conditioned Taylor-residual panel or causal rescue
intervention has been completed.

---

## 1. The Narrow Claim

The data do **not** support the claim that GAD is generally better than Sella.
They support the following narrower statement:

> On one HIP checkpoint and isotropically noised Transition1x test
> structures, plain projected GAD is more robust than the best completed
> full-Hessian Sella baseline at the largest perturbations. The crossover is
> retained under intended IRC-topology validation. It is absent from the
> completed DFTB0 and exact-LJ comparisons and has not been positively
> replicated on another MLIP.

Even on HIP, GAD is not uniformly better. Sella leads near the labelled
saddle and converges in fewer optimizer steps. The scientific question is
therefore why HIP produces a **noise-dependent crossover**, not why GAD is
universally superior.

---

## 2. What Was Compared

### 2.1 Plain GAD

The reported plain method reflects the force in the instantaneous lowest
Eckart-projected vibrational mode:

$$
F_{\mathrm{GAD}}
=F-2v_1(v_1^\mathsf{T}F).
$$

The Hessian affects the position update only through the rank-one projector
$v_1v_1^\mathsf{T}$. Once $v_1$ has been selected, plain GAD does not use the
magnitude of its eigenvalue, the remaining stable spectrum, or a predicted
energy change.

This method has a local index-one stability argument. It does not have a
general globalization guarantee from geometries with several negative modes.

### 2.2 Sella

Sella uses RS-P-RFO with separate ascent and stable subspaces and a restricted
step. In the headline HIP benchmark, Sella receives the current full HIP
Hessian at every optimization step. It must not be described as a
quasi-Newton or stale-Hessian baseline.

Sella uses more of the local model than GAD: the direct force, the full
Hessian, and the learned energy used in its actual-versus-predicted model
update. This richer contract can be an advantage when those products are
locally coherent and a disadvantage if they do not approximate one Taylor
expansion over the applied step.

### 2.3 Strict convergence and `fmax`

Unless explicitly stated otherwise, strict saddle convergence means

$$
n_{\mathrm{neg}}=1
\quad\land\quad
F_{\max}<0.01,
$$

where $n_{\mathrm{neg}}$ is recomputed from the Eckart-projected vibrational
Hessian and $F_{\max}$ is the maximum absolute Cartesian force component.
HIP, DFTB0, and PaiNN use eV/A force units. LJ uses reduced
$\epsilon/\sigma$ force units, so its force magnitudes cannot be compared
numerically with molecular potentials.

The 0.01 threshold is stricter than ASE's commonly used 0.05 setting but
looser than a 0.001 target. It is also not a chemical identity test.
Whenever possible, strict stationarity is reported separately from
two-sided IRC or PES-native endpoint recovery.

The optimizer ranking depends on this threshold. On HIP, fixed-timestep plain
GAD has a pronounced force floor: across the completed grid it gives no
successes below 0.005 eV/A, including a 10,000-step control at 0.05 A noise.
Sella and Newton-like refinement can enter the sub-0.005 regime, although
neither principal method produced meaningful 0.001-level recovery under the
main budget. The high-noise GAD claim is therefore specifically a
`n_neg = 1` and `fmax < 0.01` robustness claim, strengthened by IRC
connectivity, not a claim of tighter asymptotic optimization.

### 2.4 Projection and masses

All reported projected searches use physical masses to construct the
mass-weighted Eckart/vibrational problem, then un-mass-weight the direction
before applying a Cartesian displacement. The retained-mass-weighting path
errored and produced no benchmark data.

All LJ atoms are assigned hydrogen mass, $m=1.008$. Changing the assigned
element changes the vibrational eigenproblem even though it does not change
the LJ energy. Hydrogen performed better than carbon-like or argon-like
assignments, but mass choice did not explain the pure-GAD failure.

### 2.5 Perturbation scale

The molecular tests add independent Gaussian noise to every Cartesian
coordinate. Historical scripts label 0.01-0.20 A as 10-200 pm by multiplying
by 1000. That label is wrong by a factor of ten because 1 A = 100 pm. This
draft reports the actual coordinate standard deviation in A; LJ is reported
as a fraction of $\sigma$.

Equal coordinate noise does not create equal difficulty on different
surfaces. LJ noise produces close-pair collisions and extreme repulsive
curvature. T1x structures are not necessarily stationary on DFTB0, HORM, or
MACE. Cross-surface differences are therefore mechanistically informative
but are not a perfectly matched benchmark of intrinsic optimizer quality.

---

## 3. HIP Shows a High-Noise Crossover

The primary HIP comparison uses 287 Transition1x test-split structures.
The per-cell-best completed grid is:

| Cartesian noise SD (A) | Best plain GAD | Best Sella | GAD - Sella |
|---:|---:|---:|---:|
| 0.01 | 89.2% | **96.5%** | -7.3 pp |
| 0.03 | 88.9% | **95.5%** | -6.6 pp |
| 0.05 | 85.7% | **92.0%** | -6.3 pp |
| 0.10 | 72.8% | 72.8% | 0.0 pp |
| 0.15 | **58.2%** | 54.0% | +4.2 pp |
| 0.20 | **44.6%** | 27.2% | +17.4 pp |

The best plain GAD configuration at high noise is `dt=0.007`. The strongest
low-noise and high-noise Sella cells come from different completed
Cartesian-Eckart configurations, so the table is an oracle summary of the
completed grid rather than a frozen prospective comparison. The crossover
nevertheless remains visible with fixed principal configurations, especially
at 0.15 and 0.20 A.

The complete principal-configuration grid is:

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| GAD, `dt=0.003` | 89.2 | 88.5 | 85.4 | 71.1 | 55.1 | 40.8 |
| GAD, `dt=0.005` | 89.2 | 88.5 | 85.7 | 71.8 | 57.1 | 43.2 |
| GAD, `dt=0.007` | 89.2 | 88.9 | 85.7 | **72.8** | **58.2** | **44.6** |
| Sella Cartesian, tuned label | 92.0 | 91.3 | 87.5 | 65.5 | 42.9 | 18.8 |
| Sella Cartesian+Eckart, config A | 92.7 | 92.0 | 88.2 | 70.7 | 54.0 | 27.2 |
| Sella internal | 79.1 | 77.4 | 71.8 | 50.9 | 26.8 | 13.9 |
| Sella Cartesian+Eckart, historical config B | **96.5** | **95.5** | **92.0** | **72.8** | 50.5 | 23.3 |

The three plain-GAD timesteps produce the same qualitative crossover. Config
A is the strongest completed Sella row at 0.15-0.20 A; historical config B is
stronger near the labelled saddle but still below GAD at both high-noise
cells. Thus the high-noise conclusion is not created by selecting one
pathological Sella configuration.

### 3.1 Intended IRC recovery

The high-noise difference survives intended-reaction validation. The
fixed-configuration table is:

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| Plain GAD, `dt=0.005` | 88.9 | **89.2** | **88.9** | **78.0** | **61.7** | **44.6** |
| Sella Cartesian+Eckart, config A | **89.2** | **89.2** | 87.5 | 72.5 | 49.8 | 23.3 |

The best completed plain-GAD row at 0.10 is 78.4%, rather than the fixed
configuration's 78.0%. At 0.15 and 0.20 A, the positive result is not merely
caused by the `fmax` stopping definition.

A five-tier rerun also corrected an earlier interpretation. Across 5166
method/noise/sample records, only four outcomes were classified as
unintended. Sella's high-noise deficit is dominated by nonconvergence and
one-sided/partial IRC outcomes, not confident convergence to a different
reaction.

The complete five-tier counts are shown as
`intended / partial / unintended / TS error`:

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---|---|---|---|---|---|
| GAD | 253/33/0/1 | 255/30/0/2 | 250/35/0/2 | 220/49/0/18 | 176/74/0/37 | 130/89/0/68 |
| Hybrid | 256/30/0/1 | 256/29/0/2 | 255/28/0/4 | 221/46/0/20 | 163/73/0/51 | 110/91/0/86 |
| Sella | 254/33/0/0 | 256/30/0/1 | 250/32/0/5 | 209/54/0/24 | 143/88/2/54 | 68/101/2/116 |

The hybrid row is included because it was part of the same complete IRC
campaign, but it is not plain GAD and is not used to define the GAD/Sella
crossover.

### 3.2 Paired trajectory evidence

At 0.15 A, paired strict outcomes are:

| Outcome | Count | Fraction |
|---|---:|---:|
| Both succeed | 133 | 46.3% |
| GAD only | 34 | 11.8% |
| Sella only | 22 | 7.7% |
| Neither | 98 | 34.1% |

The exact paired test on the 56 discordant outcomes gives $p=0.141$, so this
single 0.15 A cell is descriptive rather than statistically decisive.
The larger 0.20 A effect and the intended-IRC trend are the stronger evidence.

Initial index, `fmax`, eigengap, first-step size, time to index one, and
consecutive $v_1$ overlap do not separate GAD-only from Sella-only starts.
Median first maximum-atom displacement is approximately 0.088 A for GAD and
0.100 A for Sella in both exclusive-success classes. The result is not
explained by one universally enormous first Sella step.

The later Sella trajectory is more informative:

| Sella diagnostic | GAD-only starts | Sella-only starts |
|---|---:|---:|
| Median minimum distance to labelled TS (A) | 0.226 | 0.117 |
| Median final distance to labelled TS (A) | 0.447 | 0.128 |
| Median fraction of stored rows at minimum trust radius | 97.9% | approximately 0% |

This establishes a characteristic sequence of departure followed by
trust-radius collapse. It does not establish whether the cause is
energy/force/Hessian incompatibility, early basin selection, trust-policy
hysteresis, or ordinary optimizer tuning.

### 3.3 Full `fmax` threshold dependence

All entries below also require projected $n_{\mathrm{neg}}=1$:

| Method | Noise (A) | <0.05 | <0.023 | <0.01 | <0.005 | <0.001 |
|---|---:|---:|---:|---:|---:|---:|
| GAD `dt=0.007` | 0.01 | 98.3 | 95.1 | 89.2 | 0.0 | 0.0 |
|  | 0.03 | 97.9 | 95.5 | 88.9 | 0.0 | 0.0 |
|  | 0.05 | 94.8 | 91.6 | 85.7 | 0.0 | 0.0 |
|  | 0.10 | 82.6 | 77.7 | 72.8 | 0.0 | 0.0 |
|  | 0.15 | 71.4 | 64.1 | 58.2 | 0.0 | 0.0 |
|  | 0.20 | 68.3 | 56.4 | 44.6 | 0.0 | 0.0 |
| Sella Cartesian+Eckart A | 0.01 | 98.6 | 97.9 | 92.7 | 33.8 | 0.0 |
|  | 0.03 | 98.3 | 97.6 | 92.0 | 32.1 | 0.0 |
|  | 0.05 | 95.1 | 93.7 | 88.2 | 32.4 | 0.0 |
|  | 0.10 | 83.3 | 77.7 | 70.7 | 24.7 | 0.0 |
|  | 0.15 | 66.2 | 59.9 | 54.0 | 15.3 | 0.0 |
|  | 0.20 | 46.0 | 39.0 | 27.2 | 7.0 | 0.0 |

The threshold sweep changes the interpretation substantially. At the common
0.01 criterion, GAD has the high-noise recovery advantage. At 0.005, plain
GAD has no successes and Sella retains 7.0-33.8% depending on noise. At
0.001, both principal rows have zero successes under this budget. A separate
10,000-step GAD run at 0.05 A remains 0/287 below 0.005, confirming that the
plain-GAD floor is not removed by a fivefold step budget.

### 3.4 Efficiency and terminal geometry

The completed timing campaign reports wall seconds per strict success and
median steps among successful trajectories:

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| GAD `dt=0.005` | 47.7 / 100 | 57.1 / 204 | 74.3 / 278 | 156.0 / 458 | 278.5 / 614 | 472.3 / 738 |
| Sella Cartesian+Eckart A | 14.5 / 4 | 15.8 / 6 | 23.2 / 7 | 65.2 / 9 | 132.6 / 11 | 393.8 / 13 |

Sella requires far fewer optimizer iterations. The wall gap is smaller
because implementation and hardware utilization differ, and these timings
must be reprofilied on a frozen allocation before publication.

Among strict successes, Kabsch/Hungarian RMSD to the labelled T1x saddle is:

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| GAD `dt=0.005`, median/p95 A | .005/.018 | .008/.021 | .011/.028 | .014/.044 | .016/.088 | .014/.456 |
| Sella Cartesian+Eckart A, median/p95 A | .008/.073 | .009/.071 | .009/.072 | .009/.201 | .013/.617 | .017/.838 |

These geometry statistics are consistent with broader high-noise tails for
Sella, but HIP and T1x share training provenance and RMSD is not a substitute
for endpoint identity.

---

## 4. DFTB0 Does Not Reproduce the HIP Crossover

SCINE DFTB0 supplies energy, gradient, and Hessian from one semiempirical
electronic-structure approximation. On the same 287 noised T1x structures,
GAD has a small low-noise lead and Sella leads thereafter:

| Noise SD (A) | GAD strict | Sella strict | GAD - Sella |
|---:|---:|---:|---:|
| 0.01 | **89.9%** | 87.8% | +2.1 pp |
| 0.03 | 66.6% | **73.5%** | -6.9 pp |
| 0.05 | 43.9% | **56.1%** | -12.2 pp |
| 0.10 | 19.5% | **32.4%** | -12.9 pp |
| 0.15 | 6.3% | **15.3%** | -9.0 pp |
| 0.20 | 1.4% | **3.8%** | -2.4 pp |

The GAD runs use the completed long-budget recipe, while Sella was already
saturated under its standard budget. The ranking is not explained by
starving Sella of steps.

### 4.1 Native endpoint caveat

The initial DFTB0 IRC analysis was invalid because the installed ASE
generator could accept a saddle at IRC step zero. The corrected validator
forces the initial displacement, applies saddle-aware endpoint convergence,
and scores both methods against one shared set of DFTB0-relaxed endpoints.

At 0.01 A:

| Outcome | GAD | Sella |
|---|---:|---:|
| Strict search success | **258/287** | 252/287 |
| DFTB0-native topology correct | 42/287 | **55/287** |
| DFTB0-native strict geometry correct | 21/287 | 21/287 |

On the paired evaluable subset, topology favors Sella 51 to 41 and strict
geometry is effectively tied, 20 to 21. Thus even the one DFTB0 cell where
GAD has a strict-convergence lead does not support better chemical recovery.

### 4.2 Surface-label mismatch

On 231 structures where HIP and DFTB0 searches both converge, the median
HIP-to-T1x TS RMSD is 0.005 A, whereas the median DFTB0-to-T1x TS RMSD is
0.444 A. DFTB0 is optimizing a meaningfully different surface. This makes
the original T1x IRC labels unsuitable as the only chemical score and also
means the same Cartesian perturbation is not equally local on HIP and DFTB0.

| Geometry pair | Median RMSD (A) | IQR (A) | p95 (A) |
|---|---:|---:|---:|
| HIP TS vs T1x TS | **0.005** | 0.003-0.006 | 0.018 |
| DFTB0 TS vs T1x TS | 0.444 | 0.241-0.656 | 1.019 |
| DFTB0 TS vs HIP TS | 0.444 | 0.243-0.659 | 1.021 |

Among candidates evaluated under the original cross-PES topology score,
successful cases have median DFTB0-to-HIP TS RMSD 0.157 A, while failures
have median 0.504 A. A bond cutoff sweep from 1.10 to 1.50 and post-IRC BFGS
minimization rescue none of the inspected failures.

The defensible DFTB0 result is still clear: with internally coherent local
derivatives, Sella exploits the full Hessian effectively and plain GAD does
not show HIP-like high-noise robustness.

Other semiempirical probes are:

| Surface/probe | Completed result | Status |
|---|---|---|
| DFTB2 full GAD grid | 87.5, 47.7, 13.6, 0.7, 0.0, 0.0% strict over 0.01-0.20 A | No matched Sella grid |
| DFTB3 partial | 14.6% at 0.05 A; 0.7% at 0.10 A | Incomplete |
| PM6, 20 cases at 0.01 A | 90% strict | Compatibility smoke |
| AM1, 20 cases at 0.01 A | 80% strict | Compatibility smoke |
| GFN1-xTB | 30-case, 10,000-step job timed out at 2.5 h | No optimizer conclusion |
| GFN2-xTB | Out of memory after approximately 22 min | No optimizer conclusion |

These probes do not add a matched GAD/Sella conclusion.
At HIP-labelled TS coordinates, xTB forces and indices are already very
large, so a fair xTB comparison requires xTB-native endpoints and paths.

---

## 5. Exact LJ Exposes a Different Pure-GAD Failure

LJ7 provides an exact analytic surface with cheap full Hessians. It was also
the most extensive implementation smoke:

- the full LJ7 Hessian has shape $21\times21$;
- finite-difference Hessian errors are approximately
  $6.3\times10^{-4}$ maximum and $6.6\times10^{-5}$ RMS;
- batched GAD and Sella/ASE agree on energy, force, Hessian, and the checked
  Eckart vibrational spectrum to numerical precision;
- the GAD recurrence agrees across independent implementations;
- replacing the approximate LJ7 reference with a force-balanced D5h
  structure does not change the high-noise pure-GAD conclusion.

There is no remaining evidence that Sella and GAD see different LJ
potentials or that the pure-GAD recurrence is malformed.

### 5.1 Pure GAD versus Sella

Using hydrogen masses and the strict reduced-unit criterion:

| Noise (fraction of $\sigma$) | Pure GAD | Sella |
|---:|---:|---:|
| 0.10 | 69.7% | **95.5%** |
| 0.15 | 51.2% | **83.6%** |
| 0.20 | 36.2% | **74.9%** |

Unlike HIP, exact LJ strongly favors Sella over pure GAD.

At 0.20 noise, the median initial projected index is eight and median initial
reduced $F_{\max}$ is $2.045\times10^3$. Independent coordinate noise gives
an expected per-atom displacement norm of $0.346\sigma$. In a 100,000-start
check, 54.93% of starts contain a pair closer than $0.75\sigma$, and 20.16%
contain a pair closer than $0.60\sigma$. Because the repulsive force scales
approximately as $r^{-13}$, this creates extreme force and curvature tails.

Single-mode GAD flips only $v_1$. At an index-$k>1$ point, the other negative
modes remain locally unstable. The LJ failure is therefore a direct
globalization problem: pure one-mode GAD is asked to enter an index-one basin
from a stiff, commonly high-index region.

### 5.2 Why timestep and displacement caps were insufficient

The major LJ interventions were:

| Intervention | High-noise result | Interpretation |
|---|---|---|
| Best hydrogen fixed cap | 51.2%/38.3% at 0.15/0.20 | Modest improvement |
| Ramped cap | 50.5%/39.7% | No decisive gain |
| Smaller 0.001 cap | approximately 34.8% at 0.20 | Safer motion but too slow |
| `dt=0.002-0.007` with active cap | approximately 49.8% at 0.15 and 36.6-38.0% at 0.20 | Smaller `dt` does not rescue failures |
| Hydrogen/carbon/argon assignment | approximately 38/28/10% in a comparable 0.20 screen | Hydrogen is best; mass is not causal |

Successful capped traces have a median 217 cap hits, compared with 1021 for
failures. Twenty-four failures remain capped for all 8000 steps. A fixed cap
limits step magnitude but does not decide whether lowest-mode ascent is
appropriate. Making every capped step smaller can therefore slow the same
wrong high-index dynamics rather than fix it.

### 5.3 The smooth $\lambda_2$ gate

The index-aware extension uses

$$
w=\operatorname{sigmoid}(k\lambda_2),
\qquad
F_{\mathrm{step}}
=F-2w(F^\mathsf{T}v_1)v_1.
$$

When the second vibrational eigenvalue is negative, $w$ is near zero and the
method behaves as projected force descent. As $\lambda_2$ becomes positive,
$w$ approaches one and the method becomes ordinary single-mode GAD. A hard
variant simply performs descent while $n_{\mathrm{neg}}>1$ and switches to
GAD at index one.

| Method | 0.10 | 0.15 | 0.20 |
|---|---:|---:|---:|
| Pure GAD | 69.7% | 51.2% | 36.2% |
| Hard descent to index one, then GAD | 99.0% | 97.2% | 96.5% |
| Smooth $\lambda_2$ gate | **100.0%** | **100.0%** | **99.7%** |

The gate is a compelling diagnosis and a useful algorithmic extension, but
it is **not pure GAD**. It also changes which saddles are reached. At 0.20
noise, only 33 of 104 pure-GAD successes finish at the same energy, within
$10^{-4}$, as their gated counterparts. LJ has no completed saddle-family or
endpoint classifier, so 99.7% strict convergence must not be reported as
99.7% intended chemical recovery.

LJ therefore explains one way that other surfaces differ from HIP. On LJ,
the dominant failure occurs before subtle learned-model interface effects
matter: the start is too stiff and too high-index for unglobalized one-mode
GAD. HIP trajectories appear to enter or remain in an index-one-relevant
region more readily, allowing the low-mode method to retain an advantage.

---

## 6. Other MLIP Controls Have Not Replicated the HIP Advantage

### 6.1 Conservative PaiNN/NeuralNEB

PaiNN is the cleanest current conservative T1x-trained control because force
and Hessian are obtained from one learned energy. A small PES-native
construction produced two two-sided IRC-accepted saddles.

The frozen pilot used those two saddles, two noise levels, three seeds per
saddle, 300 steps, plain GAD without a cap or gate, and Sella with a fresh
exact full Hessian at every step:

| Pilot cell | Pure GAD strict | Sella strict |
|---|---:|---:|
| Lower-noise cell | 0/6 | **4/6** |
| Higher-noise cell | 0/6 | **1/6** |
| Total | 0/12 | **5/12** |

A prespecified GAD timestep grid from 0.00025 to 0.005 gives 0/12 strict
successes at every value. However, all five strict Sella terminals fail the
predeclared intended native connectivity, so intended chemical success is
0/12 for both methods.

This is negative evidence for transfer of the HIP result, but it is only a
12-run pilot on two saddles. It cannot establish broad Sella superiority or
separate optimizer behavior from an inadequate native saddle set.

### 6.2 HORM LEFTNet

The HORM adapter passes directional derivative smokes for matched
energy-force and energy-force-Hessian LEFTNet checkpoints. The initial
optimizer screen uses four labelled T1x structures that were not established
as HORM-native saddles:

| Method | Strict result |
|---|---:|
| Stabilized/index-aware GAD | 0/4; terminal indices 5, 3, 2, 2 |
| Sella Cartesian+Eckart | 1/4; two index 2, one strict, and one index 1 with $F_{\max}=0.0547$ |

Pure GAD, hard gating, and smooth gating also fail on the detailed trace that
was tested. Native-set construction examines 15 endpoint pairs but does not
produce a sufficiently large two-sided IRC-validated set. The HORM result is
therefore inconclusive. It is not an independent positive replication, but
it is also not evidence that GAD generally fails on HORM.

This remains the most valuable future mechanism experiment because matched
E-F and E-F-H checkpoints could test whether Hessian supervision changes
low-mode persistence, full Taylor compatibility, and optimizer ranking on
the same native saddles.

### 6.3 MACE-OFF23

MACE-OFF23 is out of domain for this benchmark. Only 10 of 20 labelled T1x
transition structures are index one on its surface. Small probes initially
favor Sella and one retuned GAD case converges, but no common-pool matched
comparison exists. MACE is excluded from quantitative optimizer claims.

### 6.4 Cross-surface result table

| Surface | Best completed optimizer result | Chemical validation | Defensible conclusion |
|---|---|---|---|
| HIP/T1x | Sella leads at 0.01-0.05 A; tie at 0.10 A; GAD leads at 0.15-0.20 A | Intended IRC retains high-noise GAD lead | Real HIP-specific crossover |
| DFTB0 | GAD +2.1 pp at 0.01 A; Sella leads thereafter | Native topology favors Sella 55 vs 42 at 0.01 A | No broad GAD advantage |
| Exact LJ7 | Sella strongly beats pure GAD; index-aware gate is nearly perfect | No saddle-family classifier | Pure GAD lacks high-index globalization |
| PaiNN | GAD 0/12, Sella 5/12 strict | Both 0/12 intended | Small negative pilot |
| HORM LEFTNet | GAD 0/4, Sella 1/4 on non-native starts | Native set incomplete | Inconclusive |
| MACE-OFF23 | Exploratory probes only | T1x labels often nonstationary | Excluded |

---

## 7. What May Be Special About HIP

The cross-surface results suggest that two separate axes control the ranking.

### 7.1 Entry-region globalization

Exact LJ demonstrates the first axis. Plain single-mode GAD is vulnerable
when isotropic noise creates a stiff region with several negative modes.
Sella has an explicit restricted-step globalization mechanism, while plain
GAD does not. The $\lambda_2$ gate repairs this failure by withholding ascent
until the second mode stabilizes.

This mechanism explains why GAD can fail on LJ without implying that the
implementation is wrong. It also shows that HIP's advantage cannot arise
merely from GAD being intrinsically more robust from arbitrary starts.

### 7.2 The optimizer-interface contract

HIP exposes three learned products:

$$
\widehat E(x),\qquad
\widehat F(x),\qquad
\widehat H(x).
$$

The Hessian is directly supervised by DFT Hessians and its loss emphasizes
the lowest reference subspace. This is potentially a useful feature: the
model may retain an informative lowest-mode projector even when the full
energy-force-Hessian tuple is not an exactly integrable Taylor jet.

Plain GAD asks primarily for the current force and $v_1v_1^\mathsf{T}$.
Sella constructs a full quadratic model and adapts its trust radius using
agreement with the learned energy. If the learned products disagree over a
finite step, reducing the trust radius need not restore the same model ratio
that would occur for derivatives of one scalar potential.

The local consistency measurements support the prerequisite for this
hypothesis:

| Directional finite-difference diagnostic | HIP | Conservative PaiNN |
|---|---:|---:|
| $\lvert F\cdot v+dE/ds\rvert$, eV/A | 0.32244 | $5.61\times10^{-4}$ |
| $\|Hv+dF/ds\|/\|dF/ds\|$ | 0.07631 | $2.34\times10^{-4}$ |
| $\lvert v^\mathsf{T}Hv-d^2E/ds^2\rvert$, eV/A$^2$ | 27.738 | $2.23\times10^{-2}$ |

These values show that HIP's exposed energy, direct force, and direct Hessian
are observably distinct at the tested finite-difference scale. They do **not**
show that HIP's direct Hessian is inaccurate relative to DFT, and they should
not be framed as a defect. A directly supervised low-mode Hessian can be
scientifically useful without being the exact Jacobian of a separately
learned force.

The measurements also do not yet show that product incompatibility causes
Sella's failures. The panel is not conditioned on GAD-only, Sella-only, both,
and neither outcomes, and no stable-block replacement or trust-policy rescue
has been completed.

### 7.3 Why the controls are consistent with this explanation

- **DFTB0:** energy, gradient, and Hessian come from one electronic-structure
  approximation. Sella can use a coherent full local model, and it leads at
  all but the smallest perturbation.
- **PaiNN:** force and Hessian are derivatives of one learned energy. The
  small strict-stationarity pilot favors Sella, although the native chemistry
  benchmark is not yet successful.
- **LJ:** derivatives are exact and coherent, but the stronger fact is that
  the starts are often extremely high-index. Pure GAD fails at
  globalization; index-aware descent repairs it.
- **HORM:** current starts and sample size are insufficient to decide whether
  Hessian supervision alone changes the ranking.
- **MACE:** the T1x structures are not a valid native saddle pool, so the
  optimizer interface cannot be isolated.

This pattern is compatible with the hypothesis that HIP combines an unusually
useful low-mode field with a stricter challenge for full finite-step
energy-model optimization. It does not establish that this combination is
unique to HIP or that it is the sole cause of the observed crossover.

---

## 8. Remaining Caveats

1. **The positive result is high-noise only.** Sella clearly leads on HIP at
   0.01-0.05 A and uses far fewer successful steps.
2. **The headline threshold matters.** Plain HIP GAD is competitive at
   `fmax < 0.01` but has a sub-0.005 force floor under fixed-step dynamics.
3. **Strict convergence is not reaction identity.** HIP has intended IRC
   validation; LJ does not. PaiNN's strict Sella successes are chemically
   wrong under the predeclared native connectivity test.
4. **The HIP table contains per-cell selection.** Publication requires one
   validation-selected GAD and Sella configuration or an explicit oracle-grid
   label.
5. **Noise is not matched by physical difficulty.** Equal A noise creates
   different force, index, stiffness, and out-of-distribution profiles on
   each surface.
6. **The historical pm labels are wrong.** Use A for molecular surfaces and
   fractions of $\sigma$ for reduced LJ.
7. **The smooth gate is not pure GAD.** Its near-perfect LJ convergence is an
   algorithmic extension and lacks intended-saddle validation.
8. **The independent MLIP controls are underpowered.** PaiNN uses two native
   saddles; HORM lacks a completed native set; MACE is out of domain.
9. **The causal HIP mechanism is unproved.** Product inconsistency and
   Sella's trust-floor signature are correlated observations, not a completed
   intervention.
10. **Version and metric provenance matter.** Historical Sella results use
    Sella 2.3.4, and final success is recomputed from common projected
    `n_neg` and Cartesian-component `fmax`, not accepted from the optimizer's
    own status.

---

## 9. Conclusion

The current answer is not that GAD is broadly superior, nor that the other
surfaces are poor. It is that HIP creates a particular high-noise regime in
which a low-information optimizer contract appears more robust than a richer
full-quadratic contract.

Two control results sharpen that statement. LJ shows that pure one-mode GAD
can fail for an ordinary theoretical reason: it lacks globalization from
stiff, high-index starts. DFTB0 and conservative PaiNN show that when energy,
force, and Hessian are locally coherent, Sella can exploit full curvature and
plain GAD has no comparable advantage. HIP may be different because its
directly supervised low-mode Hessian remains useful to force-reflection
dynamics even where its separately learned products are less suitable for
repeated finite-step energy-model updates.

That is the strongest interpretation consistent with all completed surfaces.
It is a hypothesis supported by the cross-surface pattern, HIP/PaiNN
consistency diagnostics, and paired HIP trust-floor trajectories. It is not
yet a causal result.

The shortest decisive continuation would be:

1. compute outcome-conditioned Taylor residual orders along the actual HIP
   Sella proposals before departure;
2. test one predeclared intervention tied to the measured failure, such as
   reject-before-apply globalization or replacement of only the stable
   Hessian block while preserving HIP's $v_1$;
3. build a sufficiently large HORM-native saddle set and compare matched E-F
   and E-F-H checkpoints; and
4. freeze one GAD and one Sella configuration before evaluating a held-out
   reaction set.

Until those tests are completed, the publishable empirical result is a
surface-dependent optimizer crossover centered on HIP, with exact LJ
providing a clean high-index counterexample and the conservative controls
providing preliminary evidence that HIP's optimizer interface is unusual.

---

## Source Documents

- `WORKING_DRAFT_ALL_RESULTS_2026_07_20.md`
- `BENCHMARK_RESULTS_2026_07_16.md`
- `HIP_GAD_SELLA_SYNTHESIS_2026_07_17.md`
- `HIP_GAD_SELLA_THEORY_AUDIT_2026_07_17.md`
- `LJ_FINDINGS_2026_07_09.md`
- `SCINE_XTB_FINDINGS_2026_05_15.md`
