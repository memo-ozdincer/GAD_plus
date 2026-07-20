# When Does Low-Mode Ascent Outperform Full-Hessian Saddle Optimization?

## A results-heavy working draft on GAD, Sella, HIP, and control potential-energy surfaces

**Status:** Internal working manuscript, 2026-07-20.  
**Scope:** This draft consolidates the completed HIP, hybrid, Lennard-Jones,
SCINE, PaiNN, HORM, MACE, implementation-audit, and mechanism experiments in
this repository. It deliberately distinguishes measured results from
interpretation and proposed tests. It is not yet a submission-ready claim of
causality.

## Abstract

We compare a projected, single-mode gentlest-ascent-like dynamics (GAD) with
Sella's restricted-step partitioned rational-function optimizer (RS-P-RFO)
for molecular transition-state optimization using full Hessian information.
On 287 noised Transition1x test structures evaluated with the Hessian
Interatomic Potential (HIP), the optimizer ranking changes with perturbation
size. Sella gives higher strict stationary-point recovery near the labelled
transition state, while plain GAD becomes more robust at the largest
perturbations. At Cartesian noise standard deviations of 0.15 and 0.20 A,
the best completed plain-GAD rates are 58.2% and 44.6%, compared with 54.0%
and 27.2% for the best completed Sella configurations. Intended IRC-topology
recovery retains this high-noise advantage. The result is not universal:
Sella outperforms pure GAD on analytic Lennard-Jones clusters and on SCINE
DFTB0 at moderate and high perturbation, and a small HORM screen supplies no
independent positive replication.

Several implementation explanations were eliminated. GAD and Sella receive
the same LJ energy, force, and Hessian; the GAD recurrence agrees across
implementations; masses are used to construct the Eckart/vibrational problem
and then removed before applying a Cartesian displacement; and the headline
Sella benchmark receives the current full HIP Hessian at every optimization
step. At 0.15 A on HIP, paired outcomes comprise 133 joint successes, 34
GAD-only successes, 22 Sella-only successes, and 98 joint failures. Initial
index, force, eigengap, first-step size, and consecutive lowest-mode overlap
do not distinguish the two exclusive-success groups. In GAD-only cases,
however, Sella moves farther from the labelled saddle and spends a median
97.9% of stored iterations at its minimum trust radius.

We develop a candidate explanation based on optimizer-interface
compatibility. HIP independently exposes a learned energy, a direct force,
and a directly DFT-supervised Hessian. GAD uses only the force and the
rank-one projector defined by the lowest mode. Sella constructs a full
quadratic model from the direct force and Hessian and evaluates it against
the learned energy. If these products do not form one local Taylor jet,
shrinking an energy-ratio trust region need not restore model agreement. A
directional diagnostic confirms substantially larger energy-force-Hessian
incompatibility for HIP than for conservative PaiNN, but the diagnostic has
not yet been conditioned on optimizer outcome and no causal rescue
intervention has been completed. The present evidence therefore establishes
a surface-dependent optimizer crossover and a credible mechanism hypothesis,
not the reason for that crossover.

---

## 1. Research question and current answer

The original research question was whether GAD is a more robust molecular
transition-state optimizer than Sella when both consume full Hessian
information. The evidence rejects that universal claim. The narrower
questions are:

1. Why does plain GAD outperform Sella on the HIP surface at high noise, even
   though Sella wins near the labelled saddle?
2. Why does this crossover not appear on analytic LJ, SCINE DFTB0, or the
   current conservative-MLIP controls?
3. Does HIP preserve a chemically useful low-dimensional reaction direction
   farther from the path than it preserves a reliable full finite-step
   energy/force/Hessian model?
4. Can that distinction be measured before optimization fails and converted
   into a causal optimizer intervention?

The strongest current answer is conditional:

> HIP may preserve a useful lowest-mode projector beyond the region where its
> separately learned energy, direct force, and direct Hessian form a reliable
> finite-step Taylor model. GAD asks only for the force and that projector.
> Sella asks for a coherent full quadratic model and uses a learned-energy
> actual/predicted ratio to adapt its trust radius. This difference is
> consistent with the HIP crossover and Sella's observed trust-floor
> signature, but it has not yet been shown to cause them.

### Manuscript-readiness assessment

- We are confident that the HIP crossover itself is real.
- We are not yet confident that we know its causal mechanism.
- A complete working manuscript can be written now.
- A submission claiming to explain *why* GAD beats Sella on HIP is premature.
- The present mechanism position is approximately `4/10` as a paper.
- Outcome-conditioned Taylor diagnostics plus a successful tied intervention
  could raise it to roughly `6/10`.
- A matched HIP loss ablation or independent reactive-MLIP replication could
  raise a clean causal result toward `7-8/10`.

---

## 2. Background and theory

### 2.1 Projected single-mode GAD

For a conservative potential \(E(x)\), force \(F(x)=-\nabla E(x)\), and a
normalized lowest Hessian eigenvector \(v_1\), the position update used by
the repository's plain projected method is proportional to

\[
F_{\mathrm{GAD}} =
F - 2v_1(v_1^\mathsf{T}F)
= (I-2v_1v_1^\mathsf{T})F.
\]

The Hessian enters the position update through the rank-one projector
\(v_1v_1^\mathsf{T}\). Once \(v_1\) is fixed, the update does not use the
magnitude of \(\lambda_1\), any stable-subspace eigenvalue, or a predicted
energy change.

Near an index-one saddle \(x^\star\), write \(e=x-x^\star\) and diagonalize
the Hessian with \(\lambda_1<0<\lambda_2\le\cdots\). The linearized
continuous position dynamics contract as

\[
\dot e_i = -|\lambda_i|e_i.
\]

For explicit Euler,

\[
e_{k+1,i} = (1-\Delta t|\lambda_i|)e_{k,i}.
\]

Thus a fixed timestep must be small enough for the stiffest mode while the
softest stable mode controls slow contraction. This is the proper local
explanation for timestep sensitivity; a Newton step does not literally grow
as the force shrinks.

At an index-\(k>1\) geometry, flipping only \(v_1\) leaves the other negative
modes unflipped. Their linearized components remain unstable. GAD theory
therefore supplies a local index-one fixed-point guarantee, not a
globalization guarantee from stiff, high-index molecular starts
([E and Zhou, 2011](https://doi.org/10.1088/0951-7715/24/6/008)).

Strictly, non-gradient GAD theory uses the Jacobian and, in general, left and
right eigenvectors. Because HIP supplies a symmetric learned Hessian that is
not necessarily the Jacobian of its direct force, the repository method is
most precisely described as **learned-mode force-reflection dynamics**. We
retain "GAD" as the project name but do not claim all non-gradient GAD
theorems apply unchanged.

### 2.2 Sella and the stronger Taylor-model contract

Sella uses constrained RS-P-RFO and a restricted step
([Hermes et al., 2022](https://doi.org/10.1021/acs.jctc.2c00395)). In this
benchmark it receives the current full HIP Hessian at every optimization
step. It consumes curvature in the ascending and stable subspaces, not only
the lowest eigenvector.

For HIP's learned products,

\[
\widehat E(x),\qquad
\widehat F(x),\qquad
\widehat H(x),\qquad
g_F(x)=-\widehat F(x),
\]

the local quadratic energy model is

\[
m_x(s)=\widehat E(x)+g_F^\mathsf{T}s+
\frac{1}{2}s^\mathsf{T}\widehat Hs.
\]

Let \(g_E=\nabla\widehat E\) and
\(H_E=\nabla^2\widehat E\). Then

\[
\widehat E(x+s)-m_x(s)
=
(g_E-g_F)^\mathsf{T}s
+\frac{1}{2}s^\mathsf{T}(H_E-\widehat H)s
+O(\lVert s\rVert^3).
\]

This identity separates first-order energy/direct-force disagreement from
second-order Hessian disagreement.

If Sella evaluates a proposed direction \(s=\delta d\) with the
actual/predicted energy ratio

\[
\rho(\delta d)=
\frac{\widehat E(x+\delta d)-\widehat E(x)}
{\delta g_F^\mathsf{T}d+
\frac{1}{2}\delta^2d^\mathsf{T}\widehat Hd},
\]

then, generically,

\[
\lim_{\delta\rightarrow 0}\rho(\delta d)
=\frac{g_E^\mathsf{T}d}{g_F^\mathsf{T}d},
\]

not one, when \(g_E\neq g_F\). Shrinking the trust radius therefore need not
repair an energy-ratio model built from incompatible products. Sella 2.3.4
also applies the proposed step before adapting the next trust radius; it is
not a reject-and-roll-back line search in the benchmarked path.

The analogous force/Hessian compatibility residual is

\[
r_F(s)=
\left\|
\widehat F(x+s)-\widehat F(x)+\widehat H(x)s
\right\|.
\]

For a coherent smooth force/Hessian pair, \(r_F=O(\lVert s\rVert^2)\).
A persistent \(O(\lVert s\rVert)\) term means that the supplied Hessian is
not the local derivative of the force field followed by the optimizer.

These are optimizer-interface statements, not DFT-accuracy statements. A
directly supervised Hessian can be more accurate relative to DFT while being
less compatible with a separately learned force or energy.

### 2.3 Why HIP is a scientifically plausible special case

HIP directly predicts a symmetric Hessian supervised by DFT Hessian labels
([Burger et al., 2025](https://arxiv.org/abs/2509.21624)). Its training uses
approximately 1.73 million HORM/Transition1x geometries and a loss that
emphasizes the eight lowest reference modes, including the two lowest
non-rigid modes. The HIP ablations report modest improvements in the first
eigenvector and second eigenvalue from this subspace-aware objective, with
small tradeoffs in some full-matrix metrics.

Transition1x contains structures sampled on and around DFT NEB reaction
paths, not arbitrary isotropic perturbations
([Schreiner et al., 2022](https://arxiv.org/abs/2207.12858)). This gives HIP
unusually direct supervision in reaction regions and makes a privileged
low-mode field plausible.

The HIP paper also reports successful RFO from GSM-generated path-local
guesses. Therefore, neither "HIP Hessians break RFO" nor "Sella cannot use
HIP" is defensible. Any explanation must depend on the start distribution,
the optimizer interface, or both.

**Isolation tests still needed**

1. Compare \(v_1\) with a HIP-native path tangent, not only with the previous
   iteration's \(v_1\).
2. Measure the Taylor residual orders along actual Sella proposals before
   trust collapse.
3. Compare isotropic Cartesian noise with GSM/NEB-stage starts matched by
   force, index, spectral stiffness, and minimum pair distance.
4. Run matched HIP MAE-, MSE-, and low-subspace-loss checkpoints.

---

## 3. Computational protocol

### 3.1 Surfaces and model families

| Surface/model | Character of \(E/F/H\) | Role in this study |
|---|---|---|
| HIP on Transition1x | Learned energy, direct force, separately DFT-supervised direct Hessian | Primary surface showing the optimizer crossover |
| SCINE DFTB0/2/3 | Energy, gradient, and Hessian from one semiempirical electronic-structure method | Internally coherent non-ML control with different saddle locations |
| Analytic LJ7 | Exact analytic/reduced-unit energy, force, and Hessian | Cheap stiff/high-index control and implementation smoke |
| PaiNN/NeuralNEB | Conservative learned energy with autograd force/Hessian | Independent T1x-trained conservative MLIP control |
| HORM LEFTNet E-F/E-F-H | Energy-derived force and autograd Hessian | Matched Hessian-supervision control, pending a robust native set |
| HORM LEFTNet-df E-F-H | Direct force and force Jacobian | HIP-like nonconservative field diagnostic |
| MACE-OFF23 | Out-of-domain conservative molecular MLIP | Negative compatibility control |
| PM6, AM1, GFN1-xTB, GFN2-xTB | Semiempirical exploratory probes | Surface-compatibility and cost probes only |

### 3.2 Starting geometries and the noise-unit correction

The primary HIP set contains \(n=287\) Transition1x test-split labelled
transition structures. Independent Gaussian noise was added to each
Cartesian coordinate with standard deviations

\[
\sigma_x \in \{0.01,0.03,0.05,0.10,0.15,0.20\}\ {\rm A}.
\]

Historical scripts label these cells as 10, 30, 50, 100, 150, and 200 pm by
multiplying A by 1000. This is wrong by a factor of ten because
\(1\ {\rm A}=100\ {\rm pm}\). All manuscript tables use A and retain the
historical label only for artifact lookup.

Equal Cartesian noise does not imply equal optimization difficulty across
PESs. This is especially important for LJ, where 0.20 in reduced
\(\sigma\)-units creates near-collisions, and for DFTB0/MACE, where T1x
structures are not necessarily stationary on the target PES.

### 3.3 Coordinate, mass, and projection convention

Physical atomic masses are used to construct the mass-weighted
Eckart/vibrational Hessian and remove translational and rotational modes.
The resulting direction is then un-mass-weighted before applying the
Cartesian GAD displacement. The implementation that retained the mass
weighting errored and produced no benchmark data. Historical notes that
interpret completed results as still-mass-weighted dynamics are superseded.

For LJ7, all atoms are assigned hydrogen mass (\(m=1.008\)). Hydrogen,
carbon-like, and argon-like assignments alter the vibrational
eigendecomposition even though the final displacement is Cartesian.

### 3.4 Optimizers

**Plain GAD.** Projected single-mode force reflection with fixed timestep;
the primary HIP grid uses \(\Delta t=0.003,0.005,0.007\).

**Index-aware GAD extension.** For LJ, the smooth gate

\[
w=\operatorname{sigmoid}(k\lambda_2),\qquad
F_{\rm step}=F-2w(F^\mathsf{T}v_1)v_1
\]

behaves as descent while \(\lambda_2<0\) and approaches ordinary GAD once
the geometry becomes approximately index one. This is explicitly not pure
GAD.

**Hybrid GAD-Newton.** GAD walks until an index-one criterion is reached,
after which an eigenvector-following/Newton-like step refines the stationary
point. The headline hybrid is the damped Eckart eigenswitch configuration
with trust cap 0.05 A.

**Sella.** Cartesian or internal RS-P-RFO variants were evaluated. The
headline Cartesian-Eckart baseline receives a current full HIP Hessian every
step. Historical labels such as `d=1`, `d=3`, and `Hess.Freq.` are retained
only to identify artifacts; they must not be interpreted as evidence that
the learned HIP Hessian was stale between optimization steps. The old
interpretation of Sella's `gamma` as a line-search parameter is also wrong:
in Sella 2.3.4 it is an iterative eigensolver tolerance.

### 3.5 Outcome metrics

The common strict stationary-point criterion is

\[
n_{\rm neg}=1
\quad\land\quad
F_{\max}<0.01\ {\rm eV/A},
\]

where \(n_{\rm neg}\) is recomputed from the Eckart-projected vibrational
Hessian. LJ uses the analogous reduced force units.

Strict stationarity does not establish reaction identity. Where available,
we separately report:

- two-sided IRC endpoint topology (`IRC_TOPO`);
- PES-native relaxed endpoint connectivity;
- RMSD to the labelled transition structure;
- five-tier outcomes: intended, one-sided/partial, unintended, and failed;
- wall time and steps among strict successes.

The HIP tables include both fixed-configuration and post-hoc per-cell-best
summaries. Per-cell best values describe the completed grid but are not a
frozen prospective benchmark.

---

## 4. Results I: HIP/Transition1x optimizer crossover

### 4.1 Strict stationary-point recovery

Table 1 gives the strongest completed plain-GAD and Sella result in each
noise cell. Sella leads by 6.3-7.3 percentage points at 0.01-0.05 A, the
methods tie at 0.10 A, and plain GAD leads by 4.2 and 17.4 points at 0.15 and
0.20 A.

**Table 1. Per-cell best strict success on HIP (\(n=287\)).**

| Cartesian noise SD (A) | Historical label | Best plain GAD | Best Sella | GAD - Sella |
|---:|---:|---:|---:|---:|
| 0.01 | 10 pm | 89.2% | **96.5%** | -7.3 pp |
| 0.03 | 30 pm | 88.9% | **95.5%** | -6.6 pp |
| 0.05 | 50 pm | 85.7% | **92.0%** | -6.3 pp |
| 0.10 | 100 pm | 72.8% | 72.8% | 0.0 pp |
| 0.15 | 150 pm | **58.2%** | 54.0% | +4.2 pp |
| 0.20 | 200 pm | **44.6%** | 27.2% | +17.4 pp |

The full fixed-configuration grid shows that the crossover is not caused by
one isolated GAD timestep.

**Table 2. HIP strict success across principal configurations (\(n=287\)).**

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| GAD, dt=0.003 | 89.2 | 88.5 | 85.4 | 71.1 | 55.1 | 40.8 |
| GAD, dt=0.005 | 89.2 | 88.5 | 85.7 | 71.8 | 57.1 | 43.2 |
| GAD, dt=0.007 | 89.2 | 88.9 | 85.7 | **72.8** | **58.2** | **44.6** |
| Sella Cartesian, tuned label | 92.0 | 91.3 | 87.5 | 65.5 | 42.9 | 18.8 |
| Sella Cartesian+Eckart, config A | 92.7 | 92.0 | 88.2 | 70.7 | 54.0 | 27.2 |
| Sella internal | 79.1 | 77.4 | 71.8 | 50.9 | 26.8 | 13.9 |
| Sella Cartesian+Eckart, historical config B | **96.5** | **95.5** | **92.0** | **72.8** | 50.5 | 23.3 |
| Hybrid damped Eckart | 85.4 | 85.0 | 81.5 | 66.9 | 50.9 | 33.1 |
| Hybrid undamped Eckart | 84.7 | 84.3 | 81.5 | 65.5 | 49.8 | 31.0 |

**What this establishes.** The defensible claim is a noise-dependent HIP
crossover, not general GAD superiority and not general Sella failure.

**Isolation tests still needed**

1. Freeze one GAD and one Sella configuration on a validation split, then
   repeat the six-cell test without per-cell selection.
2. Report paired bootstrap confidence intervals and McNemar tests.
3. Audit reaction-level overlap between the 287 test reactions and HIP/HORM
   training data.
4. Repeat with path-generated guesses matched to the HIP paper's GSM-to-RFO
   setting.

### 4.2 Intended IRC/topology recovery

The high-noise advantage survives intended-reaction validation. Table 3 uses
fixed principal configurations rather than selecting each IRC row
independently.

**Table 3. HIP intended IRC_TOPO success (\(n=287\)).**

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| Plain GAD, dt=0.005 | 88.9 | **89.2** | **88.9** | **78.0** | **61.7** | **44.6** |
| Sella Cartesian+Eckart, config A | **89.2** | **89.2** | 87.5 | 72.5 | 49.8 | 23.3 |
| Hybrid damped Eckart | **89.2** | 88.9 | **88.9** | 76.7 | 57.5 | 38.7 |

At 0.10, 0.15, and 0.20 A, the best completed plain-GAD IRC rates are
78.4%, 61.7%, and 44.6%, versus 72.5%, 49.8%, and 23.3% for the matched
Sella baseline. The high-noise crossover is therefore not merely a
stationary-point stopping semantic.

A later five-tier rerun corrected an earlier manuscript interpretation that
Sella frequently found an unintended saddle. It did not.

**Table 4. Five-tier IRC outcomes as counts out of 287. Entries are
intended / partial / unintended / TS error.**

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---|---|---|---|---|---|
| GAD | 253/33/0/1 | 255/30/0/2 | 250/35/0/2 | 220/49/0/18 | 176/74/0/37 | 130/89/0/68 |
| Hybrid | 256/30/0/1 | 256/29/0/2 | 255/28/0/4 | 221/46/0/20 | 163/73/0/51 | 110/91/0/86 |
| Sella | 254/33/0/0 | 256/30/0/1 | 250/32/0/5 | 209/54/0/24 | 143/88/2/54 | 68/101/2/116 |

Only four of 5166 method/noise/sample outcomes were classified as
unintended. At 0.20 A, Sella's deficit is dominated by failure to converge
and one-sided IRC outcomes, not confident convergence to a different saddle.
Any old draft language about Newton/RFO "jumping to wrong saddles" must be
deleted.

**Isolation tests still needed**

1. Diagnose why one IRC branch fails in partial cases: shallow mode,
   endpoint coalescence, numerical IRC failure, or true reaction branching.
2. Match candidate saddles by destination basin before comparing optimizer
   dynamics.
3. Repeat with an independent IRC implementation on a stratified subset.
4. Report intended success both ungated and conditional on strict
   stationarity.

### 4.3 Paired 0.15 A outcome analysis

The matched 0.15 A cell provides the cleanest descriptive mechanism panel.

**Table 5. Paired HIP outcomes at 0.15 A.**

| Outcome class | Count | Fraction |
|---|---:|---:|
| Both succeed | 133 | 46.3% |
| GAD only | 34 | 11.8% |
| Sella only | 22 | 7.7% |
| Neither | 98 | 34.1% |

An exact paired McNemar/binomial test on the 56 discordant strict outcomes
gives \(p=0.141\). The 0.15 A strict-rate difference is therefore
descriptive rather than statistically decisive by itself; the larger 0.20 A
effect and intended-IRC trend motivate the mechanism analysis.

Initial \(n_{\rm neg}\), \(F_{\max}\), eigengap, first GAD step, time to
index one, and consecutive \(v_1\) overlap do not cleanly distinguish
GAD-only from Sella-only starts. Recomputing a common maximum-atom step from
stored coordinates gives a median first step of 0.088 A for GAD and 0.100 A
for Sella in both exclusive-success groups. The simple explanation
"Sella takes one huge step while GAD takes tiny steps" is unsupported.

The later trajectory signature is different:

| Sella trajectory statistic | GAD-only starts | Sella-only starts |
|---|---:|---:|
| Median minimum distance to labelled TS (A) | 0.226 | 0.117 |
| Median final distance to labelled TS (A) | 0.447 | 0.128 |
| Median final trust radius (A) | \(10^{-4}\) | not the defining signature |
| Median fraction of stored rows at trust floor | 97.9% | substantially lower |

This establishes departure followed by trust collapse. It does not establish
whether incompatible Taylor products, basin choice, or trust-policy
hysteresis causes the departure.

**Isolation tests still needed**

1. Log proposed and applied steps, predicted and actual energy changes, and
   \(\rho\) before the first departure.
2. Evaluate Taylor residuals along the exact applied Sella directions.
3. Record distance and tangent overlap to a HIP-native path.
4. Test reject-before-apply globalization on the same 34 GAD-only cases.
5. Preserve HIP \(v_1\) while replacing only the stable Hessian block.

---

## 5. Results II: force thresholds, convergence floors, and hybrid refinement

### 5.1 Sensitivity to the force threshold

The headline criterion \(F_{\max}<0.01\) is strict relative to ASE's common
0.05 eV/A setting but looser than the 0.001 eV/A recommendation cited in
Sella documentation. Table 6 shows the full threshold behavior for one
representative method from each family. All entries also require
\(n_{\rm neg}=1\).

**Table 6. Strict success (%) across force thresholds.**

| Method | Noise (A) | <0.05 | <0.023 | <0.01 | <0.005 | <0.001 |
|---|---:|---:|---:|---:|---:|---:|
| GAD dt=0.007 | 0.01 | 98.3 | 95.1 | 89.2 | 0.0 | 0.0 |
|  | 0.03 | 97.9 | 95.5 | 88.9 | 0.0 | 0.0 |
|  | 0.05 | 94.8 | 91.6 | 85.7 | 0.0 | 0.0 |
|  | 0.10 | 82.6 | 77.7 | 72.8 | 0.0 | 0.0 |
|  | 0.15 | 71.4 | 64.1 | 58.2 | 0.0 | 0.0 |
|  | 0.20 | 68.3 | 56.4 | 44.6 | 0.0 | 0.0 |
| Hybrid damped Eckart | 0.01 | 97.2 | 93.0 | 85.4 | 7.3 | 0.0 |
|  | 0.03 | 96.9 | 92.3 | 85.0 | 9.4 | 0.0 |
|  | 0.05 | 93.0 | 88.5 | 81.5 | 10.8 | 0.0 |
|  | 0.10 | 78.0 | 73.5 | 66.9 | 9.1 | 0.0 |
|  | 0.15 | 59.6 | 55.7 | 50.9 | 7.0 | 0.0 |
|  | 0.20 | 44.9 | 37.6 | 33.1 | 6.6 | 0.0 |
| Sella Cartesian+Eckart A | 0.01 | 98.6 | 97.9 | 92.7 | 33.8 | 0.0 |
|  | 0.03 | 98.3 | 97.6 | 92.0 | 32.1 | 0.0 |
|  | 0.05 | 95.1 | 93.7 | 88.2 | 32.4 | 0.0 |
|  | 0.10 | 83.3 | 77.7 | 70.7 | 24.7 | 0.0 |
|  | 0.15 | 66.2 | 59.9 | 54.0 | 15.3 | 0.0 |
|  | 0.20 | 46.0 | 39.0 | 27.2 | 7.0 | 0.0 |

Plain fixed-timestep GAD has zero successes below 0.005 eV/A throughout
this grid. A 10,000-step run at 0.05 A confirmed that the result is not
removed by a fivefold budget increase:

| Threshold (eV/A) | GAD dt=0.005, 10,000 steps, \(n=287\) |
|---:|---:|
| 0.05 | 95.5% |
| 0.023 | 91.6% |
| 0.01 | 85.7% |
| 0.005 | 0.0% |
| 0.001 | 0.0% |

Partial long-budget controls reached 47.2% below 0.005 for the hybrid
(\(n=72\)) and 77.3% for Sella (\(n=66\)); those rates are biased by timeout
and are included only to show that Newton/RFO refinement can enter the
sub-0.005 regime.

The correct local interpretation is not that \(H^{-1}F\) grows as \(F\)
shrinks. Rather, a coherent Newton/mode-following step estimates the current
coordinate error and can contract it quadratically near a stationary point,
while explicit fixed-timestep GAD has a linearly conditioned recurrence and
can encounter a learned-force/mode noise floor.

**Isolation tests still needed**

1. Measure the terminal force floor against HIP force-label error and
   run-to-run GPU variation.
2. Use adaptive integration with local truncation-error control rather than
   only fixed timestep or displacement cap.
3. Repeat the threshold curve on a conservative PES with matched force units.
4. Separate numerical oscillation, model noise, and product inconsistency by
   local linear replay around the terminal orbit.

### 5.2 Newton-polish experiment

A spectral NR-GAD ping-pong experiment at 0.05 A (\(n=80\)) tested whether
explicit Newton refinement can cross the plain-GAD force floor.

| Variant | \(F_{\max}<0.05\) | <0.01 | <0.005 | <0.001 |
|---|---:|---:|---:|---:|
| Loose NR-GAD, stops at 0.01 | 70.0% | 46.2% | 0.0% | 0.0% |
| Strict NR-GAD, target \(10^{-4}\) | 73.8% | 45.0% | **31.2%** | **1.2%** |
| Plain GAD dt=0.007 reference | 94.8% | 85.7% | 0.0% | 0.0% |

Newton polishing can break the force floor, but it trades roughly 40 points
of 0.01-level recovery for 31 points of 0.005-level recovery. The 1.2% value
at 0.001 is one trajectory out of 80 and is not a precise rate.

**Isolation tests still needed**

1. Trigger refinement using a validated mode-overlap criterion, not only
   \(n_{\rm neg}=1\).
2. Select the trust cap prospectively and report intended IRC outcomes.
3. Compare the same terminal refinement applied after both GAD and Sella.

### 5.3 Wall time and iteration counts

Measured wall time depends on the allocation and implementation, but the
completed campaign consistently found the hybrid fastest per strict success.

**Table 7. Wall seconds per strict success / median successful steps.**

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| GAD dt=0.005 | 47.7 / 100 | 57.1 / 204 | 74.3 / 278 | 156.0 / 458 | 278.5 / 614 | 472.3 / 738 |
| Sella Cartesian+Eckart A | 14.5 / 4 | 15.8 / 6 | 23.2 / 7 | 65.2 / 9 | 132.6 / 11 | 393.8 / 13 |
| Hybrid damped Eckart | **11.0 / 6** | **12.0 / 12** | **15.7 / 19** | **34.9 / 37** | **65.0 / 61** | **130.7 / 95** |

Sella uses the fewest steps, while the hybrid gives the lowest measured
wall cost per strict success. These timings are implementation-specific and
must be reprofilied before publication on a frozen hardware allocation.

### 5.4 Geometry relative to the labelled HIP/T1x saddle

Among strict successes, the hybrid produces the tightest high-noise
distribution relative to the labelled T1x geometry.

**Table 8. Kabsch/Hungarian RMSD to labelled TS among strict successes,
median / p95 in A.**

| Method | 0.01 A | 0.03 A | 0.05 A | 0.10 A | 0.15 A | 0.20 A |
|---|---:|---:|---:|---:|---:|---:|
| Sella Cartesian+Eckart A | .008/.073 | .009/.071 | .009/.072 | .009/.201 | .013/.617 | .017/.838 |
| Plain GAD dt=0.005 | .005/.018 | .008/.021 | .011/.028 | .014/.044 | .016/.088 | .014/.456 |
| Hybrid damped Eckart | .007/.047 | .007/.047 | .007/.049 | **.008/.055** | **.007/.062** | **.008/.109** |

At 0.20 A, the hybrid p95 is 7.7 times smaller than Sella's and 4.2 times
smaller than GAD's. This does not by itself establish better chemistry:
HIP and T1x share training provenance, and RMSD is not endpoint identity.

### 5.5 Starting-condition dependence

The hybrid is not a general basin finder from reactant minima.

| Starting condition and method | Result |
|---|---:|
| Hybrid damped from reactant, 2000 steps | 6/287 (2.1%) |
| Hybrid undamped from reactant, 2000 steps | 6/287 (2.1%) |
| Hybrid 10,000-step reactant probe | 0/72 partial |
| Sella Cartesian+Eckart from reactant | 80.8% |
| Sella from midpoint, zero added noise | 46.7% strict and 46.7% IRC_TOPO |

The partial plain-GAD reactant runs reported 40.1% (\(n=152\)) for
dt=0.003 and 54.2% (\(n=179\)) for dt=0.005, but timeout ordering biases
those rates toward early/smaller systems. The hybrid result supports a
narrow role: it accelerates refinement after GAD has entered a saddle
neighborhood; it does not reliably create that neighborhood from a minimum.

**Isolation tests still needed**

1. Compare noised TS, NEB maximum, GSM node, midpoint, and reactant starts at
   matched local index and force.
2. Test a proven basin-finding front end such as NEB/dimer followed by the
   same frozen refinement.
3. Reprofile all methods with batched model calls and identical GPU
   accounting.

---

## 6. Results III: SCINE semiempirical surfaces

### 6.1 DFTB0 strict stationary-point recovery

DFTB0 does not reproduce the HIP high-noise GAD advantage. GAD has a small
2.1-point lead at 0.01 A, after which Sella leads in every cell.

**Table 9. SCINE DFTB0 strict success (\(n=287\)).**

| Noise SD (A) | GAD | Sella | GAD - Sella |
|---:|---:|---:|---:|
| 0.01 | **89.9%** (258/287) | 87.8% (252/287) | +2.1 pp |
| 0.03 | 66.6% | **73.5%** | -6.9 pp |
| 0.05 | 43.9% | **56.1%** | -12.2 pp |
| 0.10 | 19.5% | **32.4%** | -12.9 pp |
| 0.15 | 6.3% | **15.3%** | -9.0 pp |
| 0.20 | 1.4% | **3.8%** | -2.4 pp |

Increasing the GAD budget did not change the ranking sufficiently to support
a robustness claim. DFTB0 energy, gradient, and Hessian arise from the same
electronic-structure approximation, so Sella receives an internally coherent
local model. This is consistent with the Taylor-compatibility hypothesis,
but equal Cartesian noise creates different local difficulty than on HIP.

### 6.2 DFTB0-native endpoint validation

The first DFTB0 IRC analysis was invalid because the installed ASE generator
could accept the saddle at IRC step zero. The corrected validator forces the
initial displacement, uses saddle-aware endpoint convergence, and scores both
methods against shared DFTB0-relaxed endpoints.

**Table 10. Corrected DFTB0-native outcomes at 0.01 A.**

| Metric | GAD | Sella |
|---|---:|---:|
| Strict search success | **258/287 (89.9%)** | 252/287 (87.8%) |
| Native topology correct | 42/287 (14.63%) | **55/287 (19.16%)** |
| Native strict geometry correct | 21/287 (7.32%) | 21/287 (7.32%) |
| Native reference evaluable among strict candidates | 256/258 | 250/252 |

On 229 starts where both searches converged, 228 native endpoint pairs were
evaluable. Topology favored Sella 51 to 41; strict geometry was effectively
tied, 20 to 21. DFTB0 therefore does not support GAD superiority even when
the misleading T1x endpoint semantics are removed.

### 6.3 PES disagreement with Transition1x labels

On 231 structures where both HIP and DFTB0 searches converged:

| Geometry pair | Median RMSD (A) | IQR | p95 |
|---|---:|---:|---:|
| HIP TS vs T1x TS | **0.005** | 0.003-0.006 | 0.018 |
| DFTB0 TS vs T1x TS | 0.444 | 0.241-0.656 | 1.019 |
| DFTB0 TS vs HIP TS | 0.444 | 0.243-0.659 | 1.021 |

For DFTB0 candidates that passed the original T1x topology criterion, the
median DFTB0-to-HIP TS RMSD was 0.157 A; failures had median 0.504 A. A bond
cutoff sweep from 1.10 to 1.50 rescued none of 20 inspected failures, and
post-IRC BFGS minimization rescued none. The low cross-PES topology rate is
therefore a PES-label disagreement, not primarily an IRC integration or bond
cutoff problem.

### 6.4 Other SCINE and xTB probes

| Surface/probe | Completed result | Evidentiary status |
|---|---|---|
| DFTB2 full GAD grid | Strict success 87.5, 47.7, 13.6, 0.7, 0.0, 0.0% over 0.01-0.20 A | Worse than DFTB0; no matched Sella headline |
| DFTB3 partial | 14.6% at 0.05 A and 0.7% at 0.10 A in the recorded partial grid | Incomplete |
| PM6 smoke, 20 cases at 0.01 A | 90% strict | Compatibility smoke only |
| AM1 smoke, 20 cases at 0.01 A | 80% strict | Compatibility smoke only |
| DFTB2 smoke, 20 cases | 75% strict | Superseded by full GAD grid |
| DFTB3 smoke, 20 cases | 70% strict | Exploratory |
| GFN1-xTB | 30-case, 10k-step job timed out at 2.5 h | No optimizer conclusion |
| GFN2-xTB | OOM after approximately 22 min | No optimizer conclusion |

At the HIP-labelled TS, xTB forces were 4-15 eV/A and indices were roughly
7-22. Those structures are not meaningful xTB saddle starts.

**Isolation tests still needed**

1. Build at least 50 DFTB0-native, two-sided endpoint-validated saddles.
2. Match perturbations by dimensionless local difficulty rather than A.
3. Compare Sella model ratios on DFTB0 and HIP at matched step norms.
4. For xTB, construct starts from xTB-relaxed endpoints and NEB paths rather
   than noised HIP/T1x saddles.

---

## 7. Results IV: analytic Lennard-Jones clusters

### 7.1 Implementation validation

The LJ potential is implemented locally and independently of HIP/SCINE.
For LJ7, the predictor returns a full \(21\times21\) Hessian.

| Smoke check | Result |
|---|---:|
| Hessian finite difference, maximum absolute error | \(6.3\times10^{-4}\) |
| Hessian finite difference, RMS error | \(6.6\times10^{-5}\) |
| Batched vs predictor force maximum difference | \(7\times10^{-15}\) |
| Batched vs predictor Hessian maximum difference | \(1.5\times10^{-11}\) |
| Sella/ASE vs batched energy difference | 0 |
| Sella/ASE vs batched force maximum difference | \(4.7\times10^{-10}\) |
| Sella/ASE vs batched raw-Hessian difference | \(3.0\times10^{-8}\) |
| Checked Eckart vibrational-spectrum difference | \(9\times10^{-8}\) |

The uniform-hydrogen branch recurrence smoke gave a maximum direction error
of \(7.299\times10^{-12}\) after the expected uniform-mass scaling and a
maximum coordinate difference of \(8.332\times10^{-14}\) A after five
steps. No evidence remains that GAD and Sella see different LJ surfaces or
that a broken GAD recurrence explains the comparison.

The original pentagonal-bipyramid helper was not force-balanced
(\(F_{\max}=1.95618\)). Replacing it with the relaxed D5h geometry reduced
\(F_{\max}\) to \(1.06\times10^{-7}\) and gave zero vibrational negative
modes. The 0.15-noise pure-GAD rate remained 51.2%, so the reference-geometry
fix did not explain the high-noise failure.

### 7.2 Pure GAD, Sella, and index-aware gating

**Table 11. LJ7 strict success (\(n=287\), hydrogen atom type).**

| Noise (fraction of \(\sigma\)) | Pure GAD | Sella | Smooth \(\lambda_2\) gate |
|---:|---:|---:|---:|
| 0.10 | 69.7% | **95.5%** | 100.0% |
| 0.15 | 51.2% | **83.6%** | 100.0% |
| 0.20 | 36.2% | **74.9%** | 99.7% |

The gate is not pure GAD, and LJ lacks an intended-saddle classifier.
At 0.20 noise, only 33 of 104 pure-GAD successes ended at the same energy
within \(10^{-4}\) as their gated counterpart. Near-perfect strict recovery
must not be presented as intended-saddle recovery.

### 7.3 Timestep, cap, mass assignment, and replay tests

**Table 12. LJ interventions and their observed effect.**

| Intervention | Completed result | Interpretation |
|---|---|---|
| Original fixed-cap tuning | Approximately 41.5/29.3% at 0.15/0.20 | Baseline |
| Best hydrogen fixed cap | 51.2/38.3% | Modest improvement |
| Ramped cap | 50.5/39.7% | No decisive gain over fixed cap |
| Smaller cap 0.001 | Approximately 34.8% at 0.20 | Safer but too slow |
| dt sweep 0.002-0.007 with active cap | Approximately 49.8% at 0.15 and 36.6-38.0% at 0.20 | Lower dt alone does not rescue failures |
| Hydrogen/carbon/argon atom assignment | Roughly 38/28/10% at 0.20 in the comparable capped screen | Hydrogen is best; mass is not the main failure |
| Hard descent until \(n_{\rm neg}\le1\) | 99.0/97.2/96.5% at 0.10/0.15/0.20 | High-index entry is decisive |
| Smooth \(\lambda_2\) gate | 100.0/100.0/99.7% | Robust globalization extension |

At 0.20 noise, median initial \(n_{\rm neg}=8\) and median initial reduced
\(F_{\max}=2.045\times10^3\). Independent Cartesian noise creates an
expected per-atom displacement norm of
\(\sqrt{3}(0.20)=0.346\sigma\). In a 100,000-start check, 54.93% of
structures had a closest pair below \(0.75\sigma\), and 20.16% below
\(0.60\sigma\). The repulsive force scales approximately as \(r^{-13}\),
producing an extreme force and curvature tail.

Successful fixed-cap traces had median 217 cap hits, compared with 1021 for
failures. Twenty-four failures remained capped for all 8000 steps; no
successful trajectory did. In a hard-failure replay panel, the median raw
first-step atom displacement before capping was approximately 213 in reduced
coordinate units, p90 approximately 2292. Smaller caps changed speed but did
not supply a criterion for when ascent was locally appropriate.

The LJ result has a comparatively strong diagnosis:

> Pure one-mode GAD is being asked to globalize from a stiff, commonly
> high-index region. A descent phase until the second mode becomes stable
> removes the failure. The problem is not a bad Hessian, hydrogen mass, or
> merely one oversized step.

**Isolation tests still needed**

1. Construct an LJ saddle-family/endpoint classifier before claiming
   intended recovery for the gate.
2. Match starts by index and harmonic displacement energy to HIP.
3. Compare smooth gating with a standard dimer/eigenvector-following
   globalization method.
4. Test adaptive acceptance based on actual LJ energy change rather than a
   fixed displacement cap.

---

## 8. Results V: conservative PaiNN/NeuralNEB control

### 8.1 Native saddle construction and field validation

Three of the first three endpoint pairs produced tight PaiNN candidates after
endpoint relaxation and climbing NEB, with NEB maximum-image
\(F_{\max}=0.00951,\ 0.00122,\ 0.0000207\) eV/A and projected
\(n_{\rm neg}=1\). Full-Hessian IRC validation retained two candidates with
two distinct relaxed endpoint topologies; a third was rejected because one
endpoint retained \(n_{\rm neg}=1\).

The conservative field-consistency panel behaved as expected:

| Directional diagnostic, median over 15 directions | PaiNN |
|---|---:|
| \(|F\cdot v+dE/ds|\) | \(5.61\times10^{-4}\) eV/A |
| \(\|Hv+dF/ds\|/\|dF/ds\|\) | \(2.34\times10^{-4}\) |
| \(|v^\mathsf{T}Hv-d^2E/ds^2|\) | \(2.23\times10^{-2}\) eV/A\(^2\) |
| Hessian antisymmetry | zero at printed precision |

### 8.2 Paired optimizer pilot

The frozen pilot used two IRC-accepted native saddles, two noise levels,
three seeds per candidate, 300 steps, pure GAD without cap/gate/blending,
and Sella with a fresh exact full Hessian after every step.

| Noise | Pure GAD strict | Sella strict |
|---:|---:|---:|
| Historical 50 pm pilot label | 0/6 | **4/6** |
| Historical 100 pm pilot label | 0/6 | **1/6** |
| Total | 0/12 | **5/12** |

All five strict Sella terminals reached two relaxed minima but failed the
predeclared intended native IRC connectivity. Chemical success was therefore
0/12 for both methods. A prespecified GAD timestep grid
\(\{0.00025,0.0005,0.001,0.002,0.005\}\) gave 0/12 strict successes at
every value.

This pilot is negative for the claim that pure GAD's HIP advantage transfers
to a conservative T1x-trained MLIP. It is also too small and chemically
unsuccessful to establish broad Sella superiority.

**Isolation tests still needed**

1. Build a larger PaiNN-native saddle set with unambiguous endpoints.
2. Diagnose why all strict Sella terminals connect different endpoints.
3. Match the noise scale to local curvature instead of reusing historical
   labels.
4. Use a prospectively fixed budget that allows both methods to reach their
   characteristic convergence regime.

---

## 9. Results VI: HORM Hessian-supervised models

### 9.1 Adapter and derivative smokes

Matched LEFTNet E-F and E-F-H checkpoints were checked with full autograd
Hessians. At a directional finite-difference step of 0.01 A, the relative
RMS force/Hessian error was 0.548% for `left_orig.ckpt` (E-F) and 1.29% for
`left.ckpt` (E-F-H). Errors increased at smaller epsilon, consistent with
float32 cancellation rather than an order-one adapter inconsistency.

For LEFTNet-df E-F-H, the direct-force Jacobian was operational but strongly
asymmetric: directional relative RMS error 0.0686 at 0.01 A and maximum
antisymmetric entry 6.54 eV/A\(^2\). It is a useful nonconservative contrast
but requires a declared symmetrization for vibrational optimization.

### 9.2 Optimizer screens and native-set status

The original four-structure zero-noise LEFTNet screen used labelled T1x
starts that were not established HORM-native saddles:

| Method | Strict result | Terminal details |
|---|---:|---|
| Stabilized GAD | 0/4 | Final indices 5, 3, 2, 2 |
| Sella Cartesian+Eckart | 1/4 | One strict; two index 2; one index 1 with \(F_{\max}=0.0547\) |

Pure GAD, smooth gating, and hard gating also failed on the detailed
sample-2 trace. Because the starts were not HORM-native, this is an
exploratory negative screen, not a model-family conclusion.

Native construction examined 15 endpoint pairs and produced one candidate
(sample 14) passing NEB, \(F_{\max}\le0.02\) eV/A, and projected index one.
Another shard candidate, sample 80, was locally converged and index one but
both full-Hessian IRC branches relaxed to the reactant-side minimum; it was
invalidated for an intended-saddle optimizer benchmark. No sufficiently
large, validated HORM-native set was completed, and no positive GAD result
was obtained.

**Isolation tests still needed**

1. Build at least 50 HORM-native, two-sided IRC-validated saddles.
2. Compare matched E-F and E-F-H checkpoints on identical native starts.
3. Include LEFTNet-df only with a predeclared symmetric curvature interface
   and raw asymmetry diagnostic.
4. Ask whether Hessian supervision changes low-mode persistence, full Taylor
   compatibility, and optimizer ranking together.

---

## 10. Results VII: MACE-OFF23 and other out-of-domain controls

MACE-OFF23 was screened as an out-of-domain molecular MLIP. All labelled
reactant/product geometries could be relaxed, but only 10 of 20 labelled
Transition1x TS geometries were index one on the MACE surface. Small
optimizer probes favored Sella under the initial settings, while one retuned
GAD case converged. There is no completed common-pool matched comparison.

The result is useful only as a warning: a labelled TS from one PES is not
automatically a valid starting saddle or reaction label on another PES.
MACE is excluded from quantitative claims.

**Isolation tests still needed**

1. Construct MACE-native paths and stationary points.
2. Restrict chemistry to MACE-OFF23's reliable domain.
3. Require identical endpoint labels and paired starts before optimizer
   comparison.

---

## 11. Cross-surface summary

**Table 13. Best defensible GAD-versus-Sella conclusion by surface.**

| Surface | Completed optimizer result | Chemical validation | Conclusion |
|---|---|---|---|
| HIP/T1x | Sella leads at low noise; GAD ties near 0.10 A and leads at 0.15-0.20 A | Intended IRC confirms high-noise GAD advantage | Real surface-dependent crossover |
| DFTB0 | GAD +2.1 pp at 0.01 A; Sella leads thereafter | Native topology favors Sella 55 vs 42 at 0.01 A | No broad GAD advantage |
| LJ7 | Sella strongly beats pure GAD; index gate nearly perfect | No saddle-family endpoint classifier | Pure GAD lacks high-index globalization |
| PaiNN | GAD 0/12, Sella 5/12 strict | Both 0/12 intended | Small negative conservative-MLIP pilot |
| HORM LEFTNet | GAD 0/4, Sella 1/4 on non-native screen | Native set not completed | Inconclusive, no positive replication |
| MACE-OFF23 | Exploratory probes initially favor Sella | T1x labels often nonstationary | Excluded from comparative evidence |

The surface comparison suggests two distinct axes:

1. **Globalization difficulty.** Exact LJ creates stiff, high-index starts
   that defeat pure one-mode GAD but are repaired by index-aware descent.
2. **Curvature-interface structure.** HIP provides separately supervised
   energy, force, and curvature channels, potentially allowing a useful
   lowest mode to coexist with weaker full Taylor-model agreement.

DFTB0 and PaiNN provide coherent local derivatives, where Sella can exploit
the full Hessian. This pattern is consistent with the candidate mechanism,
but the controls are not matched in start difficulty, chemistry, or sample
size and therefore do not prove it.

---

## 12. Direct tests of why HIP behaves differently

### 12.1 HIP energy-force-Hessian field consistency

A fixed directional finite-difference panel compared HIP with conservative
PaiNN.

**Table 14. Median local product-consistency diagnostics.**

| Diagnostic | HIP | PaiNN |
|---|---:|---:|
| \(|F\cdot v+dE/ds|\), eV/A | **0.32244** | \(5.61\times10^{-4}\) |
| \(\|Hv+dF/ds\|/\|dF/ds\|\) | **0.07631** | \(2.34\times10^{-4}\) |
| \(|v^\mathsf{T}Hv-d^2E/ds^2|\), eV/A\(^2\) | **27.738** | \(2.23\times10^{-2}\) |

HIP's energy, direct force, and direct Hessian are observably distinct at
this finite-difference scale. That is expected for separately supervised
heads and does not show that the direct Hessian is inaccurate relative to
DFT. It establishes the prerequisite for a Taylor-compatibility mechanism,
not its causal relevance to Sella.

### 12.2 Curvature-source implementation panel

On 12 fixed T1x test HDF5 geometries:

| Quantity | Result |
|---|---:|
| HIP predicted Hessian index one | 12/12 |
| Symmetric direct-force Jacobian index one | 0/12 |
| Learned-energy Hessian index one | 0/12 |
| Median predicted-Hessian vs force-Jacobian \(v_1\) overlap | 0.9903 |
| Median relative full-matrix disagreement | 0.4735 |
| Median predicted-Hessian vs energy-Hessian \(v_1\) overlap | 0.0116 |
| Median force-Jacobian relative antisymmetry | 0.2631 |

The predicted Hessian and force Jacobian can share a lowest direction while
disagreeing strongly in the full matrix and index. The force Jacobian was
never intended as a polished replacement for HIP's directly predicted
Hessian; these results are implementation/interface diagnostics, not a
headline scientific comparison.

A one-start, 80-step substitution smoke found:

| Curvature supplied with HIP direct E/F | GAD | Sella |
|---|---|---|
| HIP predicted Hessian | \(F_{\max}=0.0868\) after 80 steps | Strict in 12 steps, \(F_{\max}=0.00303,\ n_{\rm neg}=1\) |
| Symmetric force Jacobian | Failed high-index | Failed high-index |
| Energy Hessian | Failed high-index | Failed high-index |

The smoke proves curvature-source sensitivity, but it cannot compare GAD and
Sella because historical GAD received up to 2000 steps. Public HIP
evaluations also showed non-bitwise GPU force variation up to 0.00699 eV/A.
A later attempted long paired substitution was invalidated by split and
configuration mismatches. No causal source-substitution claim is retained.

### 12.3 Predeclared Taylor-compatibility test

For an observed Sella direction \(d\), normalized to one A maximum-atom
displacement, and probe \(s=\delta d\), the planned diagnostics are

\[
\begin{aligned}
r_{E1} &= |\Delta E-g_F^\mathsf{T}s|,\\
r_{E2} &= |\Delta E-g_F^\mathsf{T}s-\tfrac12s^\mathsf{T}Hs|,\\
r_{F2} &= \|\Delta F+Hs\|.
\end{aligned}
\]

A coherent smooth Taylor jet gives orders \(O(\delta^2)\),
\(O(\delta^3)\), and \(O(\delta^2)\), respectively.

The analytic LJ implementation smoke reproduced fitted orders:

| Residual | Fitted order across sampled LJ phases |
|---|---:|
| \(r_{E1}\) | 2.00-2.03 |
| \(r_{E2}\) | 2.87-3.05 |
| \(r_{F2}\) | 1.99-2.05 |

This validates the diagnostic numerics on an exact potential. All HIP and
SCINE jobs for the outcome-conditioned panel were cancelled before
scientific evaluation. There is **no HIP Taylor-panel result**.

**Decisive isolation test**

Run three predeclared 0.15 A cases from each of the four paired outcome
classes. Evaluate raw and applied-Eckart Hessians at initial, departure, and
trust-floor phases for probe lengths 0.08, 0.04, 0.02, 0.01, and 0.005 A.
Support the mechanism only if:

1. GAD-only/neither trajectories show worse residual scaling before outcome
   divergence than both/Sella-only trajectories;
2. the residual predicts trust collapse prospectively;
3. exact LJ retains coherent orders; and
4. a tied globalization or stable-block intervention rescues the
   predeclared failures.

If the residuals do not separate outcomes and the intervention does not
rescue Sella, reject the Taylor-compatibility mechanism.

---

## 13. Mechanistic synthesis

### 13.1 Explanations eliminated or weakened

| Candidate explanation | Status | Evidence |
|---|---|---|
| GAD and Sella use different LJ potentials | Rejected | E/F/H and vibrational spectra agree numerically |
| Missing or malformed full Hessian | Rejected for tested paths | Full \(3N\times3N\) Hessians and FD checks passed |
| Sella receives stale HIP Hessians | Rejected by project provenance | Current full HIP Hessian supplied every step |
| Retained mass weighting causes the benchmark | Rejected | Weighting removed before Cartesian displacement; retained-weighting path produced no data |
| One huge first Sella step explains exclusive failures | Not supported | 0.100 vs 0.088 A median first max-atom step |
| Consecutive \(v_1\) continuity explains the split | Not supported | Does not distinguish GAD-only from Sella-only |
| LJ failure is just timestep too large | Rejected | dt/cap sweeps plateau; index gate rescues |
| DFTB0 IRC failure proves bad optimizers | Rejected | PES-native labels and corrected IRC change the interpretation |
| Sella frequently converges to unintended HIP saddles | Rejected | Four unintended outcomes in 5166 five-tier records |
| Full learned Hessians are intrinsically bad for RFO | Rejected | HIP's own RFO results and NewtonNet are counterexamples |

### 13.2 Leading hypothesis

The strongest coherent sequence is:

1. Large isotropic perturbations move some HIP starts away from the
   path-local HORM/Transition1x distribution.
2. HIP's low-subspace-aware direct Hessian may retain a useful reaction-mode
   projector in part of this region.
3. Its independently learned energy, force, and full Hessian need not remain
   one accurate finite-step Taylor model in every direction.
4. GAD repeatedly consumes only the current force and lowest-mode projector.
5. Sella applies a full RS-P-RFO proposal and then adapts its next radius
   using learned-energy model agreement.
6. In some trajectories, an early departure plus persistent model-ratio
   disagreement may push Sella to its radius floor, while GAD remains inside
   the labelled basin.

Steps 1-3 and 6 are plausible and partly measured; their causal connection
is not.

### 13.3 Live alternative hypotheses

| Hypothesis | Distinguishing prediction |
|---|---|
| First-order energy/direct-force incompatibility | \(r_{E1}=O(\delta)\); \(\rho\) remains poor as radius shrinks |
| Second-order force/Hessian incompatibility | \(r_{F2}=O(\delta)\); replacing the stable block improves Sella |
| Privileged low-mode reliability | HIP \(v_1\) stays aligned with a native path after full-model residuals degrade |
| Isotropic-noise distribution shift | Crossover weakens for GSM/NEB starts and tracks OOD/compression descriptors |
| Basin/partial-IRC semantics | Gap shrinks when candidates are matched by destination basin |
| Trust-policy hysteresis | Reject-before-apply rescues cases without changing curvature |
| Ordinary optimizer tuning | Frozen validation-selected settings or another RS-P-RFO implementation closes the gap |
| HIP-specific spectral conditioning | Stable eigengaps/condition numbers predict failure even when Taylor residuals are normal |

### 13.4 Why the other PESs may not show the crossover

- **LJ:** exact curvature exposes an extreme, high-index repulsive entry
  region. Pure GAD fails before any subtle HIP-style interface distinction
  matters.
- **DFTB0:** internally coherent derivatives give Sella the full benefit of
  second-order modeling, while the T1x starts are often displaced from
  DFTB0-native saddles.
- **PaiNN:** conservative E/F/H eliminates product incompatibility, and the
  small pilot favors Sella for strict stationarity, although neither method
  recovers intended chemistry.
- **HORM:** the current screens are too small and insufficiently native to
  decide whether Hessian supervision shifts the boundary.
- **MACE:** the labelled structures are out of domain and often not
  stationary, preventing a fair optimizer comparison.

---

## 14. Discussion

### 14.1 What can be claimed now

1. Plain GAD and full-Hessian Sella have a reproducible, noise-dependent
   ranking crossover on the evaluated HIP/Transition1x test set.
2. The high-noise GAD advantage survives intended IRC/topology validation.
3. The same plain-GAD advantage does not generalize to DFTB0, LJ7, PaiNN, or
   the current HORM screen.
4. Pure GAD's LJ failure is well explained by stiff, high-index
   globalization and is nearly removed by an index-aware descent gate.
5. HIP's learned energy, direct force, and direct Hessian do not behave like
   the conservative PaiNN derivatives in local consistency tests.
6. Sella's GAD-only HIP failures exhibit departure and trust-floor collapse.

### 14.2 What cannot be claimed now

1. We do not yet know that Taylor incompatibility causes the HIP crossover.
2. We do not know that HIP \(v_1\) remains chemically correct in the
   GAD-only region.
3. We cannot call the force Jacobian a superior or intended HIP Hessian.
4. We cannot claim universal GAD superiority, universal Sella weakness, or
   universal benefit from direct Hessian prediction.
5. We cannot use HORM, MACE, or xTB as negative optimizer benchmarks without
   larger PES-native saddle sets.
6. We cannot call gated LJ's near-100% strict rate chemical recovery without
   a saddle-family endpoint classifier.

### 14.3 Potential paper framing

A coauthor-compatible and scientifically constructive framing is:

> Direct Hessian prediction exposes several levels of useful curvature
> information. Under path-local starts, the full HIP Hessian supports
> efficient RFO. Under sufficiently degraded starts, low-dimensional
> reaction-mode information may remain useful beyond the region where the
> complete learned energy/force/Hessian bundle is a reliable finite-step
> Taylor model. This creates an opportunity for reliability-aware curvature
> consumers that interpolate between force reflection and full RFO.

This treats HIP's independently supervised curvature channel as a feature
and design opportunity, not a defect. The novelty is not merely that \(v_1\)
is useful; learned-leftmost-mode RFO already exists. The potentially novel
claim is a **measurable reliability-radius gap that predicts which optimizer
interface will be robust**.

### 14.4 Fastest route to a defensible mechanism paper

1. Run the already implemented 12-case outcome-conditioned Taylor panel.
2. Instrument Sella's \(\rho\), predicted/actual energy change, proposed and
   applied step, and path distance before divergence.
3. Run one reject-before-apply or force-consistent globalization control.
4. Run one \(v_1\)-preserving, stable-block-replacement control.
5. Reject the mechanism if diagnostics are not predictive or the tied
   intervention does not rescue failures.
6. If causal, run matched HIP loss checkpoints.
7. Then test one independent PES-native reactive MLIP family, preferably
   matched HORM E-F/E-F-H or NewtonNet.

---

## 15. Limitations and invalidated analyses

- The HIP headline table reports completed-grid best values, including
  post-hoc per-cell configuration choice.
- Historical noise labels are wrong by a factor of ten.
- Training/test reaction overlap has not been fully audited.
- Direct-force/Jacobian substitution is an implementation diagnostic, not
  HIP's intended scientific product.
- A long paired Hessian-source substitution was invalidated by train/test
  split and GAD-configuration mismatches.
- The first DFTB0 IRC comparison was invalidated by a zero-step IRC-driver
  artifact and is superseded by the native endpoint result.
- Historical mass-dynamics interpretations are superseded by the confirmed
  mass-weighted-Eckart then Cartesian-displacement convention.
- HORM and MACE lack sufficiently large native saddle sets.
- PaiNN has only two validated native candidates and 12 paired trials.
- LJ strict success lacks reaction/saddle-family endpoint validation.
- Some timing and long-budget results are partial after Slurm timeout and
  biased toward early/smaller systems.
- GPU HIP evaluations are not bitwise deterministic at the precision needed
  for tiny force differences.

---

## 16. Reproducibility and artifact map

### Canonical documents

- `BENCHMARK_RESULTS_2026_07_16.md`
- `HIP_GAD_SELLA_SYNTHESIS_2026_07_17.md`
- `HIP_GAD_SELLA_THEORY_AUDIT_2026_07_17.md`
- `LJ_FINDINGS_2026_07_09.md`
- `SCINE_XTB_FINDINGS_2026_05_15.md`
- `FINDING_IRC_5TIER_2026_06_23.md`
- `legacy/notes/HYBRID_FINDINGS_CATALOG.md`
- `legacy/gad-paper/paper.tex`

### Principal result tables

- `analysis_2026_04_29/master_2026_05_16.csv`
- `analysis_2026_04_29/threshold_sweep_2026_05_16.csv`
- `analysis_2026_04_29/irc_outcomes_5tier_test287.csv`
- `analysis_2026_04_29/rmsd_to_known_ts_compare.csv`
- `analysis_2026_04_29/scine_topo_debug_10pm.csv`

### Structured mechanism records

- `experiments/2026-07-16-hip-field-consistency.json`
- `experiments/2026-07-16-painn-field-consistency.json`
- `experiments/2026-07-16-painn-paired-pilot.json`
- `experiments/2026-07-16-painn-paired-terminal-irc.json`
- `experiments/2026-07-17-hip-hessian-source-panel.json`
- `experiments/2026-07-17-hip-hessian-source-substitution.json`
- `experiments/2026-07-17-hip-paired-outcome-diagnostics.json`
- `experiments/2026-07-17-hip-taylor-compatibility.json`
- `experiments/2026-07-17-lj-gad-recurrence-audit.json`

### Important scratch roots

- HIP paired outcomes:
  `/lustre07/scratch/memoozd/gadplus/runs/hip-paired-outcomes-65828024/`
- LJ Taylor smoke:
  `/lustre07/scratch/memoozd/gadplus/runs/lj-taylor-local-smoke-20260717b/`
- PaiNN paired pilot:
  `/lustre07/scratch/memoozd/gadplus/runs/painn-gad-sella-65764476/`
- Corrected DFTB0 native topology:
  `/lustre07/scratch/memoozd/gadplus/runs/scine_native_topo_forced_10pm_intersection_20260710/`
- HORM exploratory screen:
  `/lustre07/scratch/memoozd/gadplus/runs/horm_formula_starts_zero_20260711/`

---

## 17. Primary literature

1. E and Zhou, [The Gentlest Ascent
   Dynamics](https://doi.org/10.1088/0951-7715/24/6/008), 2011.
2. Banerjee et al., [Search for Stationary Points on
   Surfaces](https://doi.org/10.1021/j100247a015), 1985.
3. Hermes et al., [Sella, an Open-Source Automation-Friendly Molecular
   Saddle Point Optimizer](https://doi.org/10.1021/acs.jctc.2c00395), 2022.
4. Schreiner et al.,
   [Transition1x](https://arxiv.org/abs/2207.12858), 2022.
5. Yuan et al., [Full analytical Hessians for transition-state optimization
   with neural network
   potentials](https://doi.org/10.1038/s41467-024-52481-5), 2024.
6. Cui et al., [HORM](https://arxiv.org/abs/2505.12447), 2025.
7. Burger et al., [Shoot from the
   HIP](https://arxiv.org/abs/2509.21624), 2025.
8. Wu et al., [Machine-Learned Leftmost Hessian Eigenvectors for Robust
   Transition State Finding](https://arxiv.org/abs/2603.21323), 2026.
9. Bigi et al., [The Dark Side of the Forces: Assessing Non-Conservative
   Force Models for Atomistic Machine
   Learning](https://arxiv.org/abs/2412.11569), ICML 2025.
10. Carter, [On the Global Convergence of Trust Region Algorithms Using
    Inexact Gradient Information](https://doi.org/10.1137/0728014), 1991.
11. Bandeira et al., [Convergence of Trust-Region Methods Based on
    Probabilistic Models](https://doi.org/10.1137/130915984), 2014.

---

## Appendix A. Structured experiment status ledger

| Experiment | Status | Result used in this draft |
|---|---|---|
| HIP field consistency | Passed | Quantitative HIP/PaiNN Taylor-interface residuals |
| PaiNN field consistency | Passed | Conservative control residuals |
| PaiNN native set, tight | Passed | Three constructed candidates |
| PaiNN native IRC | Passed | Two accepted candidates |
| PaiNN paired pilot | Completed | GAD 0/12, Sella 5/12 strict |
| PaiNN terminal IRC | Completed | Both 0/12 intended |
| PaiNN GAD dt grid | Completed | 0/12 at all five dt values |
| HORM LEFTNet matched smoke | Passed | Both E-F and E-F-H adapters operational |
| HORM LEFTNet FD grid | Passed | 0.548% and 1.29% best directional RMS residuals |
| HORM native set smoke | Completed | One candidate from 15 pairs |
| HORM LEFTNet-df smoke | Completed | Nonconservative direct field operational |
| HORM shard-3 IRC | Invalidated | Both branches reached same reactant-side minimum |
| HIP Hessian-source panel | Completed | 12-geometry curvature-source statistics |
| HIP Hessian substitution smoke | Completed | One-start source sensitivity only |
| HIP long paired substitution | Invalidated | Split mismatch; no optimizer rows |
| HIP dt-grid source substitution | Implementation failed | GAD settings did not match history |
| HIP paired outcome diagnostics | Completed | 133/34/22/98 and trust-collapse signature |
| HIP Taylor compatibility | Documented, not run | LJ numerical smoke only |
| LJ recurrence audit | Completed | Recurrence equivalence at numerical precision |
| MACE compatibility | Exploratory | No common-pool optimizer result |

Records left with `submitted` or `planned` status but without a completed
artifact are not counted as results. The project queue was empty at closure.

## Appendix B. Draft figures to assemble

1. HIP strict and IRC crossover versus corrected noise in A.
2. Four-way paired outcome Sankey or stacked bar at 0.15 A.
3. Sella distance-to-TS and trust-radius trajectories for GAD-only versus
   Sella-only starts.
4. Force-threshold curves showing the plain-GAD floor and hybrid/Sella
   refinement.
5. Cross-surface strict-success table or heatmap.
6. LJ initial-index/force distributions and the gate ablation.
7. HIP versus PaiNN Taylor-interface residuals.
8. Proposed causal figure: residual order and \(\rho\) before departure,
   followed by intervention rescue.
