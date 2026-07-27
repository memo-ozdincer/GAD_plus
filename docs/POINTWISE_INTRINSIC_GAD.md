# Pointwise intrinsic smooth-index GAD

## Status and scope

`gadplus.search.intrinsic_gad` is an experimental transition-state optimizer
for analytic Lennard-Jones potentials and calculators that provide a coherent
energy, gradient, and full Cartesian Hessian. It is separate from the
historical fixed-step `lambda_2`-gated runner and does not change the maintained
GAD/Sella contract.

The method is designed to satisfy a strict requirement: the update at `q`
must be a deterministic function of only `q`, `g(q)`, and `H(q)`. It therefore
has no line search, rejected trial, adaptive trust-radius state, mode tracking,
momentum, quasi-Newton history, or global pseudo-potential.

## 1. Geometry and local spectral data

Let

\[
x=M^{1/2}q,\qquad
g_x=M^{-1/2}g_q,\qquad
H_x=M^{-1/2}H_qM^{-1/2}.
\]

At the current geometry, construct an orthonormal Eckart vibrational basis
`Q(q)` in mass-weighted space and diagonalize

\[
K(q)=Q^\top H_xQ=V\Lambda V^\top,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_m).
\]

Here `m=3N-6` for a nonlinear cluster and `m=3N-5` for a linear molecule. The
gradient coefficients in the instantaneous vibrational eigenbasis are

\[
c=V^\top Q^\top g_x.
\]

The reported numerical Morse index uses the maintained tolerance

\[
n_{\rm neg}^{(\tau_I)}=\#\{i:\lambda_i<-\tau_I\},
\qquad \tau_I=10^{-4}\ \text{by default}.
\]

This discrete index is used only for final TS acceptance, not to branch the
dynamics.

## 2. Basis-invariant soft lowest-mode operator

Define the local RMS curvature scale

\[
s_H=\left(\frac1m\sum_i\lambda_i^2\right)^{1/2}
\]

and a dimensionless spectral temperature `tau_s`. The normalized matrix
soft-min is

\[
\rho_{\tau_s}(K)
=
\frac{\exp[-K/(\tau_s s_H)]}
     {\operatorname{tr}\exp[-K/(\tau_s s_H)]}
=V\operatorname{diag}(p_i)V^\top,
\]

where

\[
p_i=\frac{\exp[-\lambda_i/(\tau_s s_H)]}
          {\sum_j\exp[-\lambda_j/(\tau_s s_H)]}.
\]

For finite `tau_s`, this matrix function is analytic in `K`. At an exact
eigenvalue degeneracy, equal eigenvalues receive equal weights, so `rho` is
invariant under arbitrary rotations of the eigensolver basis. As
`tau_s -> 0` and the lowest eigenvalue is isolated, `rho` approaches
`u_1 u_1^T`.

There is no globally smooth, rotation-equivariant rule that selects one vector
from an exactly degenerate eigenspace without adding symmetry-breaking data.
The density operator is the non-arbitrary alternative; it does not conceal a
tracked previous mode.

## 3. Smooth `lambda_2` gate

The gate is

\[
w(q)=\sigma\!\left(\frac{\lambda_2(q)}{\tau_s s_H(q)}\right).
\]

The same dimensionless temperature controls the mode resolution and gate
width. Ordered eigenvalues of a symmetric matrix are continuous under matrix
perturbations, including at crossings, but they need not be differentiable
there. Consequently, this gate is continuous and piecewise smooth rather than
globally `C-infinity`.

This is intentional. A finite-temperature surrogate for the second order
statistic can be made analytic, but introduces dimension-dependent bias away
from the actual zero crossing. The current construction keeps the meaningful
boundary `lambda_2=0`, while the density operator removes the dangerous
eigenvector ambiguity.

In the eigenbasis, the gated gradient is

\[
b_i=(1-2wp_i)c_i.
\]

For a separated lowest mode and `lambda_2 >> 0`, `p_1 -> 1`, `w -> 1`, and
`b_1 -> -c_1` while `b_i -> c_i` for `i>1`: ordinary one-mode GAD. For
`lambda_2 << 0`, `w -> 0` and `b -> c`: vibrational force descent. Unlike a
projector onto every negative mode, this does not make an index-`k` saddle
attracting by ascending along all `k` negative modes.

## 4. Closed-form pointwise regularization

A conventional adaptive trust region retains a radius based on prior accepted
or rejected trials. It therefore violates the strict pointwise Markov
requirement. An energy-decrease acceptance test is also inappropriate because
valid GAD motion deliberately raises energy along the selected mode.

Instead, define a current-geometry length

\[
\ell(q)
=
\left[
\frac{1}{N_p}\sum_{a<b}r_{ab}^{-2}
\right]^{-1/2},
\]

where `N_p=N(N-1)/2`. This inverse-RMS pair length is permutation and
rigid-motion invariant, scales linearly with the geometry, and contracts
smoothly toward zero as any pair collides.

The allowed mass-weighted RMS scale is

\[
R(q)=\eta\,\ell(q)\sqrt{\sum_a m_a},
\]

with dimensionless locality fraction `eta`. Define the pointwise regularizer

\[
\mu(q)=\frac{\|b(q)\|_2}{R(q)}
\]

and the closed-form step coefficients

\[
a_i(q)
=
-\frac{b_i(q)}{\sqrt{\lambda_i(q)^2+\mu(q)^2}}.
\]

Finally,

\[
q^+=q+M^{-1/2}QV a.
\]

This is a pointwise regularized eigenvector-following map, not Euler
integration and not a minimizer trust-region algorithm.

### Algebraic step bound

Every denominator is at least `mu`, hence

\[
\|a\|_2^2
\le
\frac{\|b\|_2^2}{\mu^2}
=R^2.
\]

Therefore

\[
\frac{\|M^{1/2}\delta q\|_2}{\sqrt{\sum_a m_a}}
\le \eta\ell(q).
\]

No step is computed and then clipped. Curvature and the current gradient
regularize the step before it is formed.

### Scale covariance

Under a positive energy rescaling `E -> alpha E`, the quantities `c`,
`lambda`, `s_H`, and `mu` all multiply by `alpha`; `p`, `w`, and `a` are
unchanged. Under a uniform mass rescaling `M -> gamma M`, the mass-weighted
step coefficients scale by `sqrt(gamma)` and the Cartesian back-transform by
`1/sqrt(gamma)`, so the Cartesian update is again unchanged.

## 5. Audit against the four strict criteria

### Pointwise Markov property

Satisfied. The map is

\[
q^+=\Psi(q,g(q),H(q)).
\]

Iteration count, previous coordinates, previous modes, past forces, and past
radii do not enter `Psi`. Recording a returned diagnostic history does not
affect the map and can be disabled.

### Local evaluability through second order

Satisfied. Only the current coordinates, gradient, and Hessian are needed for
the step. Current energy is queried for reporting, not for step acceptance.
No minimum, endpoint, reaction path, or reaction coordinate is required.

### Non-conservative-field tolerance

Satisfied. The code evaluates the map directly. It never assumes that the
gated field is the gradient of a global scalar and never performs scalar
energy minimization, a Metropolis test, or an energy line search.

### Spectral-boundary robustness

Satisfied as a `C0`, piecewise-smooth method. There is no hard branch on
`n_neg`. The soft density is basis-invariant at eigenvector crossings, the
ordered `lambda_2` gate is continuous, and the pointwise step is algebraically
bounded. Derivatives of the map can still be nonsmooth where ordered
eigenvalues meet; the method does not claim otherwise.

## 6. Target selection and the symmetry limit

A deterministic, symmetry-equivariant pointwise algorithm cannot select one
of several exactly degenerate exit directions at a perfectly symmetric
stationary minimum. Moreover, every force-based saddle method has zero step at
an exact stationary point unless extra information is supplied.

Target selection therefore belongs in initialization:

1. Relax a starting minimum.
2. Compute its vibrational modes.
3. Choose a physical mode and sign.
4. Scan a small, declared displacement ladder `alpha` along that mode.
5. Start the pointwise optimizer from the displaced geometry.

The mode and displacement are experimental inputs, not hidden optimizer
memory. For a degenerate subspace, choose a physically interpretable linear
combination or sample directions uniformly within that subspace.

A fixed bias potential `E_bias(q)` remains pointwise if it is a declared
function of the current geometry. It changes the surface and must be reported
as a different method. Turning the bias off according to elapsed iterations or
trajectory events introduces history and violates the strict position-only
criterion.

Thermal momenta may be used to sample the initial mode and sign. Carrying
momentum during the search violates a position-only Markov specification,
although it can be Markovian in an expanded `(q,p)` state. The maintained
pointwise method does not carry momentum.

## 7. Validation protocol

### Mathematical tests

- positive energy-rescaling invariance of the policy and step;
- uniform-mass-rescaling invariance of the Cartesian step;
- basis invariance inside exact degenerate eigenspaces;
- algebraic mass-weighted RMS step bound;
- descent limit for clear index greater than one;
- one-mode ascent limit for clear index one;
- deterministic equality of updates from identical local evaluations.

### Analytic LJ tests

- LJ7 pushed-mode deterministic smoke;
- the historical 287 fixed-seed LJ7 starts at noise `0.10`, `0.15`, and `0.20`;
- paired comparison against pure GAD, historical `k=50` gating, and Sella;
- LJ13 and LJ38 mode-push panels stratified by initial index and close-pair
  distance;
- final projected index, force threshold, energy family, permutation-aligned
  structure, and both downhill endpoint basins.

Strict TS convergence remains

\[
n_{\rm neg}^{(\tau_I)}=1
\quad\land\quad
\|F\|_\infty<f_{\max}.
\]

Convergence alone is not intended-saddle recovery. LJ comparisons must add an
endpoint or saddle-family classifier before making selectivity claims.

### MLIP and DFT tests

- verify energy/force/Hessian coherence before optimizer comparison;
- use identical geometries, masses, projection, thresholds, and Hessian
  refresh policy across methods;
- report energy, gradient, and Hessian evaluations separately because their
  costs differ by orders of magnitude;
- report wall time and calculator failures in addition to iteration count;
- validate accepted candidates with a higher-level Hessian and two downhill
  branches when feasible.

The method requests one full local Hessian per pointwise update. It is not
expected to beat gradient-only dimer methods in raw gradient-equivalent cost
unless Hessians are inexpensive, batched, or already required by the surface.

## 8. Current tests

`tests/test_intrinsic_gad.py` covers energy-scale covariance, degeneracy
invariance, both gate limits, the closed-form radius proof numerically, and a
deterministic analytic LJ7 pushed-mode convergence test.

