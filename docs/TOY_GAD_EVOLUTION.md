# Soft-Spectral Gentlest Ascent Dynamics: a five-minute toy

Run this single file after installing NumPy and Matplotlib:

```bash
python examples/gad_evolution_toy.py --output gad_evolution_toy.png
```

It produces one figure with four trajectories on the same analytic surface
and one small spectral inset. The code has no calculator, optimizer state,
dataset, or molecular machinery.

## The surface

```math
E(x,y)=(x^2-1)^2+(y^2-1)^2.
```

It has four minima \((\pm1,\pm1)\), four index-one edge saddles, and an
index-two centre at \((0,0)\). Therefore it contains the exact failure that
ordinary one-mode GAD does not address: far from an index-one saddle, more
than one direction can have negative curvature.

All panels use the same local, identity-mass specialization of the intrinsic
step:

```math
b_i=(1-2wr_i)c_i,
\qquad
\mu=\lVert b\rVert/R,
\qquad
a_i=-b_i/\sqrt{\lambda_i^2+\mu^2},
\qquad
q^+=q+Ua.
```

Here \(U\) diagonalizes the current Hessian, \(c=U^Tg\), and \(R\) is a
fixed visual-locality scale. In the molecular method, \(R(q)\) is the
intrinsic pair-distance scale described in `POINTWISE_INTRINSIC_GAD.md`.

## Reading the figure

1. **Ordinary GAD** sets \(w=1\) and reflects only the lowest eigenvector.
   It has no instruction for the additional unstable direction near the
   four-well centre.
2. **Smooth \(\lambda_2\) gate** uses
   \(w_2=\sigma[\lambda_2/(\tau_s s_H)]\). It becomes descent in the clear
   index-two region, then turns continuously into one-mode GAD as
   \(\lambda_2\) becomes positive. This is the essential LJ7 fix.
3. **Competitive gate** uses the current gradient activity to turn ascent on
   earlier only when the soft direction is active relative to the *other
   negative* directions. It is the g-xTB minimum-capture fix.
4. **Competitive Soft-Spectral GAD (CS²-GAD)** keeps that scalar gate but replaces the
   normalized soft density \(p_i\) in the reflection with
   \(\widetilde p_i=p_i/\max_jp_j\). The inset shows why: near a degenerate
   soft spectrum, \(p_1<1\), so the rank-one multiplier
   \(1-2wp_1\) can be only weakly negative. The relative density restores a
   full reflection for the lowest soft subspace without adding a parameter or
   path memory.

The toy is deliberately not offered as an LJ7 or chemical benchmark. LJ7 is
the more realistic validation surface; this figure is the shortest honest
explanation of why each gate was introduced.
