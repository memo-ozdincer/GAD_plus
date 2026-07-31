# GADplus design contract

This is the maintained implementation contract.  It is deliberately narrower
than the historical experiment record: a change that violates this document is
not a new method variant, it is a regression unless the contract is updated
with explicit scientific justification.

## 1. Common calculator interface

All search and validation code receives a `PredictFn`:

```python
predict_fn(coords, atomic_nums, do_hessian, require_grad) -> {
    "energy": ...,
    "forces": ...,
    "hessian": ...,
}
```

Energy is eV, forces are eV/Angstrom, and Hessians are eV/Angstrom².  Search
code never imports a particular potential package directly.

## 2. Hessian source is explicit

- HIP uses its model-provided direct Hessian (`do_hessian=True`).  It does not
  substitute an autograd or finite-difference Hessian in maintained runs.
- g-xTB invokes the release `xtb --hess` calculation.  `--grad --hess` is not
  valid for this release because it silently omits the Hessian.
- A full Hessian is required whenever a projected Morse index is evaluated.

## 3. Projection order

For every projected Hessian:

```text
Cartesian H
  -> mass-weight H_mw = M^(-1/2) H M^(-1/2)
  -> Eckart vibrational projection in mass-weighted coordinates
  -> eigendecomposition in the reduced vibrational subspace
```

When Sella requires a Cartesian Hessian callback, the projected matrix is
mapped back exactly once:

```text
H_cart_projected = M^(1/2) H_mw_projected M^(1/2)
```

The optimizer therefore receives Cartesian units, not a residual
mass-weighted Hessian.  Translation/rotation modes are removed by the
Eckart basis; threshold filtering is not a substitute for that projection.

## 4. Acceptance gates

The only maintained TS acceptance gate is:

```text
projected n_neg == 1 AND fmax < configured threshold
```

The default threshold is `0.01 eV/Angstrom`.  Optimizer-native convergence
flags are recorded but cannot override this gate.

`IRC_TOPO` is a downstream gate: candidates must first pass the TS gate, both
IRC branches must relax to projected minima (`n_neg == 0`), and the resulting
endpoint topology must match the labelled T1x reactant/product pair.

## 5. Sella rule

The maintained Sella path is Cartesian plus Eckart.  It must use
`diag_every_n=1` and a full-Hessian callback.  The callback cache only avoids
recomputing an identical geometry inside the same optimizer evaluation;
`refresh_hessian_after_kicks` forces a fresh supplied Hessian after every PES
kick.  BFGS updates cannot become the Hessian source for the next step.

## 6. g-xTB parallelism

g-xTB is CPU parallel through `xtb --parallel`.  One trajectory consumes one
Slurm allocation; do not create a process pool around a threaded Hessian
calculation.  GPU requests are appropriate only for a separate GPU-backed
surface, not for a g-xTB Hessian.

## 7. Calibration rule

GAD timestep is surface-specific.  A HIP timestep is a starting hypothesis,
not a g-xTB calibration.  Every g-xTB campaign records a small fixed-seed
dt sweep and per-Hessian timing before its production run.
