# GADplus

GADplus is a research codebase for transition-state search and validation on
Transition1x.  It implements gentlest-ascent dynamics (GAD), Sella workflows,
Eckart-projected vibrational Hessian analysis, and IRC/topology validation on
neural and semiempirical potential-energy surfaces.

## Results gallery

The consolidated [LJ, g-xTB, and HIP results gallery](docs/research/BEST_METHODS_CROSS_SURFACE_2026_08_11.md)
contains the selected non-Sella implementations, every matched decision-grid
method, explicit Sella deltas, steps-to-convergence comparisons, and
convergence-criterion sensitivity tables. The corresponding minimal readable
methods are in [`examples/best_methods/`](examples/best_methods/).
For a tables-only view, use [every method on every surface](docs/research/ALL_METHODS_ALL_SURFACES_TABLES_2026_08_14.md).

## Current maintained workflow

The active g-xTB workflow is:

```text
labelled T1x TS
  -> projected, mass-weighted GAD or Cartesian+Eckart Sella
  -> n_neg == 1 AND fmax < 0.01 eV/Angstrom
  -> full-Hessian IRC + endpoint minimisation
  -> IRC_TOPO gate
```

The exact operational commands, gates, and calibration protocol are in
[docs/OPERATIONS.md](docs/OPERATIONS.md).  The Trillium transfer checklist is
in [docs/TRILLIUM_MIGRATION.md](docs/TRILLIUM_MIGRATION.md), and the
non-negotiable numerical conventions are in
[docs/DESIGN_CONTRACT.md](docs/DESIGN_CONTRACT.md).

## Repository map

| Location | Purpose |
| --- | --- |
| `src/gadplus/core/` | GAD, convergence, timestep, and mode-tracking logic |
| `src/gadplus/projection/` | mass weighting and Eckart vibrational projection |
| `src/gadplus/calculator/` | HIP, xTB, SCINE, g-xTB, and ASE adapters |
| `src/gadplus/search/` | GAD, Sella/IRC, and endpoint validation workflows |
| `configs/` | Hydra search, calculator, and cluster configuration |
| `scripts/` | maintained runners plus archived experiment scripts |
| `docs/` | current operational and migration documentation |
| `legacy/` | historical reports and reproducibility material |

## Quick start

```bash
# On the target Alliance cluster
CLUSTER=trillium bash scripts/setup_env.sh

# Verify the Python package
source .venv/bin/activate
python -c 'from gadplus.core.convergence import is_ts_converged; print("OK")'
```

For a g-xTB run additionally set `GADPLUS_T1X_H5`, `GADPLUS_GXTB_EXE`, and
`GADPLUS_GXTB_PARALLEL`; see the operations guide.

## Scientific convention

`n_neg` always refers to the number of negative eigenvalues of the
mass-weighted, Eckart-projected vibrational Hessian.  A TS candidate is not
accepted merely because an optimizer reports convergence: it must pass the
common projected `n_neg==1` and `fmax` gate.  IRC_TOPO is a separate,
downstream mechanism gate.

## Historical material

The prior result syntheses remain useful research context, but do not define
the current execution interface:

- [HIP/GAD/Sella synthesis](docs/research/HIP_GAD_SELLA_SYNTHESIS_2026_07_17.md)
- [theory and implementation audit](docs/research/HIP_GAD_SELLA_THEORY_AUDIT_2026_07_17.md)
- [cross-benchmark ledger](docs/research/BENCHMARK_RESULTS_2026_07_16.md)
- [experiment organisation](docs/EXPERIMENT_ORGANIZATION.md)
- [legacy archive](legacy/README.md)
