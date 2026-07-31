# GADplus contributor guide

Read these before changing a scientific workflow:

1. [`docs/DESIGN_CONTRACT.md`](docs/DESIGN_CONTRACT.md) — non-negotiable
   numerical conventions.
2. [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — maintained commands and
   campaign protocol.
3. [`docs/TRILLIUM_MIGRATION.md`](docs/TRILLIUM_MIGRATION.md) — target-cluster
   migration checklist.

## Maintained surface

- `src/gadplus/search/gad_search.py`: projected GAD loop.
- `scripts/gad_smoke.py`: direct batch-friendly GAD runner.
- `scripts/sella_smoke.py`: Cartesian Eckart Sella runner.
- `src/gadplus/calculator/gxtb.py`: external g-xTB adapter.
- `scripts/gxtb_irc_topo.py`: common n_neg/fmax + IRC_TOPO gate.
- `configs/`: portable Hydra settings for Narval and Trillium.

Keep calculator-specific code inside `src/gadplus/calculator/`.  Search code
uses the common `PredictFn` interface and must not acquire a HIP, xTB, or
Sella-specific import.

## Rules that must not drift

- HIP Hessians are model-provided direct Hessians, not autograd substitutes.
- Projected Hessians are mass-weighted, Eckart-projected, and analysed in the
  reduced vibrational subspace.  Sella receives the result mapped back to
  Cartesian units.
- TS acceptance is `n_neg == 1 AND fmax < threshold`; optimizer status alone
  is not acceptance.
- Maintained Sella runs supply a full Hessian every step.
- g-xTB is CPU parallel through `xtb --parallel`; avoid nested process pools.

Historical scripts, result reports, and experiment notes are retained under
`scripts/`, `docs/research/`, and `legacy/` for provenance.  Do not use them
as the default execution path without explicitly deciding to reproduce a
historical experiment.
