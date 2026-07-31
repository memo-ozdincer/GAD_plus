# Maintained scripts

This directory deliberately contains only the active interfaces.

| File | Role |
| --- | --- |
| `setup_env.sh` | create a cluster-local project environment |
| `migrate_to_trillium.sh` | non-destructive Narval-to-Trillium transfer helper |
| `gad_smoke.py` | projected GAD runner with common TS gate |
| `sella_smoke.py` | Cartesian+Eckart Sella runner with full Hessian each step |
| `gxtb_irc_topo.py` | full-Hessian IRC/topology validation |
| `run_gxtb_*.slurm` | Narval g-xTB templates |
| `trillium/run_gxtb_*.slurm` | Trillium CPU g-xTB templates |
| `trillium/run_gxtb_gad_dt_grid.slurm` | fixed-seed, 4-point g-xTB timestep calibration array |
| `trillium/run_gxtb_hessian_compare.slurm` | g-xTB Hessian timing and projected-Hessian comparison |
| `lib/gxtb_runtime.sh` | shared runtime/environment validation |
| `compare_gxtb_dxtb_hessians.py` | controlled g-xTB/dxTB Hessian comparison |

Read [`../docs/OPERATIONS.md`](../docs/OPERATIONS.md) before running any of
these files.  The scientific invariants live in
[`../docs/DESIGN_CONTRACT.md`](../docs/DESIGN_CONTRACT.md).

## Historical scripts

`archive/` holds prior campaign launchers, plotting code, and one-off
analysis.  They are preserved as research provenance but are not maintained
and should not be used as templates for new runs.  Their original names are
retained to make Git history and old reports traceable.
