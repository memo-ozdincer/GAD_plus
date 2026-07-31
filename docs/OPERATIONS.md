# GADplus operations guide

This document defines the maintained execution paths.  It separates them from
the large `scripts/` archive, which is retained for reproducibility of prior
experiments but is not a supported interface for new work.

## Scientific invariants

All maintained transition-state runs use the same acceptance gate:

```text
TS accepted = projected vibrational n_neg == 1 AND fmax < 0.01 eV / Angstrom
```

`n_neg` is evaluated after mass weighting and Eckart projection in the
reduced vibrational subspace.  Translation/rotation modes are not counted.

For g-xTB, acceptance requires a second gate:

```text
IRC_TOPO accepted = TS accepted
                 AND both IRC endpoints relax to minima (n_neg == 0)
                 AND endpoint topology matches labelled T1x reactant/product
```

## Maintained commands

| Purpose | Entry point | Output |
| --- | --- | --- |
| Projected GAD | `scripts/gad_smoke.py` | `summary_*.parquet`, trajectories |
| Cartesian Eckart Sella | `scripts/sella_smoke.py` | `summary_*.parquet` |
| g-xTB GAD job | `scripts/run_gxtb_gad.slurm` | GAD summary and trajectories |
| g-xTB Sella job | `scripts/run_gxtb_sella.slurm` | Sella summary |
| g-xTB IRC_TOPO | `scripts/gxtb_irc_topo.py` / `run_gxtb_irc_topo.slurm` | `irc_topo.parquet` |
| g-xTB dt grid | `scripts/trillium/run_gxtb_gad_dt_grid.slurm` | one GAD summary per fixed timestep |
| g-xTB Hessian timing | `scripts/trillium/run_gxtb_hessian_compare.slurm` | raw/projected comparison and timing JSON |
| Hydra/HIP workflows | `python -m gadplus.orchestration.run` | cluster scratch run directory |

The Sella runner requests a supplied full Hessian at every optimizer
diagnostic (`--diag-every 1`) and refreshes its callback after each PES kick.
This prevents BFGS carry-over from silently replacing the g-xTB Hessian.

## Required environment variables

No maintained g-xTB command requires a Narval path embedded in its Python
driver.  Set these variables in each batch script or shell:

```bash
export GADPLUS_T1X_H5=/path/to/data/transition1x.h5
export GADPLUS_GXTB_EXE=/path/to/xtb
export GADPLUS_GXTB_PARALLEL=32
```

The executable is the `xtb` binary distributed with the cloned
[Grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb) release.  It is not
tracked in this repository.  `GADPLUS_GXTB_PARALLEL` is passed to
`xtb --parallel`; do not combine it with multiple simultaneous trajectory
workers unless the total requested cores remains within the Slurm allocation.

## Pre-flight checks

Run these before submitting an expensive campaign:

```bash
python -m py_compile scripts/gad_smoke.py scripts/sella_smoke.py \
  scripts/gxtb_irc_topo.py src/gadplus/calculator/gxtb.py
"$GADPLUS_GXTB_EXE" --version
python -c 'from sella import Sella; import torch; print(Sella, torch.__version__)'
```

Then run one sample with two steps.  A valid job must show an `xtb` child
process and produce a nonempty summary parquet.  A completed Slurm job alone
is not evidence that a Hessian was evaluated.

## Timing and tuning protocol for g-xTB GAD

1. On a representative labelled T1x TS, time one `do_hessian=True` call with
   the intended core count.
2. Submit `scripts/trillium/run_gxtb_gad_dt_grid.slurm`: it runs the same
   sample, seed, and noise at `0.001, 0.003, 0.005, 0.007` by default.
3. Compare stability, projected `n_neg`, `fmax`, and wall time per completed
   step.  Pick a timestep from observed behaviour; do not carry over a HIP
   timestep as a calibrated g-xTB parameter.
4. Run the selected GAD configuration and Cartesian+Eckart Sella with full
   Hessian every step.
5. Send only rows passing the common TS gate into `gxtb_irc_topo.py`.

## Result provenance

Each campaign should retain:

- Slurm script and submitted environment values;
- g-xTB executable version/commit;
- exact T1x HDF5 path and split;
- summary parquet(s), IRC parquet, and job logs;
- a small README beside the output recording timestep, noise, cores, and
  gate counts.

Do not put generated run output, parquet, virtual environments, or the g-xTB
binary clone under version control.
