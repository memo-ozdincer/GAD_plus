# IRC Overview

This is the fastest orientation doc for the current IRC pipeline.

## What The Current IRC Pipeline Does

The current workflow is:

1. Take converged GAD trajectories from `noise_survey_300`.
2. For one noise level, select up to `N` converged rows from `summary_*.parquet`.
3. For each row, choose a TS candidate from the saved trajectory parquet.
4. Recompute TS quality with HIP on that candidate:
   - projected `n_neg`
   - `force_norm`
   - `fmax`
5. If enabled, refine the TS candidate with projected GAD.
6. Apply the TS quality criterion.
7. If the candidate passes, run forward and reverse IRC using Sella's IRC routine.
8. Compare the two IRC endpoints to the dataset-labeled reactant/product using:
   - aligned geometry (`aligned_rmsd_by_element`)
   - bond-topology graph matching
9. Save a rich validation parquet row plus a viewer bundle.

Important distinction:
- GAD is used for TS search and optional TS refinement.
- Sella is only used for the IRC path-following step.

## Which Geometries Are Being Tested

The default `run_irc_validate.slurm` path uses:
- input source: `noise_survey_300`
- starting geometry in that survey: `Transition1x` labeled TS + Gaussian noise
- current array noise levels: `0 pm`, `10 pm`, `50 pm`

So this is not the geodesic-midpoint workflow by default.

## File Map

### Main driver

- [scripts/irc_validate.py](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/scripts/irc_validate.py)

This is the top-level script that:
- loads converged rows from `summary_*.parquet`
- chooses TS candidates from `traj_*.parquet`
- recomputes TS quality
- optionally refines with projected GAD
- calls the IRC runner
- writes Parquet outputs
- writes viewer bundles

Key arguments:
- `--noise-pm`
- `--max-validate`
- `--rmsd-threshold`
- `--ts-pick`
- `--ts-force-criterion`
- `--ts-force-threshold`
- `--refine-ts`
- `--refine-steps`
- `--refine-dt`
- `--refine-force-criterion`
- `--refine-force-threshold`
- `--skip-if-ts-poor`

### IRC engine

- [src/gadplus/search/irc_validate.py](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/src/gadplus/search/irc_validate.py)

This file contains `run_irc_validation(...)` and the `IRCResult` dataclass.

Responsibilities:
- build ASE atoms + HIP ASE calculator
- call Sella's `IRC`
- run both directions
- compute endpoint RMSDs to dataset reactant/product
- compute topology-based endpoint matches
- return the final labels:
  - `intended`
  - `half_intended`
  - `topology_intended`
  - `topology_half_intended`

### Alignment helper

- [src/gadplus/geometry/alignment.py](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/src/gadplus/geometry/alignment.py)

Used for element-aware aligned RMSD.

This is important because the current IRC comparison is not a naive coordinate RMSD:
- Kabsch alignment
- assignment constrained by element identity

### Batch launchers

- [scripts/run_irc_validate.slurm](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/scripts/run_irc_validate.slurm)
- [scripts/run_geodesic_irc.slurm](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/scripts/run_geodesic_irc.slurm)

`run_irc_validate.slurm` is the normal scaled batch launcher.

Current defaults in that script:
- pre-IRC screen: `fmax < 0.01`
- refinement enabled
- refinement: `600` steps at `dt=0.003`
- refined criterion: `fmax < 0.006`
- noise array: `0, 10, 50 pm`

`run_geodesic_irc.slurm` is a mixed script:
- task `0`: geodesic-midpoint starting-geometry run
- later tasks: IRC validation jobs

### Visualization and re-export

- [scripts/visualize_irc.py](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/scripts/visualize_irc.py)

This re-exports saved IRC rows as viewer bundles.

Current viewer payload is endpoint/context, not a dense per-step IRC movie:
- dataset reactant reference
- reverse IRC endpoint
- TS input
- forward IRC endpoint
- dataset product reference

Related plotting scripts:
- [scripts/plot_irc_refined_summary.py](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/scripts/plot_irc_refined_summary.py)
- [scripts/plot_irc_refined_fmax.py](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/scripts/plot_irc_refined_fmax.py)

### Report / snapshot doc

- [IRC_RESULTS_2026-04-15.tex](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/IRC_RESULTS_2026-04-15.tex)
- [IRC_RESULTS_2026-04-15.pdf](/home/memoozd/projects/rrg-aspuru/memoozd/GAD_plus/IRC_RESULTS_2026-04-15.pdf)

This is the standalone IRC note with:
- current results
- threshold analysis
- figures
- current queue snapshot at compile time

## Output Locations

Current scaled outputs live under:

- `/lustre07/scratch/memoozd/gadplus/runs/irc_validation_300/`

Important files there:
- `irc_validation_0pm.parquet`
- `irc_validation_10pm.parquet`
- `irc_validation_50pm.parquet`
- `viewer_noise_0pm/`
- `viewer_noise_10pm/`
- `viewer_noise_50pm/`

## What Counts As "Failure"

There are two different failure modes.

### 1. Pre-IRC failure

The row gets:
- `error = "ts_quality_gate_failed"`

This means:
- a candidate TS was selected
- optionally refined
- but the final TS still did not satisfy the quality criterion
- so IRC was never run

### 2. Post-IRC unintended result

IRC did run, but the endpoints did not recover the labeled reactant/product pair.

Geometric threshold:
- current default `rmsd_threshold = 0.3 Å`

Topology is tracked separately, so a row can be:
- geometrically unintended
- but topology-half-intended

## Useful Commands

Run the default array:

```bash
cd /lustre06/project/6033559/memoozd/GAD_plus
sbatch scripts/run_irc_validate.slurm
```

Re-export one saved result as a viewer bundle:

```bash
cd /lustre06/project/6033559/memoozd/GAD_plus
source .venv/bin/activate
python scripts/visualize_irc.py \
  --results-parquet /lustre07/scratch/memoozd/gadplus/runs/irc_validation_300/irc_validation_10pm.parquet \
  --run-id noise_10pm_ca33b8bf \
  --sample-id 175
```

Rebuild the standalone IRC note:

```bash
cd /lustre06/project/6033559/memoozd/GAD_plus
pdflatex -interaction=nonstopmode -halt-on-error IRC_RESULTS_2026-04-15.tex
```

## Current Limitations

1. The current visualization is endpoint/context only; it does not save every internal IRC integration frame.
2. Endpoint comparison is against dataset-labeled reactant/product, not against an unsupervised family of minima on HIP's PES.
3. The current default batch script only covers `0/10/50 pm` from the noise-survey pipeline.

## If You Resume This On Another Machine

Read these in this order:

1. `IRC_OVERVIEW.md`
2. `IRC_RESULTS_2026-04-15.pdf`
3. `scripts/irc_validate.py`
4. `src/gadplus/search/irc_validate.py`

That should be enough to recover the logic without re-deriving the whole experiment history.
