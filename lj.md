# Lennard-Jones LJ7 optimizer notes

## Transition-state criterion

For these notes, count a structure as a transition-state candidate when:

- `final_n_neg == 1`
- `final_force_max < 0.05`

The `converged` column in older summary parquet files was written with the
runner default `--force-threshold 1e-3`, so it undercounts successes when we
use the looser `0.05` force threshold.

## Completed runs

### Smoke run

Output directory: `runs/lj_smoke`

This was a one-sample smoke test across regular GAD, plain hybrid, Eckart
hybrid, and damped Eckart hybrid. It was useful only as a plumbing check.
None of the one-sample smoke jobs found a TS candidate under `fmax < 0.05`.

| Method | Samples | TS at `fmax < 0.05` | Final index pattern | What happened |
| --- | ---: | ---: | --- | --- |
| `gad` | 1 | 0 | `{0: 1}` | Landed at index 0 with high force. |
| `hybrid` | 1 | 0 | `{2: 1}` | Stayed high-index/high-force. |
| `hybrid_eckart` | 1 | 0 | `{2: 1}` | Stayed in `projected_gad`; no useful convergence. |
| `hybrid_damped_eckart` | 1 | 0 | `{2: 1}` | Stayed in `projected_gad`; no useful convergence. |

### Main LJ sweep

Output directory: `runs/lj_sweep_20260528_060902`

Inferred shared setup from the summary files:

- LJ7, `epsilon=1`, `sigma=1`
- `start_from=minimum_noised`, `noise=0.05`
- `n_samples=24`
- GAD `dt=0.001`
- Hybrid `gad_dt=0.001`, `trust_radius=0.01`, `switch_force=0.001`
- Eckart variants used the runner default `--atomic-number 18`, so their internal
  coordinates were argon mass-weighted.

| Method | Samples | TS at `fmax < 0.05` | Index-1 finals | Final `fmax` range | Last-step behavior |
| --- | ---: | ---: | ---: | --- | --- |
| `gad` | 24 | 20 | 21 | `0.0009903..1.312` | Regular GAD found the LJ7 saddle family often. |
| `hybrid` | 24 | 4 | 4 | `1.45e-14..0.0009994` | Most failed cases relaxed to index-0 minima after Newton. |
| `hybrid_eckart` | 24 | 0 | 12 | `1.065..3.516` | All rows ended with `projected_gad`; it never switched to Newton. |
| `hybrid_damped_eckart` | 24 | 0 | 12 | `1.065..3.516` | Same behavior as plain Eckart in this run. |

Regular GAD was the clear winner under the `0.05` force criterion: 20 of 24
samples reached an index-1 structure with small enough force. The successful
structures all appear to be the same LJ7 saddle family, with energy around
`-15.4447`, `eig0` around `-0.25`, and `eig1` around `0.788`.

The plain non-Eckart hybrid found 4 of 24 TS candidates. Its failed cases did
not fail because of force convergence; they mostly ended as very well-converged
index-0 structures with near-zero force, so Newton polished them into minima
rather than preserving index 1.

The two Eckart hybrids did not find TS candidates. They did reach index 1 in
12 of 24 samples, but the forces stayed large (`fmax` roughly `1.7..3.5` on
the index-1 rows), so these are not TS candidates under `fmax < 0.05`.

## Main lesson from the completed runs

The initial Eckart settings were too conservative and never entered the Newton
phase. Every completed Eckart row ended with `last_step_method = projected_gad`.
The switch criterion used `switch_force=0.001`, but the internal force norms
were still much larger than that, so the optimizer kept taking projected GAD
steps until `n_steps` was exhausted.

The default LJ Eckart mass choice was also probably wrong for this analytic
surface. The runner default `--atomic-number 18` makes the Eckart internal
coordinates argon mass-weighted. For LJ this is arbitrary, and it effectively
shrinks the projected GAD progress relative to the non-Eckart runs.

## Follow-up jobs launched on 2026-06-08

These are not counted as completed results yet.

First attempt, requested changes:

- `--atomic-number 1`
- larger `--gad-dt`
- larger `--switch-force`
- `--force-threshold 0.05`

Submitted jobs:

| Job ID | Status at last check | Output directory | Command intent |
| --- | --- | --- | --- |
| `28849639` | running | `runs/lj_unitmass_sf005_tr005/hybrid_eckart_dt001` | `hybrid_eckart`, unit mass, `gad_dt=0.01`, `switch_force=0.05`, `trust_radius=0.05` |
| `28849640` | running | `runs/lj_unitmass_sf005_tr005/hybrid_eckart_dt002` | `hybrid_eckart`, unit mass, `gad_dt=0.02`, `switch_force=0.05`, `trust_radius=0.05` |
| `28849641` | canceled before running | `runs/lj_unitmass_sf005_tr005/hybrid_damped_eckart_dt001_idxctl` | superseded by a higher-switch-force test |
| `28849642` | canceled before running | `runs/lj_unitmass_sf005_tr005/hybrid_damped_eckart_dt002_idxctl` | superseded by a higher-switch-force test |

Early trajectory output from jobs `28849639` and `28849640` showed that
`switch_force=0.05` is still too small for the unit-mass Eckart coordinates.
The first samples stayed in `projected_gad`, hit the `trust_radius=0.05` step
cap, and oscillated with internal force norms around `5.8..6.7`. Their final
sample-level lines so far were index 1 but high force (`fmax` about `2.5..3.1`),
not TS candidates.

Second attempt submitted to actually exercise the Newton branch:

| Job ID | Status at last check | Output directory | Command intent |
| --- | --- | --- | --- |
| `28849659` | pending | `runs/lj_unitmass_sf10_tr005/hybrid_eckart_dt001` | `hybrid_eckart`, unit mass, `gad_dt=0.01`, `switch_force=10.0`, `trust_radius=0.05` |
| `28849660` | pending | `runs/lj_unitmass_sf10_tr005/hybrid_damped_eckart_dt001_idxctl` | damped Eckart, unit mass, `gad_dt=0.01`, `switch_force=10.0`, `trust_radius=0.05`, `target_mode_strategy=neg_force_coupling`, `high_index_descent=index_controlled` |

## Recommended next interpretation

For LJ7, regular GAD is already effective under `fmax < 0.05`. The hybrid
methods need better index control before Newton polishing. For Eckart variants,
the next useful comparison is not another small increase from `switch_force=0.05`;
the early unit-mass runs show that the internal force scale is order `1..10`.
The `switch_force=10.0` jobs should tell us whether the projected Newton branch
can polish an index-1 LJ structure once it is actually allowed to run.
