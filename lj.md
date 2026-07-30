# Lennard-Jones LJ7 First-Order GAD Notes

These notes intentionally document only the current first-order LJ experiments.
Older hybrid/Newton and pre-Gaussian-start experiments are outdated.

## Setup

Transition-state candidate criterion:

- `final_n_neg == 1`
- `final_force_max < 0.05`

Current LJ runner defaults and conventions:

- Analytical LJ derivatives only.
- `--start-from gaussian_origin`
- `--gaussian-origin-sigma 1.0`
- LJ7 with `epsilon=1`, `sigma=1`
- `--atomic-number 1` for unit-mass Eckart projection
- `--method gad`
- `--high-index-descent gradient`
- `--use-projection`
- `--force-criterion fmax`
- `--force-threshold 0.05`
- `--k-track 0` (mode tracking off; always flip along the lowest Eckart eigenvector)
- `--lj-oscillator-mode off` by default (pure pairwise LJ; no tether)
- legacy `--lj-oscillator` is an alias for `--lj-oscillator-mode linear`
- `--no-lj-compile` by default (`torch.compile` only activates on CUDA)

Reference/equilibrium sample data copied from the adjacent sampling repo lives
under `data/`, which is gitignored:

- source directory: `../adjoint_sampling/data`
- copied files: `data/test_split_LJ13-1000.npy` and
  `data/test_split_LJ55-1000-part1.npy`
- `data/test_split_LJ13-1000.npy` currently loads as flattened LJ13
  coordinates with shape `(10000, 39)`
- `data/test_split_LJ55-1000-part1.npy` currently loads as flattened LJ55
  coordinates with shape `(10000, 165)`

The dynamics are first order only:

- Use projected Eckart GAD when `n_neg <= 1`.
- Use projected Eckart gradient descent when `n_neg > 1`.
- No Newton or curvature-scaled step is used.

## 1000-Step Sweep

Output root:

`runs/lj_gaussian_origin_projected_firstorder_dt_sweep`

All rows use `n_samples=100`, `n_steps=1000`, `start_from=gaussian_origin`, and
`gaussian_origin_sigma=1.0`.

| Output directory | `dt` | TS candidates | Index-1 finals | Final index pattern | Median converged step | Mean converged step |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `dt0005` | `0.0005` | 0/100 | 4 | `{1: 4, 2: 10, 3: 9, 4: 23, 5: 23, 6: 16, 7: 11, 8: 4}` | n/a | n/a |
| `dt001` | `0.001` | 5/100 | 17 | `{1: 17, 2: 14, 3: 16, 4: 21, 5: 16, 6: 11, 7: 2, 8: 3}` | 840 | 848 |
| `dt002` | `0.002` | 20/100 | 43 | `{1: 43, 2: 26, 3: 9, 4: 10, 5: 8, 6: 3, 7: 1}` | 590 | 629 |
| `dt003` | `0.003` | 42/100 | 59 | `{1: 59, 2: 26, 3: 5, 4: 8, 5: 1, 7: 1}` | 683 | 631 |
| `dt005` | `0.005` | 61/100 | 77 | `{1: 77, 2: 16, 3: 4, 4: 2, 5: 1}` | 476 | 492 |
| `dt007` | `0.007` | 70/100 | 79 | `{1: 79, 2: 19, 3: 1, 4: 1}` | 360 | 408 |

Runtime for one 100-start, 1000-step CPU job was about 2.5-6 minutes on
`cpu_short`.

## Larger-dt and 2000-Step Follow-Up

Larger time steps at 1000 max steps:

| Output directory | `dt` | `n_steps` | TS candidates | Index-1 finals | Final index pattern |
| --- | ---: | ---: | ---: | ---: | --- |
| `dt010` | `0.01` | 1000 | 0/100 | 79 | `{1: 79, 2: 2, 3: 18, 4: 1}` |
| `dt015` | `0.015` | 1000 | 0/100 | 77 | `{0: 1, 1: 77, 2: 1, 3: 19, 4: 2}` |
| `dt020` | `0.02` | 1000 | 0/100 | 79 | `{0: 1, 1: 79, 2: 1, 3: 17, 4: 2}` |

Longer 2000-step budget:

| Output directory | `dt` | `n_steps` | TS candidates | Index-1 finals | Final index pattern | Median converged step | Mean converged step |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dt007_n2000` | `0.007` | 2000 | 73/100 | 87 | `{1: 87, 2: 12, 3: 1}` | 380 | 443 |
| `dt010_n2000` | `0.01` | 2000 | 0/100 | 80 | `{1: 80, 2: 1, 3: 19}` | n/a | n/a |
| `dt015_n2000` | `0.015` | 2000 | 0/100 | 77 | `{0: 1, 1: 77, 2: 1, 3: 19, 4: 2}` | n/a | n/a |
| `dt020_n2000` | `0.02` | 2000 | 0/100 | 79 | `{0: 1, 1: 79, 2: 1, 3: 17, 4: 2}` | n/a | n/a |

Current best setting:

`dt=0.007` with `n_steps=2000`, giving 73/100 TS candidates.

## Best-Setting Ablations (`dt=0.007`, `n_steps=2000`)

Repeated the best setting on `cpu_short` with 100 starts to compare LJ backend
options. All rows match the sweep conventions above unless noted.

| Output directory | Variant | TS candidates | Index-1 finals | Final index pattern | Slurm elapsed | Total wall time |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `dt007_n2000_baseline` | defaults | 73/100 | 87 | `{1: 87, 2: 12, 3: 1}` | 3:33 | 183 s |
| `dt007_n2000_ljcompile` | `--lj-compile` | 73/100 | 87 | `{1: 87, 2: 12, 3: 1}` | 4:20 | 170 s |
| `dt007_n2000_noosc` | `--no-lj-oscillator` | 73/100 | 87 | `{1: 87, 2: 12, 3: 1}` | 4:21 | 171 s |
| `dt007_n2000_oscillator` | `--lj-oscillator` | 1/100 | 96 | `{0: 4, 1: 96}` | 7:28 | 434 s |

Notes:

- `baseline` and `--no-lj-oscillator` are the same physics; the default already
  disables the harmonic tether.
- `--lj-compile` has no effect on `cpu_short` (compile path requires CUDA), so
  the small timing differences are node noise rather than a speedup.
- `--lj-oscillator` (linear mode, `k=1`) adds a harmonic tether toward the COM.
  It reaches index 1 more often but leaves large residual forces, so TS success
  collapses to 1/100 and runtime roughly doubles.

## Oscillator Modes (`dt=0.007`, `n_steps=2000`)

Output root:

`runs/lj_gaussian_origin_projected_firstorder_dt_sweep/oscillator_modes`

Implementation lives in `src/gadplus/calculator/lj_oscillator.py`. All length
thresholds are expressed in units of the cluster scale
`r_cluster = r_eq * N^(1/3)`.

| Mode | Energy term |
| --- | --- |
| `off` | none (pure LJ) |
| `linear` | `k Σ \|r_i\|²` toward COM (legacy `--lj-oscillator`) |
| `deadzone` | `k Σ relu(\|r_i\| - R₀)²`, `R₀ = r0 * r_cluster` |
| `pair` | `k Σ_{i<j} relu(d_ij - r_cut)²`, `r_cut = rcut * r_cluster` |
| `switch` | `σ((d_max - d_on)/w) ×` linear, `d_on = r0 * r_cluster`, `w = switch_width * r_eq` |
| `quartic` | `k Σ \|r_i\|⁴` toward COM |

CLI (`scripts/lj_runner.py`):

```bash
--lj-oscillator-mode {off,linear,deadzone,pair,switch,quartic}
--lj-oscillator-scale 1.0
--lj-oscillator-r0 1.0          # deadzone/switch threshold in r_cluster units
--lj-oscillator-rcut 1.0        # pair cutoff in r_cluster units
--lj-oscillator-switch-width 0.3  # switch sigmoid width in r_eq units
```

All rows use `n_samples=100`, `start_from=gaussian_origin`,
`gaussian_origin_sigma=1.0`, and the best-setting dynamics above.

| Config | TS candidates | Median `d_max` | Wall (s) |
| --- | ---: | ---: | ---: |
| **No oscillator (baseline)** | **73/100** | 2.39 | 183 |
| `pair`, `k=1.0` | **93/100** | 2.21 | 98 |
| `pair`, `k=0.5` | **86/100** | 2.21 | 129 |
| `deadzone`, `k=1.0` | 83/100 | 2.22 | 168 |
| `deadzone`, `k=0.5` | 82/100 | 2.22 | 104 |
| `linear`, `k=0.1` | 71/100 | 2.21 | 111 |
| `quartic`, `k=0.1` | 52/100 | 2.14 | 167 |
| `switch`, `k=0.5` | 28/100 | 2.21 | 219 |
| `linear`, `k=1.0` | 1/100 | 2.13 | 267 |

**0/100 dissociated** on all runs (none exceeded `2 r_eq N^(1/3)`).

Re-run example (`pair`, `k=1`):

```bash
sbatch --partition=cpu_short --job-name=lj_osc_pair10 --cpus-per-task=1 --mem=2G --time=1:00:00 scripts/run_batch_cpu_uv.sbatch python -u scripts/lj_runner.py --method gad --high-index-descent gradient --use-projection --atomic-number 1 --dt 0.007 --force-threshold 0.05 --force-criterion fmax --n-samples 100 --n-steps 2000 --start-from gaussian_origin --gaussian-origin-sigma 1.0 --lj-oscillator-mode pair --lj-oscillator-scale 1.0 --output-dir runs/lj_gaussian_origin_projected_firstorder_dt_sweep/oscillator_modes/pair_k10
```

### Why some modes find more TS than no oscillator

The TS criterion is conjunctive: `final_n_neg == 1` **and** `final_force_max < 0.05`.
Reaching index 1 is necessary but not sufficient. On the no-oscillator baseline,
87/100 trajectories end at index 1 but only 73/100 also satisfy the force
threshold — 14 runs stall as “index-1 with large residual forces.”

Gaussian-origin starts (`σ=1`) produce diffuse, overextended clusters. Pure LJ
has weak restoring forces at large separations (the pair potential and its
gradient both decay as `r⁻⁷` and `r⁻¹³`), so early high-index gradient descent
can linger in elongated geometries with large `d_max` before the trajectory
finds a rearrangement channel. That shows up in the baseline median
`d_max = 2.39` versus `≈2.21` for the best tethered runs.

The tether modes split into two classes:

**Contact-gated springs (`pair`, `deadzone`) — help TS yield.**

- **`pair`** only penalizes atom pairs with `d_ij > r_cut`. On a compact LJ7
  geometry (e.g. pentagonal bipyramid), every contact is below cutoff, so the
  tether contributes **zero energy, forces, and Hessian** at the target basin.
  The LJ saddle is unchanged locally; TS forces can still vanish.
- During the approach from a Gaussian cloud, many pair distances exceed
  `r_cut ≈ r_eq N^(1/3)`. The spring gently pulls overstretched contacts
  together without imposing a global COM bias. Trajectories compact faster,
  spend fewer steps in high-`d_max` wandering, and more often arrive at an
  index-1 point with `fmax < 0.05` within the 2000-step budget. Wall time
  drops accordingly (98 s vs 183 s for `pair k=1`).
- **`deadzone`** is the radial analogue: inactive for atoms with
  `‖r_i‖ < R₀`, active only for outliers far from the COM. It gives a modest
  gain (82–83/100) but is less targeted than `pair` — it cannot distinguish a
  legitimately peripheral atom in a compact cluster from an atom pulled away by
  global elongation.

**Global COM tethers (`linear`, `switch`, `quartic`) — usually hurt.**

- These add restoring forces toward the origin (or a sigmoid-weighted version)
  even near compact geometries. The LJ transition state is **not** a minimum of
  `k Σ ‖r_i‖²` (or `‖r_i‖⁴`); the extra harmonic term does not cancel at the
  saddle, so trajectories can sit at index 1 with large spurious forces.
  `linear k=1` is the extreme case: 96/100 index-1 finals but only 1/100 TS.
- **`switch`** tries to gate the linear tether on large `d_max`, but any
  nonzero weight still injects COM-biased forces and curvature into the
  vibrational Hessian near the saddle, which depresses yield (28/100).
- Weak linear/quartic (`k=0.1`) perturb the saddle less but still provide no
  contact-specific guidance; yields match or fall below baseline.

**Summary:** `pair` wins because it solves the Gaussian-start problem (overextended
contacts) without modifying the local TS landscape. It converts index-1
near-misses into true TS candidates and shortens trajectories. Global tethers
trade easy index-1 access for incorrect forces at the saddle.

## Interpretation

The step size matters strongly for Gaussian-origin LJ7 starts. Small `dt`
values spend too much of the 1000-step budget in high-index regions. Increasing
`dt` up to `0.007` improves both convergence rate and speed among converged
trajectories. Larger steps, starting at `dt=0.01`, often reach index 1 but keep
large residual forces, so they do not satisfy the TS criterion. Extending the
budget from 1000 to 2000 steps improves `dt=0.007` from 70/100 to 73/100.

For the LJ backend, prefer `--lj-oscillator-mode pair` with `k≈1` at the best
`dt`/`n_steps` setting. Pure LJ (`mode off`) is a reasonable baseline but leaves
~14% of index-1 finals with excessive residual forces. Global linear/switch
tethers still interfere with saddle forces even when index 1 is reached easily;
contact-gated `pair` and `deadzone` modes avoid that by being inactive on compact
geometries.
