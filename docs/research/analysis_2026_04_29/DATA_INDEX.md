# DATA_INDEX — query guide

Every artifact this benchmark produces, with its schema and an example
DuckDB query. Goal: any number in the IRC_TEST report should be reproducible
in one query against one of these files.

## Layout

```
/lustre07/scratch/memoozd/gadplus/runs/        # all parquets (per-sample)
  test_set/{gad_*, sella_*}/                     # 2k step canonical sweep
  test_dtgrid/gad_dt00{3..8}_fmax/               # 5k step GAD dt grid
  test_lowdt/gad_dt{001,0005,0001}_fmax/         # very-small-dt diagnostic
  test_set/sella_*_nohess/                       # Sella with no Hessian (timed out)
  test_hessfreq/sella_carteck_libdef_d{3,5,10,25}/   # Hessian-frequency sweep
  test_reactant/{gad_*,sella_*}/                 # from-reactant single-ended
  test_irc/*/                                    # IRC validation outputs
  test_sella_trajlog/carteck_libdef/             # Sella per-step trajectories on test
  test_nrpolish/nr_gad_polish_dt007_*/           # NR-GAD polish sweep
  sella_trajlog/carteck_*/                       # Sella per-step trajectories on TRAIN

/lustre06/project/6033559/memoozd/GAD_plus/analysis_2026_04_29/
  threshold_sweep.csv             # method × noise × threshold grid
  dynamics_curves.csv             # per (method, noise, step): median+IQR fmax/fnorm
  dynamics_crossings.csv          # per sample: first-crossing step + plateau-step
  test_summary_full.csv           # final-state summary per (method, noise, sample)
  saddle_quality_table.md         # narrative breakdown of "saddle vs converged"
  sella_nohess_partial.csv        # parsed from timed-out logs
  gad_test_rmsd.csv               # GAD final RMSD-to-known-TS
  FINDINGS.md                     # narrative findings, with sources
  DATA_INDEX.md                   # this file
```

---

## Schemas (analysis CSVs)

### `threshold_sweep.csv`

One row per (method, noise, threshold). Built by `scripts/analyze_threshold_sweep.py`.

| column | type | meaning |
|---|---|---|
| `method` | str | display name, e.g. `"GAD dt=0.007 (5k)"`, `"Sella libdef"` |
| `noise_pm` | int | TS noise in pm: {10, 30, 50, 100, 150, 200} |
| `n_total` | int | number of samples |
| `threshold` | float | force threshold in eV/Å: {0.05, 0.01, 0.005, 0.001, 0.0001} |
| `conv_fmax_pct` | float | % of samples with `n_neg=1 ∧ fmax<threshold` |
| `conv_fnorm_pct` | float | % with `n_neg=1 ∧ ‖F‖_mean<threshold` |
| `conv_fmax_nosad_pct` | float | % with `fmax<threshold` (no saddle req — Sella's library default) |
| `conv_fnorm_nosad_pct` | float | % with `‖F‖_mean<threshold` (no saddle req) |
| `median_steps` | float | median total steps for this (method, noise) |
| `median_wall_s` | float | median wall time per attempt |
| `median_calls` | float | median predict_fn calls per attempt |

**Example:** "What's GAD dt=0.007's conv rate at fmax<0.01 across noise?"
```sql
SELECT noise_pm, conv_fmax_pct
FROM read_csv_auto('threshold_sweep.csv')
WHERE method='GAD dt=0.007 (5k)' AND threshold=0.01
ORDER BY noise_pm;
```

### `dynamics_curves.csv` (134k rows)

One row per (method, noise_pm, step). Method names are comma-free for
duckdb auto-detect compatibility. Built by `scripts/analyze_dynamics.py`.

| column | type | meaning |
|---|---|---|
| `method` | str | one of: `gad_dt003_5k, gad_dt005_5k, gad_dt007_5k, gad_adaptive_dt, gad_dt001_20k, sella_libdef_train_trajlog, sella_libdef_test_trajlog` |
| `noise_pm` | int | TS noise (pm) |
| `step` | int | optimizer step |
| `fmax_p25/p50/p75` | float | quantile of fmax across 30 samples at this step |
| `fnorm_p50` | float | median ‖F‖_mean across 30 samples |

**Example:** "When does GAD dt=0.007 at 100pm cross fmax=0.01?"
```sql
SELECT MIN(step) FROM read_csv_auto('dynamics_curves.csv')
WHERE method='gad_dt007_5k' AND noise_pm=100 AND fmax_p50<0.01;
```

### `dynamics_crossings.csv` (930 rows)

One row per (method, noise_pm, sample_idx). Per-sample first-crossing
step for each of {0.05, 0.01, 0.005, 0.001} on both fmax and fnorm.

| column | meaning |
|---|---|
| `method`, `noise_pm`, `sample_idx` | id |
| `cross_fmax_0.05` … `cross_fmax_0.001` | first step where fmax<thresh; NaN if never |
| `cross_fnorm_0.05` … `cross_fnorm_0.001` | same for ‖F‖_mean |
| `plateau_step` | first step where `dlog(fmax)/dstep<1e-3` over 100-step window |
| `final_fmax` | fmax at end of trajectory |

### `test_summary_full.csv`

One row per (method, noise_pm, sample_id). Built by `scripts/analyze_rmsd_bimodal.py`.

| column | meaning |
|---|---|
| `method`, `noise_pm`, `sample_id` | id |
| `final_n_neg`, `final_fmax`, `final_force_norm` | final-state metrics |
| `total_steps`, `wall_time_s`, `n_calls` | compute |
| `rmsd_to_ts` | Kabsch+Hungarian RMSD (Å) from final coords to known TS |
| `is_close`, `is_far` | RMSD bin flags ({<0.05}, {>0.5}) |

### `gad_test_rmsd.csv`

Subset of test_summary_full restricted to GAD methods and computed independently.

### `sella_nohess_partial.csv`

Parsed from timed-out slurm logs. Built by `scripts/parse_nohess_logs.py`.

| column | meaning |
|---|---|
| `method` | `carteck_nohess` or `internal_nohess` |
| `noise_pm` | int |
| `n_completed` | samples that finished before 12h wall (out of 287) |
| `n_conv` | of completed, how many converged |
| `conv_pct_partial` | n_conv / n_completed × 100 |
| `conv_pct_lb` | n_conv / 287 × 100 (lower bound assuming unprocessed all failed) |
| `n_ours_TS`, `ours_TS_pct_partial` | same for `n_neg=1 ∧ force<0.01` criterion |
| `median_steps`, `median_wall_s`, `median_fmax` | compute summary |

---

## Schemas (parquet, summary)

`summary_<method>_<noise>pm.parquet` — one row per sample.

Common columns:
- `sample_id`, `formula`, `noise_pm`, `method`
- `final_n_neg`, `final_fmax` (or `final_force_max`), `final_force_norm`, `final_energy`
- `total_steps`, `wall_time_s`, `n_func_evals` (Sella)
- `converged` (boolean: `n_neg=1 ∧ fmax<0.01` for GAD; same plus `sella_converged` for Sella)
- For Sella: `is_nneg1`, `is_fmax_001`, `conv_nneg1_fmax001`, `conv_sella_and_nneg1` etc.

**Example:** "Get all converged GAD dt=0.007 samples at 100pm"
```sql
SELECT sample_id, total_steps, final_force_max
FROM 'runs/test_dtgrid/gad_dt007_fmax/summary_gad_dt007_fmax_100pm.parquet'
WHERE converged = true;
```

## Schemas (parquet, traj)

`traj_<method>_<noise>pm_<run_id>_<sample_id>.parquet` — one row per step.

Common columns:
- `step`, `phase` (`gad`, `nr`, `descent`)
- `energy`, `force_max`, `force_norm`
- `n_neg`, `eig0`, `eig1`, `bottom_spectrum`
- `n_neg_0`, `n_neg_1e4`, …, `n_neg_1e2` (cascade)
- `mode_overlap`, `eigvec_continuity`
- `disp_from_start`, `dist_to_known_ts`
- `coords_flat` (3N floats)

For Sella trajlog: `step, sample_id, formula, energy, force_max, force_norm, disp_from_last, delta_trust, coords_flat`.

**Example:** "Plot fmax vs step for sample 0 of GAD dt=0.007 at 10pm"
```sql
SELECT step, force_max
FROM 'runs/test_dtgrid/gad_dt007_fmax/traj_gad_dt007_fmax_10pm_*_0.parquet'
ORDER BY step;
```

## IRC validation parquet

`irc_validation_sella_hip_allendpoints_<noise>pm.parquet` — one row per sample.

| column | meaning |
|---|---|
| `sample_id`, `formula`, `noise_pm` | id |
| `source_gad_converged` | did the source method converge before IRC? |
| `intended` | both forward+reverse RMSD-match R/P |
| `half_intended` | one matches |
| `topology_intended` | both forward+reverse bond-graph match R/P |
| `topology_half_intended` | one matches |
| `forward_rmsd_reactant`, `forward_rmsd_product`, … | Kabsch RMSDs |
| `forward_n_neg_vib`, `forward_min_vib_eig` | Eckart-projected vib spectrum at forward endpoint |
| `wall_time_s` | IRC wall time per sample |

**Example:** "TOPO-intended rate per noise level for GAD dt=0.007"
```sql
SELECT noise_pm, AVG(CAST(topology_intended AS DOUBLE)) * 100 AS topo_pct
FROM 'runs/test_irc/gad_dt007_fmax/irc_validation_sella_hip_allendpoints_*pm.parquet'
GROUP BY noise_pm
ORDER BY noise_pm;
```

---

## Conventions

- **All file paths absolute.** `/lustre07/scratch/...` for parquet runs, `/lustre06/.../GAD_plus/...` for code + analysis.
- **All cell labels:** `noise_pm` always in pm (10, 30, 50, 100, 150, 200), never as fraction.
- **Force values in eV/Å.** `fmax = max_atom |F_a|`, `force_norm = mean |F_a|`.
- **n_neg always Eckart-projected** (vibrational eigenvalues, no T/R modes).
- **`converged`** column: GAD = `n_neg=1 ∧ fmax<0.01` (loose); paper-strict adds `force_threshold=1e-4` configs separately.
- **Method labels in `dynamics_curves.csv` and `dynamics_crossings.csv` are comma-free** for DuckDB auto-quote compatibility.
