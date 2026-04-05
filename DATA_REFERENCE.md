# GAD_plus Data Reference

> Complete reference to all experiment data, Parquet schemas, DuckDB queries, and plotting recipes. Use this to reproduce any analysis or generate any figure.

## Data Layout

All data lives under `/lustre07/scratch/memoozd/gadplus/runs/`. Each experiment outputs Parquet files queryable via DuckDB.

```
/lustre07/scratch/memoozd/gadplus/runs/
├── sweep_dt/                      # Phase 1: dt × k_track parameter sweep
│   └── sweep_dt_results.parquet   # 150 rows (10 samples × 15 configs)
│
├── noise_survey_300/              # Phase 2: noise robustness (300 samples, FINAL)
│   ├── summary_noise_*pm_*.parquet  # 9 files, 300 rows each (one per noise level)
│   └── traj_*.parquet               # 2700 files (one per sample × noise level)
│
├── noise_survey/                  # Phase 2 pilot (50 samples, superseded)
│   ├── summary_noise_*pm_*.parquet  # 9 files, 50 rows each
│   ├── traj_*.parquet               # 450 files
│   └── plots/                       # Phase 4 trajectory plots
│       ├── fast_convergence.png
│       ├── slow_convergence.png
│       └── failure.png
│
├── starting_geom_300/             # Phase 3: starting geometry (300 samples, FINAL)
│   ├── summary_start_*.parquet      # 4 files (noised_ts, reactant, product, midpoint)
│   └── traj_*.parquet               # 1200 files
│
├── starting_geom/                 # Phase 3 pilot (50 samples, superseded)
│   ├── summary_start_*.parquet
│   └── traj_*.parquet
│
├── irc_validation/                # Phase 5: IRC validation (10 samples)
│   └── irc_validation_10pm.parquet  # 10 rows
│
├── basin_map/                     # Phase 6: basin mapping (50 samples, FINAL)
│   └── basin_map_results.parquet    # 350 rows (50 samples × 7 noise levels)
│
├── method_comparison/             # Phase 7 PARTIAL: crashed at ping-pong (5 of 7 methods)
│   └── traj_*.parquet               # 1504 trajectory files (5 of 7 methods completed)
│
├── method_cmp_300/                # Phase 7 FINAL: 7 methods × 6 noise (300 samples, 1000 steps)
│   ├── summary_{method}_{noise}pm.parquet  # 42 files (one per method×noise)
│   └── traj_*.parquet                       # trajectory files per sample
│
├── sweep_dt/                      # Phase 1
│   └── sweep_dt_results.parquet
│
├── param_range/                   # Earlier param sweep (from RESULTS_2026-04-03.md)
│   └── param_*.parquet
│
└── pure_gad_sweep/                # Earlier Level 0 sweep (from RESULTS_2026-04-03.md)
    ├── summary_*.parquet            # 14 files
    └── traj_*.parquet               # 700 files
```

## Parquet Schemas

### Summary Schema (20 columns)

One row per sample per experiment. Used for convergence rate analysis.

| Column | Type | Description |
|--------|------|-------------|
| run_id | string | Unique run identifier (noise level + UUID) |
| sample_id | int64 | Index into Transition1x dataset |
| formula | string | Molecular formula (e.g., "C2H2N2O2") |
| rxn | string | Reaction SMILES |
| noise_angstrom | double | Gaussian noise std (Angstrom) |
| noise_pm | int64 | Noise in picometers (noise_angstrom × 1000) |
| start_method | string | Starting geometry type (e.g., "noised_ts_10pm") |
| search_method | string | Search algorithm (e.g., "gad_projected") |
| dt | double | Timestep used |
| k_track | int64 | Mode tracking window |
| n_steps | int64 | Max steps allowed |
| **converged** | **bool** | **True if n_neg==1 AND force_norm<0.01** |
| converged_step | double | Step at which convergence occurred (NaN if failed) |
| total_steps | int64 | Actual steps taken |
| final_n_neg | int64 | Number of negative Hessian eigenvalues at final step |
| final_force_norm | double | Force norm at final step (eV/A) |
| final_energy | double | Energy at final step (eV) |
| final_eig0 | double | Lowest vibrational eigenvalue at final step |
| wall_time_s | double | Wall clock time (seconds) |
| failure_type | null | Not populated (always null in current runs) |

### Trajectory Schema (40 columns)

One row per GAD step per sample. Used for trajectory plots and detailed analysis.

| Column | Type | Description |
|--------|------|-------------|
| run_id | string | Run identifier |
| sample_id | int32 | Dataset index |
| rxn | string | Reaction SMILES |
| formula | string | Molecular formula |
| start_method | string | Starting geometry type |
| search_method | string | Algorithm name |
| step | int32 | Step number (0-indexed) |
| phase | string | "gad" or "nr" |
| dt_eff | double | Effective timestep used this step |
| wall_time_s | double | Cumulative wall time |
| **energy** | **double** | **Potential energy (eV)** |
| **force_norm** | **double** | **Mean per-atom force norm (eV/A)** |
| force_rms | double | RMS force (eV/A) |
| **n_neg** | **int32** | **Number of negative Eckart-projected eigenvalues** |
| **eig0** | **double** | **Lowest vibrational eigenvalue** |
| **eig1** | **double** | **Second-lowest vibrational eigenvalue** |
| eig_product | double | eig0 × eig1 |
| bottom_spectrum | list\<double\> | 6 lowest vibrational eigenvalues |
| n_neg_0 .. n_neg_1e2 | int32 | Cascade: n_neg at thresholds 0, 1e-4, ..., 0.01 |
| band_neg_large .. band_pos_large | int32 | 5 eigenvalue magnitude bands |
| **mode_overlap** | **double** | **|dot(v_prev, v_current)|, eigenvector continuity** |
| mode_index | int32 | Index of selected eigenvector |
| eigvec_continuity | double | Subspace overlap metric |
| grad_v0_overlap | double | |grad · v0| / |grad| — bottleneck detector |
| grad_v1_overlap | double | |grad · v1| / |grad| |
| **disp_from_start** | **double** | **RMSD from starting geometry (A)** |
| disp_from_last | double | Displacement from previous step (A) |
| **dist_to_known_ts** | **double** | **RMSD to reference TS geometry (A)** |
| coords_flat | list\<float\> | Flattened coordinates (3N values) |

### Sweep DT Schema (12 columns)

| Column | Type | Description |
|--------|------|-------------|
| dt | double | Timestep tested |
| k_track | int64 | Mode tracking window tested |
| sample_id | int64 | Dataset index |
| formula | string | Molecular formula |
| converged | bool | Convergence flag |
| converged_step | double | Step at convergence |
| total_steps / final_n_neg / final_force_norm / final_energy / final_eig0 / wall_time_s | various | Same as summary |

### Basin Map Schema (12 columns)

| Column | Type | Description |
|--------|------|-------------|
| sample_id | int64 | Dataset index |
| formula | string | Molecular formula |
| noise_angstrom / noise_pm | double / int64 | Noise level |
| converged | bool | Convergence flag |
| converged_step | double | Step at convergence |
| **rmsd_to_original_ts** | **double** | **RMSD between converged TS and original TS (A)** |
| **same_ts** | **bool** | **True if RMSD < 0.1A** |
| final_n_neg / final_force_norm / final_energy / wall_time_s | various | Final state |

### IRC Schema (9 columns)

| Column | Type | Description |
|--------|------|-------------|
| sample_id | int64 | Dataset index |
| formula | string | Molecular formula |
| noise_pm | int64 | Noise level of the converged TS |
| **intended** | **bool** | **Both IRC endpoints match known R/P (RMSD < 0.3A)** |
| **half_intended** | **bool** | **Only one endpoint matches** |
| rmsd_reactant | double | Min RMSD from either IRC endpoint to known reactant |
| rmsd_product | double | Min RMSD from either IRC endpoint to known product |
| error | null | Error message (null if successful) |
| wall_time_s | double | IRC wall time |

---

## DuckDB Queries

### Setup

```python
import duckdb

# On Narval login nodes, set threading first:
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

RUNS = '/lustre07/scratch/memoozd/gadplus/runs'
```

### Phase 1: Parameter Sweep

```sql
-- Convergence rate by dt and k_track
SELECT dt, k_track,
       COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END)::INT as conv,
       ROUND(100.0 * conv / total, 1) as rate,
       ROUND(AVG(CASE WHEN converged THEN converged_step END), 0) as avg_steps
FROM '{RUNS}/sweep_dt/sweep_dt_results.parquet'
GROUP BY dt, k_track
ORDER BY rate DESC, avg_steps ASC;
```

### Phase 2: Noise Robustness (300 samples)

```sql
-- Convergence rate by noise level
SELECT noise_pm,
       COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END)::INT as conv,
       ROUND(100.0 * conv / total, 1) as rate,
       ROUND(AVG(CASE WHEN converged THEN converged_step END), 0) as avg_steps,
       ROUND(AVG(wall_time_s), 1) as avg_time
FROM '{RUNS}/noise_survey_300/summary_*.parquet'
GROUP BY noise_pm ORDER BY noise_pm;

-- Per-molecule difficulty: which formulas never converge?
SELECT formula,
       COUNT(*) as configs_tested,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END)::INT as conv,
       ROUND(100.0 * conv / configs_tested, 1) as rate
FROM '{RUNS}/noise_survey_300/summary_*.parquet'
GROUP BY formula ORDER BY rate ASC;

-- Distribution of final n_neg for failures
SELECT noise_pm, final_n_neg, COUNT(*) as cnt
FROM '{RUNS}/noise_survey_300/summary_*.parquet'
WHERE NOT converged
GROUP BY noise_pm, final_n_neg
ORDER BY noise_pm, final_n_neg;
```

### Phase 3: Starting Geometry (300 samples)

```sql
-- Convergence by starting geometry
SELECT start_method,
       COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END)::INT as conv,
       ROUND(100.0 * conv / total, 1) as rate,
       ROUND(AVG(CASE WHEN converged THEN converged_step END), 0) as avg_steps
FROM '{RUNS}/starting_geom_300/summary_*.parquet'
GROUP BY start_method ORDER BY rate DESC;

-- Which molecules converge from midpoint but not reactant?
SELECT a.sample_id, a.formula
FROM '{RUNS}/starting_geom_300/summary_*.parquet' a
WHERE a.start_method = 'midpoint' AND a.converged
  AND a.sample_id NOT IN (
    SELECT sample_id FROM '{RUNS}/starting_geom_300/summary_*.parquet'
    WHERE start_method = 'reactant' AND converged
  );
```

### Phase 5: IRC Validation

```sql
-- IRC results summary
SELECT
    SUM(CASE WHEN intended THEN 1 ELSE 0 END) as intended,
    SUM(CASE WHEN half_intended THEN 1 ELSE 0 END) as half,
    SUM(CASE WHEN NOT intended AND NOT half_intended THEN 1 ELSE 0 END) as unintended,
    ROUND(AVG(rmsd_reactant), 3) as avg_rmsd_R,
    ROUND(AVG(rmsd_product), 3) as avg_rmsd_P
FROM '{RUNS}/irc_validation/irc_validation_10pm.parquet';

-- Per-sample detail
SELECT sample_id, formula, intended, half_intended,
       ROUND(rmsd_reactant, 3) as rmsd_R,
       ROUND(rmsd_product, 3) as rmsd_P
FROM '{RUNS}/irc_validation/irc_validation_10pm.parquet'
ORDER BY sample_id;
```

### Phase 6: Basin Mapping

```sql
-- Basin stability by noise level
SELECT noise_pm,
       COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END)::INT as conv,
       SUM(CASE WHEN same_ts THEN 1 ELSE 0 END)::INT as same_ts,
       SUM(CASE WHEN converged AND NOT same_ts THEN 1 ELSE 0 END)::INT as diff_ts,
       ROUND(AVG(CASE WHEN converged THEN rmsd_to_original_ts END), 4) as avg_rmsd
FROM '{RUNS}/basin_map/basin_map_results.parquet'
GROUP BY noise_pm ORDER BY noise_pm;

-- Which samples find a different TS at 200pm?
SELECT sample_id, formula, rmsd_to_original_ts, final_energy
FROM '{RUNS}/basin_map/basin_map_results.parquet'
WHERE noise_pm = 200 AND converged AND NOT same_ts
ORDER BY rmsd_to_original_ts DESC;
```

### Trajectory Queries (for plots)

```sql
-- Single sample trajectory (energy, forces, eigenvalues vs step)
SELECT step, energy, force_norm, n_neg, eig0, eig1,
       mode_overlap, disp_from_start, dist_to_known_ts
FROM '{RUNS}/noise_survey_300/traj_*.parquet'
WHERE sample_id = 0 AND start_method LIKE '%10pm%'
ORDER BY step;

-- Find fast/slow/failed trajectories for plotting
SELECT run_id, sample_id, formula, start_method,
       converged, converged_step, total_steps
FROM '{RUNS}/noise_survey_300/summary_*.parquet'
WHERE converged
ORDER BY converged_step ASC LIMIT 5;  -- fastest

SELECT run_id, sample_id, formula, start_method,
       converged, converged_step, total_steps
FROM '{RUNS}/noise_survey_300/summary_*.parquet'
WHERE converged
ORDER BY converged_step DESC LIMIT 5;  -- slowest

-- Average trajectory curves (for aggregate plots)
SELECT noise_pm, step,
       AVG(energy) as avg_energy,
       AVG(force_norm) as avg_force,
       AVG(CAST(n_neg AS DOUBLE)) as avg_nneg,
       AVG(eig0) as avg_eig0,
       AVG(dist_to_known_ts) as avg_dist_ts
FROM '{RUNS}/noise_survey_300/traj_*.parquet' t
JOIN '{RUNS}/noise_survey_300/summary_*.parquet' s
  ON t.run_id = s.run_id AND t.sample_id = s.sample_id
WHERE s.converged
GROUP BY noise_pm, step
ORDER BY noise_pm, step;

-- Mode overlap distribution (detect mode crossings)
SELECT noise_pm,
       ROUND(AVG(mode_overlap), 3) as avg_overlap,
       ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY mode_overlap), 3) as p5_overlap,
       COUNT(CASE WHEN mode_overlap < 0.5 THEN 1 END) as n_crossings
FROM '{RUNS}/noise_survey_300/traj_*.parquet'
WHERE step > 0
GROUP BY noise_pm ORDER BY noise_pm;
```

### Level 0 vs Level 2 Comparison

```sql
-- Level 0 data (from earlier pure_gad_sweep)
SELECT start_method,
       COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END)::INT as conv,
       ROUND(100.0 * conv / total, 1) as rate
FROM '{RUNS}/pure_gad_sweep/summary_*.parquet'
GROUP BY start_method ORDER BY start_method;
```

### Phase 7: Method Comparison (300 samples, 1000 steps, 42 configs)

```sql
-- Convergence rate by method and noise
SELECT method, noise_pm,
       COUNT(*) as total,
       SUM(CASE WHEN converged THEN 1 ELSE 0 END) as conv,
       ROUND(100.0 * conv / total, 1) as rate,
       ROUND(AVG(CASE WHEN converged THEN converged_step END), 0) as avg_steps
FROM '{RUNS}/method_cmp_300/summary_*.parquet'
GROUP BY method, noise_pm
ORDER BY method, noise_pm;

-- Pivot table
SELECT * FROM (
    SELECT method, noise_pm,
           ROUND(100.0 * SUM(CASE WHEN converged THEN 1 ELSE 0 END) / COUNT(*), 1) as rate
    FROM '{RUNS}/method_cmp_300/summary_*.parquet'
    GROUP BY method, noise_pm
) PIVOT (MAX(rate) FOR noise_pm IN (10, 30, 50, 100, 150, 200));
```

Earlier partial run (crashed at ping-pong, 5 of 7 methods):
```sql
SELECT search_method, COUNT(DISTINCT sample_id) as samples
FROM '{RUNS}/method_comparison/traj_*.parquet'
GROUP BY search_method;
```

---

## Plotting Recipes

### Recipe 1: Convergence Rate vs Noise Level (Phase 2 main figure)

```python
import duckdb
import matplotlib.pyplot as plt

df = duckdb.execute("""
    SELECT noise_pm,
           ROUND(100.0 * SUM(CASE WHEN converged THEN 1 ELSE 0 END) / COUNT(*), 1) as rate
    FROM '/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/summary_*.parquet'
    GROUP BY noise_pm ORDER BY noise_pm
""").df()

plt.figure(figsize=(8, 5))
plt.plot(df['noise_pm'], df['rate'], 'o-', linewidth=2, markersize=8)
plt.xlabel('Noise (pm)')
plt.ylabel('Convergence rate (%)')
plt.title('GAD Projected (Level 2): Convergence vs Noise (300 samples)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.savefig('conv_vs_noise.png', dpi=150, bbox_inches='tight')
```

### Recipe 2: 2×2 Trajectory Plot (Phase 4)

```python
import duckdb
import matplotlib.pyplot as plt

# Pick a specific sample
run_id = '...'  # get from summary query
sample_id = 0

traj = duckdb.execute(f"""
    SELECT step, energy, force_norm, n_neg, eig0, eig1
    FROM '/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/traj_*.parquet'
    WHERE run_id = '{run_id}' AND sample_id = {sample_id}
    ORDER BY step
""").df()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(traj['step'], traj['energy'])
axes[0,0].set_ylabel('Energy (eV)')

axes[0,1].semilogy(traj['step'], traj['force_norm'])
axes[0,1].axhline(0.01, color='g', linestyle='--', label='threshold')
axes[0,1].set_ylabel('Force norm (eV/A)')
axes[0,1].legend()

axes[1,0].plot(traj['step'], traj['n_neg'], drawstyle='steps-post')
axes[1,0].axhline(1, color='g', linestyle='--', label='target')
axes[1,0].set_ylabel('n_neg')
axes[1,0].legend()

axes[1,1].plot(traj['step'], traj['eig0'], label='eig0')
axes[1,1].plot(traj['step'], traj['eig1'], label='eig1')
axes[1,1].axhline(0, color='k', alpha=0.3)
axes[1,1].set_ylabel('Eigenvalue')
axes[1,1].legend()

for ax in axes.flat:
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trajectory_2x2.png', dpi=150)
```

### Recipe 3: Basin RMSD vs Noise (Phase 6)

```python
df = duckdb.execute("""
    SELECT noise_pm, rmsd_to_original_ts, same_ts, converged
    FROM '/lustre07/scratch/memoozd/gadplus/runs/basin_map/basin_map_results.parquet'
    WHERE converged
""").df()

plt.figure(figsize=(8, 5))
same = df[df['same_ts']]
diff = df[~df['same_ts']]
plt.scatter(same['noise_pm'], same['rmsd_to_original_ts'], c='blue', label='Same TS', alpha=0.6)
plt.scatter(diff['noise_pm'], diff['rmsd_to_original_ts'], c='red', label='Different TS', alpha=0.6)
plt.axhline(0.1, color='k', linestyle='--', alpha=0.5, label='threshold')
plt.xlabel('Noise (pm)')
plt.ylabel('RMSD to original TS (A)')
plt.title('Basin Stability (50 samples)')
plt.legend()
plt.savefig('basin_rmsd.png', dpi=150, bbox_inches='tight')
```

### Recipe 4: Starting Geometry Bar Chart (Phase 3)

```python
df = duckdb.execute("""
    SELECT start_method,
           ROUND(100.0 * SUM(CASE WHEN converged THEN 1 ELSE 0 END) / COUNT(*), 1) as rate
    FROM '/lustre07/scratch/memoozd/gadplus/runs/starting_geom_300/summary_*.parquet'
    GROUP BY start_method ORDER BY rate DESC
""").df()

plt.figure(figsize=(8, 5))
plt.bar(df['start_method'], df['rate'], color=['#2196F3', '#4CAF50', '#FF9800', '#f44336'])
plt.ylabel('Convergence rate (%)')
plt.title('Convergence by Starting Geometry (300 samples)')
plt.ylim(0, 100)
plt.savefig('starting_geom.png', dpi=150, bbox_inches='tight')
```

### Recipe 5: Method Comparison Heatmap (Phase 7)

```python
import numpy as np

# Can now read directly from Parquet:
df = duckdb.execute("""
    SELECT method, noise_pm,
           ROUND(100.0 * SUM(CASE WHEN converged THEN 1 ELSE 0 END) / COUNT(*), 1) as rate
    FROM '/lustre07/scratch/memoozd/gadplus/runs/method_cmp_300/summary_*.parquet'
    GROUP BY method, noise_pm
""").df()
pivot = df.pivot_table(index='method', columns='noise_pm', values='rate')

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f'{n}pm' for n in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f'{pivot.values[i,j]:.0f}%', ha='center', va='center', fontweight='bold')
plt.colorbar(im, label='Convergence rate (%)')
plt.title('Method Comparison (300 samples, 1000 steps)')
plt.tight_layout()
plt.savefig('method_heatmap.png', dpi=150)
```

### Recipe 6: Average Convergence Trajectory (converged vs failed)

```python
# Average force_norm trajectory for converged vs failed runs at 50pm
for label, cond in [('converged', 'AND s.converged'), ('failed', 'AND NOT s.converged')]:
    df = duckdb.execute(f"""
        SELECT t.step, AVG(t.force_norm) as avg_force, AVG(CAST(t.n_neg AS DOUBLE)) as avg_nneg
        FROM '/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/traj_*.parquet' t
        JOIN '/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/summary_*.parquet' s
          ON t.run_id = s.run_id AND t.sample_id = s.sample_id
        WHERE s.noise_pm = 50 {cond}
        GROUP BY t.step ORDER BY t.step
    """).df()
    plt.plot(df['step'], df['avg_force'], label=label)
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel('Avg force norm (eV/A)')
plt.legend()
plt.title('Average force trajectory: converged vs failed (50pm, 300 samples)')
plt.savefig('avg_trajectory.png', dpi=150)
```

### Recipe 7: Distance to Known TS Over Time

```python
# For converged runs: how quickly do they approach the known TS?
df = duckdb.execute("""
    SELECT t.step, s.noise_pm, AVG(t.dist_to_known_ts) as avg_dist
    FROM '/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/traj_*.parquet' t
    JOIN '/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/summary_*.parquet' s
      ON t.run_id = s.run_id AND t.sample_id = s.sample_id
    WHERE s.converged AND s.noise_pm IN (10, 50, 100, 200)
    GROUP BY t.step, s.noise_pm ORDER BY s.noise_pm, t.step
""").df()

for noise in [10, 50, 100, 200]:
    sub = df[df['noise_pm'] == noise]
    plt.plot(sub['step'], sub['avg_dist'], label=f'{noise}pm')
plt.xlabel('Step')
plt.ylabel('RMSD to known TS (A)')
plt.legend()
plt.title('Approach to known TS (converged runs only)')
plt.savefig('dist_to_ts.png', dpi=150)
```

---

## SLURM Logs

All SLURM output at `/lustre07/scratch/memoozd/gadplus/logs/`:

| Pattern | Phase | Content |
|---------|-------|---------|
| `smoke_*.out` | Smoke test | HIP load, timing, basic checks |
| `sweep_dt_*.out` | Phase 1 | Per-sample results, summary table |
| `noise_*_*.out` | Phase 2 | Per-sample results per noise level |
| `startgeom_*_*.out` | Phase 3 | Per-sample results per start type |
| `irc_*.out` | Phase 5 | IRC forward/backward, RMSD to R/P |
| `basin_*.out` | Phase 6 | Per-sample per-noise RMSD, same/diff TS |
| `methodcmp_*_*.out` | Phase 7 | Per-sample results per method per noise |

## Key Job IDs

| Job | Phase | Description | Status |
|-----|-------|-------------|--------|
| 58833650 | 1 | dt sweep (10 test, 100 steps) | COMPLETED |
| 58835838_[0-8] | 2 | Noise survey 300 samples | COMPLETED |
| 58835839_[0-3] | 3 | Starting geom 300 samples | COMPLETED |
| 58834594 | 5 | IRC validation 10 samples | COMPLETED |
| 58835840 | 6 | Basin mapping 50 samples | COMPLETED |
| 58835900_[0-5] | 7 | Method comparison 50 samples | FAILED (ping-pong dtype) |
| 58845357_[0-41] | 7 | Method comparison 300 samples, 1000 steps | COMPLETED |

## Data Locations

| Experiment | Path |
|------------|------|
| Phase 1 (dt sweep) | `/lustre07/scratch/memoozd/gadplus/runs/sweep_dt/` |
| Phase 2 (noise 300) | `/lustre07/scratch/memoozd/gadplus/runs/noise_survey_300/` |
| Phase 3 (start geom 300) | `/lustre07/scratch/memoozd/gadplus/runs/starting_geom_300/` |
| Phase 5 (IRC) | `/lustre07/scratch/memoozd/gadplus/runs/irc_validation/` |
| Phase 6 (basin map) | `/lustre07/scratch/memoozd/gadplus/runs/basin_map/` |
| Phase 7 (method cmp, partial) | `/lustre07/scratch/memoozd/gadplus/runs/method_comparison/` |
| Phase 7 (method cmp, FINAL) | `/lustre07/scratch/memoozd/gadplus/runs/method_cmp_300/` |

## Dataset Reference

```python
from gadplus.data.transition1x import Transition1xDataset, UsePos

dataset = Transition1xDataset(
    '/lustre06/project/6033559/memoozd/data/transition1x.h5',
    split='train',  # 9561 samples. Also: 'test' (287), 'validation' (225)
    max_samples=300,
    transform=UsePos('pos_transition'),
)
# Each sample has: .pos (N,3), .z (N,), .pos_reactant (N,3),
#                  .pos_product (N,3), .formula, .rxn
```
