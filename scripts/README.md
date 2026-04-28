# Scripts inventory

Last cleanup: 2026-04-20.

## Canonical scripts (use these)

These produce the data backing the current `IRC_COMPREHENSIVE_2026-04-20.pdf`.

### TS-finding (one method × one noise → summary parquet + traj parquets)

| Slurm | Method | Output | Notes |
|---|---|---|---|
| `run_gad_eckart_fmax.slurm` | `gad_dt003_fmax` (Eckart, fmax<0.01) | `runs/gad_eckart_fmax/` | Canonical GAD Eckart with matched fmax gate |
| `run_gad_no_eckart.slurm` | `gad_dt003_no_eckart` (raw H, fmax<0.01) | `runs/gad_no_eckart/` | Raw-Hessian GAD baseline |
| `run_sella_2000_both.slurm` | Sella cart+Eckart **and** internal | `runs/sella_2000/` | 2-config × 6-noise array, 2000 steps, coords saved |
| `run_sella_2000_cart_no_eckart.slurm` | Sella cartesian no-Eckart | `runs/sella_2000/` | 6-noise array, 2000 steps, coords saved |

Underlying drivers (don't edit, called by the slurms):
- `method_single.py` — GAD-style method runner. Method configs in METHOD_CONFIGS dict at top.
- `sella_baseline.py` — Sella runner. `--internal/--cartesian`, `--apply-eckart`, `--max-steps`, `--fmax`. Saves `coords_flat`+`atomic_nums` (added 2026-04-17).

### IRC validation (sella_hip on a TS set → IRC parquet)

| Slurm | Source TSs | Output |
|---|---|---|
| `run_irc_gad_fmax.slurm` | GAD Eckart fmax (`runs/gad_eckart_fmax/`) | `runs/irc_gad_eckart_fmax/` |
| `run_irc_new_methods.slurm` | 4 new TS sets in one 23-task array (GAD no-Eckart, Sella 2000 × 3 configs) | `runs/irc_*/` (4 different dirs) |
| `run_irc_sella_int_200.slurm` | Sella internal 200pm (the one that was running when the array submitted) | `runs/irc_sella_int_2000/` |
| `run_irc_sellahip_allendpoints.slurm` | GAD Eckart force_norm, all endpoints | `runs/irc_sellahip_allendpoints/` |
| `run_irc_sellahip_full.slurm` | GAD Eckart force_norm, converged-only | `runs/irc_sellahip_full/` |
| `run_irc_sellahip_on_sella_allep.slurm` | Sella 1000-step, all endpoints (historical) | `runs/irc_sellahip_on_sella_allep/` |
| `run_irc_sellahip_on_sella.slurm` | Sella 1000-step, converged-only (historical) | `runs/irc_sellahip_on_sella/` |
| `run_irc_rigorous_full.slurm` | Rigorous IRC on GAD Eckart (parked, see backburner) | `runs/irc_rigorous_full/` |

Underlying driver:
- `irc_validate.py` — handles GAD-style traj coord lookup and Sella-style summary coord lookup via `--coords-source {traj,summary}`. Supports `--all-endpoints` to drop the converged filter.

## Analysis & figures

| Script | Output |
|---|---|
| `analyze_irc.py` | Per-(method,noise) summary table; quick smoke test of any IRC dir |
| `analyze_sella_deep.py` | A/B/C/D/E/F/G/H deep dive on one IRC dataset |
| `analyze_full_2026_04_20.py` | 5-method comparison numbers (TS + IRC) and main figures |
| `figures_bars_generic.py` | `<input_dir> <output_prefix>` → 3 bar charts (topo/rmsd/endpoint) |
| `figures_master_2026_04_20.py` | Per-method conv-line + IRC-bar figures; cluttered comparison line charts |
| `figures_comprehensive.py` | Earlier 04-17 figures (still used in the 04-17 PDF) |
| `figures_sella_bars.py` | First per-method bar charts (April 17 era) |
| `figures_sella_irc.py` | Earlier comprehensive figure set |

## One-off / retry scripts (kept for record, not for general use)

These were reactive fixes during the sweeps; not canonical.

| Slurm | Why it exists |
|---|---|
| `run_gad_finish_200pm.slurm` | One-shot 200pm refill after the original sweep timed out at 259/300 |
| `run_sella_2000_cartecktrail.slurm` | Sella cart+Eckart 200pm retry after 8h budget timeout (16h + 48GB) |
| `run_sella_2000_internal_highmem.slurm` | Sella internal 100/150/200pm retry after 16GB OOM (48GB) |
| `run_sella_2000_int_200_96gb.slurm` | Sella internal 200pm retry after 48GB *also* OOM'd (96GB) |
| `run_sella_rerun_coords.slurm` | First Sella rerun-with-coords; superseded by `run_sella_2000_both.slurm` |

## Historical (Round 1 / 2 / 3 era — not actively maintained)

These produced data that pre-dates the unified 5-method comparison. Kept for reproducibility of older runs.

`run_basin_map.slurm`, `run_geodesic_irc.slurm`, `run_method_cmp_300.slurm`, `run_method_comparison.slurm`,
`run_multimode.slurm`, `run_narval.slurm`, `run_noise_survey.slurm`, `run_nr_then_gad_quick.slurm`,
`run_nr_then_gad.slurm`, `run_param_range.slurm`, `run_precond_gad.slurm`, `run_round2.slurm`,
`run_round3.slurm`, `run_round3_extra.slurm`, `run_round4_quick.slurm`, `run_sella_1000.slurm`,
`run_sella_baselines.slurm`, `run_smoke_test.slurm`, `run_smoke_test_trillium.slurm`,
`run_starting_geom.slurm`, `run_sweep_dt.slurm`, `run_targeted_tests.slurm`,
`run_irc_validate.slurm`, `run_irc_validate_three_way.slurm`, `run_irc_validate_trust_test.slurm`,
`run_irc_smoke.slurm`, `run_irc_sellahip_round3.slurm`, `run_flagship_visualizations_cpu.slurm`.

## Smoke test recipes

For the IRC pipeline (uses GAD Eckart 10pm converged TSs):
```bash
sbatch scripts/run_irc_smoke.slurm
```
Should produce a 4-task array completing in <2 min (10pm samples are very fast).

For a TS-finding smoke test (any noise level, single method):
```bash
sbatch scripts/run_gad_eckart_fmax.slurm   # full sweep — substitute manually for shorter
```
Or set `--n-samples 5 --n-steps 200` in a custom invocation of `method_single.py`.
