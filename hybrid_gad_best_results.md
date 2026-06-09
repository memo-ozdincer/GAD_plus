# Hybrid GAD-Newton Best Result

Date: 2026-05-28

## Best Setting

Best current reactant-start setting:

| Method | Start | switch force | trust radius | final trust radius | high-index descent | max steps | Converged | Rate | Avg converged step |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `hybrid_damped_eckart` | `reactant` | `10.0` | `0.05` | `0.01` | `newton` | `1000` | `241/287` | `83.97%` | `54.4` |

This tied the best convergence rate observed in the trust/high-index sweep, but had the fastest average convergence among the tied runs.

Summary parquet:

`runs/t1x_geodesic/hybrid_reactant_trust_hiindex_sweep_fmax0p05_n1000/tr0p05_hinewton/summary_hybrid_damped_eckart_swFORCE_dt0.005_tr0.05_hinewton_polishtr0.01_0pm.parquet`

## Exact Run Command

From the repo root:

```bash
sbatch scripts/run_batch_singlenode_uv.sbatch python -u scripts/hybrid_gad_newton_runner.py --method hybrid_damped_eckart --switch-by-eig false --switch-force 10.0 --gad-dt 5e-3 --trust-radius 0.05 --final-trust-radius 0.01 --final-trust-force 0.05 --high-index-descent newton --start-from reactant --force-threshold 0.05 --noise 0 --n-samples 287 --n-steps 1000 --split test --output-dir runs/t1x_geodesic/hybrid_reactant_best_tr0p05_hinewton_polish0p01
```

The SLURM wrapper runs the command through `uv run`.

## Sweep Results

All jobs used `hybrid_damped_eckart`, reactant start, force-based switching, `switch_force=10.0`, `gad_dt=5e-3`, `final_trust_radius=0.01`, `final_trust_force=0.05`, `force_threshold=0.05`, `noise=0`, `n_samples=287`, and `n_steps=1000`.

| trust radius | high-index descent | Converged | Rate | Avg converged step |
|---:|---|---:|---:|---:|
| `0.05` | `newton` | `241/287` | `83.97%` | `54.4` |
| `0.05` | `index_controlled` | `241/287` | `83.97%` | `55.8` |
| `0.02` | `index_controlled` | `241/287` | `83.97%` | `130.9` |
| `0.02` | `newton` | `241/287` | `83.97%` | `131.2` |
| `0.01` | `index_controlled` | `241/287` | `83.97%` | `259.8` |
| `0.01` | `newton` | `233/287` | `81.18%` | `233.7` |

Earlier force-switch baseline at `n_steps=500`, `trust_radius=0.01`, no high-index override:

| switch force | Converged | Rate | Avg converged step |
|---:|---:|---:|---:|
| `10.0` | `212/287` | `73.87%` | `200.8` |
| `1.0` | `207/287` | `72.13%` | `198.5` |
| `0.1` | `201/287` | `70.03%` | `190.8` |

## Best Practices

- Use `hybrid_damped_eckart` for reactant-start searches. Pure reactant-start GAD mostly stays in or returns to `n_neg=0` minimum basins.
- Use force-based switching with `--switch-force 10.0`. In this setup, switching early to damped eigenvector-following Newton was better than spending many steps in GAD.
- Use a larger main trust radius for escape and fast convergence: `--trust-radius 0.05` worked best among `0.01`, `0.02`, and `0.05`.
- Keep a polishing trust radius: `--final-trust-radius 0.01 --final-trust-force 0.05`.
- Prefer `--high-index-descent newton` for the current best run. It tied `index_controlled` in convergence but was slightly faster.
- Keep `--force-threshold 0.05` only as a loose screening criterion. For final reporting, validate candidates with IRC/connectivity or another TS validation step.
- Save trajectories with coordinates for future animation and diagnosis. Hybrid trajectory logging now includes per-step `coords_flat` in the same parquet format as GAD.

## Notes

- Convergence criterion in these runs is `n_neg == 1` and Cartesian `fmax < 0.05`.
- The best result is for finding any transition state from reactant starts, not necessarily the dataset's intended TS.
- For publication-quality success rates, apply IRC/product-reactant connectivity validation after convergence.
