#!/usr/bin/env python
"""Generate the docs+index sidecars for a hybrid HPO grid.

Two grids are currently supported:

* ``hybrid_grid_big``    — 144 cells, fmax<0.05, noise={0, 30} pm.
                           CANCELLED (fmax too easy, response surface flat).
* ``hybrid_grid_fmax01`` — 300 cells, fmax<0.01, noise={10, 30, 100, 150, 200} pm.
                           ACTIVE.

Each grid writes (under its runs dir):

* ``MANIFEST.md``   — human-readable grid description.
* ``grid_index.csv`` — one row per array task: task_id, method,
                      switch_by_eig, gad_dt, trust_radius, noise_pm,
                      output_dir, summary_path (expected), cell_tag.
* ``README.md``     — column dictionary for summary parquets.

Idempotent. Re-run after the grid finishes to refresh.

Usage:
    python scripts/build_hybrid_grid_index.py                # default = fmax01
    python scripts/build_hybrid_grid_index.py --grid big
"""
from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GridSpec:
    name: str
    methods: list[str]
    switches: list[bool]
    dts: list[float]
    trs: list[float]
    noises_pm: list[int]
    force_threshold: float
    # bash index decomposition divisor sequence — must match the slurm script
    # order: (fastest -> slowest changing) = (noise, tr, dt, switch, method)
    moduli: tuple[int, int, int, int, int]
    array_size: int
    slurm_script: str
    slurm_job_id: str

    @property
    def root(self) -> Path:
        return Path("/lustre07/scratch/memoozd/gadplus/runs") / self.name

    def decode(self, task_id: int) -> dict:
        n_mod, tr_mod, dt_mod, sw_mod, m_mod = self.moduli
        noise_i = task_id % n_mod
        tr_i    = (task_id // n_mod) % tr_mod
        dt_i    = (task_id // (n_mod * tr_mod)) % dt_mod
        sw_i    = (task_id // (n_mod * tr_mod * dt_mod)) % sw_mod
        m_i     = (task_id // (n_mod * tr_mod * dt_mod * sw_mod)) % m_mod
        return {
            "method": self.methods[m_i],
            "switch_by_eig": self.switches[sw_i],
            "gad_dt": self.dts[dt_i],
            "trust_radius": self.trs[tr_i],
            "noise_pm": self.noises_pm[noise_i],
        }


GRIDS: dict[str, GridSpec] = {
    "big": GridSpec(
        name="hybrid_grid_big",
        methods=["hybrid_eckart", "hybrid_damped_eckart"],
        switches=[True, False],
        dts=[0.002, 0.005, 0.010],
        trs=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
        noises_pm=[0, 30],
        force_threshold=0.05,
        moduli=(2, 6, 3, 2, 2),
        array_size=144,
        slurm_script="scripts/run_hybrid_grid_big.slurm",
        slurm_job_id="61293538 (CANCELLED)",
    ),
    "fmax01": GridSpec(
        name="hybrid_grid_fmax01",
        methods=["hybrid_eckart", "hybrid_damped_eckart"],
        switches=[True, False],
        dts=[0.003, 0.005, 0.010],
        trs=[0.005, 0.01, 0.02, 0.05, 0.10],
        noises_pm=[10, 30, 100, 150, 200],
        force_threshold=0.01,
        moduli=(5, 5, 3, 2, 2),
        array_size=300,
        slurm_script="scripts/run_hybrid_grid_fmax01.slurm",
        slurm_job_id="see latest sbatch submission",
    ),
}


def cell_tag(spec: GridSpec, p: dict) -> str:
    switch_str = "true" if p["switch_by_eig"] else "false"
    return (f"{p['method']}_sw{switch_str}_dt{p['gad_dt']}_tr{p['trust_radius']}"
            f"_{p['noise_pm']}pm")


def expected_summary_filename(p: dict) -> str:
    """Replicate the naming convention in hybrid_gad_newton_runner.py."""
    switch_tag = "swEIG" if p["switch_by_eig"] else "swFORCE"
    base = f"{p['method']}_{switch_tag}_dt{p['gad_dt']:g}_tr{p['trust_radius']:g}"
    return f"summary_{base}_{p['noise_pm']}pm.parquet"


def write_manifest(spec: GridSpec) -> None:
    axes_table = "\n".join([
        f"| `method` | {', '.join(f'`{m}`' for m in spec.methods)} | {len(spec.methods)} |",
        f"| `switch_by_eig` | true (swEIG), false (swFORCE) | 2 |",
        f"| `gad_dt` | {', '.join(str(d) for d in spec.dts)} | {len(spec.dts)} |",
        f"| `trust_radius` | {', '.join(str(t) for t in spec.trs)} | {len(spec.trs)} |",
        f"| `noise_pm` | {', '.join(str(n) for n in spec.noises_pm)} | {len(spec.noises_pm)} |",
    ])

    n_mod, tr_mod, dt_mod, sw_mod, m_mod = spec.moduli
    idx_block = textwrap.dedent(f"""\
        ```
        noise_i  = T %  {n_mod}            # fastest changing
        tr_i     = (T /  {n_mod}) % {tr_mod}
        dt_i     = (T /  {n_mod*tr_mod}) % {dt_mod}
        switch_i = (T /  {n_mod*tr_mod*dt_mod}) % {sw_mod}
        method_i = (T /  {n_mod*tr_mod*dt_mod*sw_mod}) % {m_mod}    # slowest changing
        ```
        """)

    manifest = textwrap.dedent(f"""\
        # {spec.name} — {spec.array_size} cells

        SLURM job: {spec.slurm_job_id}
        Generator: `{spec.slurm_script}`
        Index decoder: `scripts/build_hybrid_grid_index.py --grid {spec.name.replace("hybrid_grid_", "")}`

        ## Fixed conditions

        | Parameter | Value |
        |---|---|
        | Dataset | Transition1x, split=test, n=287 |
        | Starting geometry | `ts_noised` (noised labelled TS) |
        | Step budget | 2000 outer steps |
        | Convergence criterion | `n_neg == 1 AND fmax < {spec.force_threshold}` |
        | n_neg threshold | eigenvalues `< -1e-4` (Eckart-projected vib Hessian) |
        | Calculator | HIP (`hip_v2.ckpt`) on CUDA |

        ## Axes ({spec.array_size} = {' × '.join(str(m) for m in [m_mod, sw_mod, dt_mod, tr_mod, n_mod])})

        | Axis | Values | Card. |
        |---|---|---|
        {axes_table}

        ## Index encoding (slurm `$SLURM_ARRAY_TASK_ID` → cell)

        {idx_block}
        ## Output layout

        Each cell writes to `{spec.root}/<cell_tag>/`:

        - `summary_<method-tag>_<noise>pm.parquet` — one row per sample.
          Hyperparameters duplicated in (a) directory name, (b) summary
          filename, (c) parquet columns (`trust_radius`, `gad_dt`,
          `switch_by_eig`, `noise_pm`). Analysis should rely on the parquet
          columns; filename parsing is a backstop.
        - `traj_<method-tag>_<noise>pm_<runid>_<sample>.parquet` — per-step log.

        See `README.md` for the column dictionary.

        ## Aggregation

        ```python
        import duckdb
        df = duckdb.sql('''
          SELECT method, switch_by_eig, gad_dt, trust_radius, noise_pm,
                 COUNT(*) AS n,
                 SUM(converged::INT) AS n_conv,
                 ROUND(100.0 * SUM(converged::INT) / COUNT(*), 1) AS pct_conv,
                 ROUND(AVG(CASE WHEN converged THEN converged_step END), 0) AS avg_conv_step
          FROM read_parquet(
            '{spec.root}/**/summary*.parquet',
            union_by_name=true)
          GROUP BY ALL
          ORDER BY noise_pm, pct_conv DESC
        ''').df()
        ```

        Or use `scripts/aggregate_hybrid_grid.py --grid {spec.name.replace("hybrid_grid_", "")}`.

        ## See also

        - `grid_index.csv` — explicit task_id ↔ hyperparameters mapping.
        - `README.md` — schema dictionary for the summary parquets.
        - `{spec.slurm_script}` — the launcher.
        """)

    (spec.root / "MANIFEST.md").write_text(manifest)


# README is the same for every grid — only the criterion differs
README_BODY = """\
# {grid_name} — summary parquet columns

Each `summary_*.parquet` has one row per sample (287 rows per cell).

## Identifiers

| Column | Meaning |
|---|---|
| `sample_id` | T1x test-split sample index (0..286) |
| `formula` | Molecular formula string |
| `method` | Method tag — combines method + switch + dt + tr |
| `noise_pm` | Starting-geometry Gaussian noise std, in pm |
| `n_steps_setting` | Step-budget cap from CLI (2000) |

## Hyperparameters (constant within a cell, varied across cells)

| Column | Meaning |
|---|---|
| `trust_radius` | NR step-size cap (Å) |
| `gad_dt` | GAD outer-loop timestep |
| `switch_by_eig` | Bool: switch phase on eigenvalue (True) or force (False) |

## Outcome columns

| Column | Meaning |
|---|---|
| `converged` | `final_n_neg==1 AND final_force_max < {force_threshold}` |
| `converged_step` | First step where convergence held; None if failed |
| `total_steps` | Steps actually taken (≤ n_steps_setting) |
| `final_n_neg` | Eckart-projected vibrational n_neg at exit (`< -1e-4` threshold) |
| `final_force_max` | max(|F|) at exit, eV/Å (fmax) |
| `final_force_norm` | mean per-atom |F| at exit, eV/Å |
| `final_step_norm_cart` | Last step's Cartesian magnitude |
| `final_force_norm_internal` | Last step's force norm in NR's internal coords |
| `final_target_eigval` | Last step's selected target (TS) eigenvalue |
| `final_eig0`, `final_eig1` | Two smallest vibrational eigenvalues at exit |
| `final_energy` | Energy at exit (eV) |
| `wall_time_s` | Wall time on this sample |
| `last_step_method` | Which kernel produced the final step (nr / gad / etc) |

## Geometry

| Column | Meaning |
|---|---|
| `coords_flat` | Final coords flattened to length 3N, Å |
| `atomic_nums` | Atomic numbers length N |

## Reading the full grid

```python
import duckdb
df = duckdb.sql('''
  SELECT * FROM read_parquet(
    '/lustre07/scratch/memoozd/gadplus/runs/{grid_name}/**/summary*.parquet',
    union_by_name=true)
''').df()
```

Robust to partial completion: `read_parquet(..., union_by_name=true)` silently
skips cells that haven't finished. Partial-grid analysis is always valid.
"""


def write_index(spec: GridSpec) -> Path:
    rows = []
    for t in range(spec.array_size):
        p = spec.decode(t)
        tag = cell_tag(spec, p)
        out_dir = spec.root / tag
        rows.append({
            "task_id": t,
            **p,
            "switch_by_eig": str(p["switch_by_eig"]).lower(),
            "output_dir": str(out_dir),
            "summary_path": str(out_dir / expected_summary_filename(p)),
            "cell_tag": tag,
        })

    idx_path = spec.root / "grid_index.csv"
    with idx_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return idx_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="fmax01", choices=sorted(GRIDS.keys()),
                        help="Which grid to document (default: fmax01)")
    args = parser.parse_args()

    spec = GRIDS[args.grid]
    spec.root.mkdir(parents=True, exist_ok=True)

    idx_path = write_index(spec)
    print(f"wrote {idx_path}")

    write_manifest(spec)
    print(f"wrote {spec.root / 'MANIFEST.md'}")

    readme = README_BODY.format(
        grid_name=spec.name, force_threshold=spec.force_threshold,
    )
    (spec.root / "README.md").write_text(readme)
    print(f"wrote {spec.root / 'README.md'}")

    print(f"\nGrid {spec.name!r}: {spec.array_size} cells. Index covers all of them.")


if __name__ == "__main__":
    main()
