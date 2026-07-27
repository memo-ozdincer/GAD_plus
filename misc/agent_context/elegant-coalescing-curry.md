# Plan: Infrastructure Improvements for GAD+ Experiments

## How to orient (for a fresh agent)

**Start here:**
- **`/lustre06/project/6033559/memoozd/GAD_plus/CLAUDE.md`** — Complete project structure, convergence criteria, predict_fn interface, cluster setup, Hydra configs, running experiments, analysis queries. This is the authoritative reference.
- **`/lustre06/project/6033559/memoozd/GAD_plus/EXPERIMENT_LOG.md`** — Full log of all 18 experiments with method descriptions, results tables, SLURM job IDs, data paths. Read the "Consolidated Results" table at the bottom for the current leaderboard.
- **`/lustre06/project/6033559/memoozd/GAD_plus/EXPERIMENTS.tex`** — Publication-ready summary (compile with pdflatex). Has Sella baselines, figures, and the narrative.

**Experiment results live in:**
- `/lustre07/scratch/memoozd/gadplus/runs/` — subdirs per experiment batch (method_cmp_300, precond_gad, round2, round3, round4, sella_baselines, sella_1000, etc.)
- Each subdir has `summary_*.parquet` (per-method aggregate) and `traj_*.parquet` (per-sample per-step)

**Agent memory** (may be stale, verify before using):
- `/home/memoozd/.claude/projects/-lustre06-project-6033559-memoozd-GAD-plus/memory/MEMORY.md`
- Useful entries: Narval SLURM constraints (WANDB_DISABLED, PYTHONUNBUFFERED, no internet on compute), Eckart n_neg fix (always use Eckart projection), inode quota warning on rrg-aspuru

**Other docs:**
- `/lustre06/project/6033559/memoozd/GAD_plus/EXPERIMENT_PLAN_ROUND2.md` — Original round 2 plan (some experiments superseded)
- `/lustre06/project/6033559/memoozd/GAD_plus/PAPER_INSIGHTS.md` — Comparison with noisyTS/LMHE paper
- `/lustre06/project/6033559/memoozd/GAD_plus/DATA_REFERENCE.md` — Parquet schemas, DuckDB queries, plot recipes
- `/lustre06/project/6033559/memoozd/GAD_plus/apr6.md` — Status handoff document between agents

## Context

We're benchmarking GAD (Gentlest Ascent Dynamics) for transition state search on molecular PES using HIP neural network potential on Transition1x (300 organic reactions). We've run ~46,000 optimizations across 18 experiments. Now we need infrastructure improvements for:
1. Fair comparison with Sella (matching convergence criteria)
2. Better IRC validation (bond topology, not just RMSD)
3. Saving converged TS geometries for future diagnostics
4. Tighter convergence for more precise TS locations

---

## Change 1: Switch convergence criterion from force_norm to fmax

### Why
Our GAD convergence uses `force_norm` = mean per-atom force magnitude. Sella uses `fmax` = max absolute force component across all atoms × 3 dimensions. fmax is stricter — a sample can pass force_norm<0.01 but fail fmax<0.01. For fair comparison and downstream use, we should use fmax everywhere.

### Files to modify

**`src/gadplus/core/convergence.py`** (currently 25 lines)
- Current: `force_mean(forces)` returns mean per-atom force norm
- Current: `is_ts_converged(n_neg, force_norm, threshold)` checks force_norm < threshold
- Add: `force_max(forces)` → `float(forces.reshape(-1).abs().max().item())` — max absolute force component (matches Sella's fmax)
- Modify: `is_ts_converged()` to accept a `criterion` parameter ("fmax" or "force_norm") with fmax as default
- Keep `force_mean()` for backward compat / logging

**`src/gadplus/search/gad_search.py`**
- `GADSearchConfig`: add `force_criterion: str = "fmax"` field
- In the convergence check (~line 161), call `force_max(forces)` when criterion is "fmax", `force_mean(forces)` when "force_norm"
- Log both values to trajectory

**`src/gadplus/search/nr_gad_pingpong.py`** and **`src/gadplus/search/blended_gad.py`** and **`src/gadplus/search/rfo_gad.py`**
- Same change: use the configured criterion for convergence check

**`src/gadplus/logging/trajectory.py`** + **`src/gadplus/logging/schema.py`**
- Add `force_max` field to trajectory schema (alongside existing force_norm, force_rms)
- Log fmax at every step

**`scripts/method_single.py`**
- Add `force_max` to summary DataFrame
- Existing: saves force_norm → also save fmax

### Force threshold
Sella's default is fmax<0.01 eV/Å. Our experiments used force_norm<0.01. Since fmax is stricter, keep threshold at 0.01 but the effective bar is higher. This is intentional — matches Sella for fair comparison.

---

## Change 2: IRC validation with bond topology comparison (graph isomorphism)

### Why
Current IRC validation compares endpoints to known reactant/product via RMSD only. Two problems: (a) RMSD is sensitive to atom permutation (same molecule, different atom ordering → large RMSD), (b) RMSD doesn't check if bonds are preserved. Bond topology comparison (graph isomorphism) is permutation-invariant and checks chemical identity.

### Packages available (all already installed, no new deps)
- **ASE 3.28.0** — `ase.neighborlist.neighbor_list('ij', atoms, cutoff)` for bond detection from coordinates
- **NetworkX 3.4.2** — `nx.is_isomorphic(G1, G2, node_match=...)` for graph comparison with element matching
- Both are already in the venv (ASE in pyproject.toml, networkx as transitive dep)

### Files to modify

**`src/gadplus/search/irc_validate.py`** (currently ~130 lines)
- Current: uses `aligned_rmsd()` from `geometry/alignment.py` (Kabsch + Hungarian)
- Add helper: `coords_to_bond_graph(coords, atomic_nums, cutoff=1.85)`:
  1. Create ASE `Atoms` object from coords + atomic_nums
  2. Use `ase.neighborlist.neighbor_list('ij', atoms, cutoff)` to get bonded pairs
  3. Build `nx.Graph` with nodes labeled by atomic number
  4. Return the graph
- Add helper: `bond_graphs_match(graph1, graph2)`:
  1. `nx.is_isomorphic(G1, G2, node_match=lambda a, b: a['Z'] == b['Z'])`
- Modify `validate_irc()` to:
  1. Compute bond graphs for: forward endpoint, reverse endpoint, known reactant, known product
  2. Compare forward↔reactant and reverse↔product via `bond_graphs_match()`
  3. Also compare reverse↔reactant and forward↔product (IRC direction is arbitrary)
  4. Return both RMSD-based and topology-based match results in `IRCResult`
- Add fields to `IRCResult`: `topology_intended: bool`, `topology_half_intended: bool`, `forward_graph_matches_reactant: bool`, etc.

### Cutoff for bond detection
Use covalent radii sum × 1.2 as cutoff (standard practice). ASE has `ase.data.covalent_radii`. For a pair (i, j): bonded if distance < (r_cov_i + r_cov_j) × 1.2.

---

## Change 3: Save converged TS xyz for all experiments

### Why
For future diagnostics: verify HIP quality, compare TS geometries across methods, visualize in 3D. Currently `SearchResult.final_coords` has the data but it's not saved to the summary Parquet (only in per-step trajectory files which are huge).

### Files to modify

**`scripts/method_single.py`**
- After each converged sample, save xyz to a file:
  ```python
  if result.converged:
      xyz_path = os.path.join(output_dir, f"ts_{method}_{noise_pm}pm_{sample_id}.xyz")
      atoms = Atoms(numbers=atomic_nums.cpu().numpy(), positions=result.final_coords.numpy())
      ase.io.write(xyz_path, atoms)
  ```
- Also add `final_coords_flat` to the summary DataFrame (list of floats, same as trajectory format) for convenience

### Alternative: batch xyz
Instead of one file per sample, write one multi-frame xyz per (method, noise):
```python
all_ts_atoms = []
# ... after loop ...
ase.io.write(os.path.join(output_dir, f"ts_all_{method}_{noise_pm}pm.xyz"), all_ts_atoms)
```
This is cleaner — one file with all converged TS geometries, viewable in any molecular viewer.

---

## Change 4: Tighter convergence option (fmax down to 1e-4)

### Why
To bring TS geometries closer together for better comparison. Two TS's might be in the same Morse basin but at different points on the flat ridge. Converging to fmax<1e-4 instead of 1e-2 places them more precisely at the saddle point.

### Implementation
This is just a parameter change — no code changes needed beyond Change 1 (which adds fmax support):
- Add method configs in `method_single.py` with `force_threshold=0.0001`
- These will need more steps (maybe 3000-5000) since convergence tails off slowly
- Run as separate experiments, not replacing the 0.01 threshold runs

### Practical concern
HIP's force accuracy at 1e-4 eV/Å may be the bottleneck. The neural network potential itself has errors at this scale. Worth testing on a few samples first before a full sweep.

---

## Change 5: Descent→GAD one-way switch (already partially implemented)

### Current state
`gad_search.py` already has `descent_until_nneg` config field and one-way switch logic (implemented this session). SLURM job 59362083 is currently running with `descent_then_gad_2` and `descent_then_gad_3` at 50/100/200pm. Results pending.

### No additional code needed — just wait for results and log.

---

## Key files reference (for fresh agent)

| File | What it does | Lines |
|------|-------------|-------|
| `src/gadplus/core/convergence.py` | TS convergence check (n_neg + force) | ~25 |
| `src/gadplus/search/gad_search.py` | Main GAD search loop, GADSearchConfig | ~240 |
| `src/gadplus/search/nr_gad_pingpong.py` | NR-GAD hybrid search | ~280 |
| `src/gadplus/search/blended_gad.py` | λ₂-blended search | ~160 |
| `src/gadplus/search/rfo_gad.py` | RFO-GAD search | ~200 |
| `src/gadplus/search/irc_validate.py` | IRC validation (Sella) | ~130 |
| `src/gadplus/logging/trajectory.py` | Per-step Parquet logging | ~200 |
| `src/gadplus/logging/schema.py` | PyArrow schema definitions | ~50 |
| `src/gadplus/geometry/alignment.py` | Kabsch + Hungarian RMSD | ~150 |
| `src/gadplus/projection/projection.py` | Eckart projection, vib_eig, GAD dynamics | ~310 |
| `src/gadplus/calculator/ase_adapter.py` | HIP→ASE Calculator wrapper | ~60 |
| `scripts/method_single.py` | Main experiment runner | ~280 |
| `scripts/sella_baseline.py` | Sella baseline runner (reference) | ~374 |

### Installed packages (no new installs needed)
- ASE 3.28.0 (bond detection via neighborlist, xyz I/O)
- NetworkX 3.4.2 (graph isomorphism)
- Sella 2.4.2 (IRC, TS optimization baseline)
- PyArrow (Parquet logging)
- OpenBabel 3.1.1.23 (backup for bond detection)
- scipy (Hungarian algorithm via linear_sum_assignment)

---

## Implementation order

1. **Change 1 (fmax criterion)** — foundation, everything else builds on it
2. **Change 3 (save TS xyz)** — quick, no dependencies
3. **Change 2 (IRC bond topology)** — needs ASE + networkx integration
4. **Change 4 (tight convergence)** — parameter-only once Change 1 is in
5. Change 5 is already done, just needs results

## Verification

- Run a quick smoke test: `python -c "from gadplus.core.convergence import force_max; print('OK')"`
- Run 5 samples at 10pm with fmax criterion, verify convergence matches Sella's definition
- Test bond graph generation on a few known molecules, verify isomorphism detects same molecule under permutation
- Verify .xyz files are readable in ASE / protein-viewer extension
