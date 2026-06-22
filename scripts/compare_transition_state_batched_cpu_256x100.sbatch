#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gad_batched_256x100
#SBATCH -p cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus/outslurm/gad-batched-256x100-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/GAD_plus

export GADPLUS_PROJECT_DIR="$PWD"
export GADPLUS_SCRATCH_DIR="$PWD"
export GADPLUS_VENV="$PWD/.venv"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

exec uv run python - <<'PY'
import os
import time

import torch

from gadplus.search.transition_state_optimizer import (
    TransitionStateVectorfield as ScalarVectorfield,
    _diagnostics as scalar_diagnostics,
    cap_displacement as scalar_cap_displacement,
    min_interatomic_distance as scalar_min_interatomic_distance,
)
from gadplus.search.transition_state_optimizer_batched import (
    TransitionStateVectorfieldBatched,
    _diagnostics as batched_diagnostics,
    cap_displacement as batched_cap_displacement,
    min_interatomic_distance as batched_min_interatomic_distance,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
print(
    f"torch={torch.__version__} threads={torch.get_num_threads()} "
    f"cuda_available={torch.cuda.is_available()}",
    flush=True,
)

n_atoms = 7
n_samples = 256
n_steps = 100
seed = 123
sigma0 = 1.0
dt = 0.007
max_atom_disp = 0.05
min_interatomic_dist = 0.75
force_threshold = 0.05
force_criterion = "fmax"

generator = torch.Generator(device="cpu").manual_seed(seed)
initial = sigma0 * torch.randn(
    (n_samples, n_atoms, 3),
    generator=generator,
    dtype=torch.float64,
)
initial = initial - initial.mean(dim=1, keepdim=True)


def run_scalar(initial_coords):
    vectorfield = ScalarVectorfield(n_atoms=n_atoms)
    finals = []
    final_diags = []
    steps_taken = []
    converged = []
    start = time.perf_counter()
    for sample_id in range(initial_coords.shape[0]):
        vectorfield.reset()
        coords = initial_coords[sample_id].clone()
        did_converge = False
        step_count = 0
        for step_idx in range(n_steps):
            final = scalar_diagnostics(vectorfield, coords, force_criterion=force_criterion)
            if int(final["n_neg"]) == 1 and float(final["force_value"]) < force_threshold:
                did_converge = True
                break
            direction = vectorfield(coords)
            step = scalar_cap_displacement(dt * direction, max_atom_disp)
            new_coords = coords + step
            if (
                min_interatomic_dist > 0
                and scalar_min_interatomic_distance(new_coords) < min_interatomic_dist
            ):
                step = 0.5 * step
                new_coords = coords + step
            coords = new_coords.detach()
            step_count = step_idx + 1
        final = scalar_diagnostics(vectorfield, coords, force_criterion=force_criterion)
        did_converge = did_converge or (
            int(final["n_neg"]) == 1 and float(final["force_value"]) < force_threshold
        )
        finals.append(coords)
        final_diags.append(final)
        steps_taken.append(step_count)
        converged.append(did_converge)
    elapsed = time.perf_counter() - start
    return torch.stack(finals), final_diags, torch.tensor(steps_taken), torch.tensor(converged), elapsed


def run_batched(initial_coords):
    vectorfield = TransitionStateVectorfieldBatched(n_atoms=n_atoms)
    vectorfield.reset()
    coords = initial_coords.clone()
    converged = torch.zeros(n_samples, dtype=torch.bool)
    steps_taken = torch.zeros(n_samples, dtype=torch.long)
    start = time.perf_counter()
    for _ in range(n_steps):
        final = batched_diagnostics(vectorfield, coords, force_criterion=force_criterion)
        converged_now = (final["n_neg"] == 1) & (final["force_value"] < force_threshold)
        converged = converged | converged_now
        active = ~converged
        if not bool(active.any().item()):
            break
        active_coords = coords[active]
        direction = vectorfield(active_coords)
        step = batched_cap_displacement(dt * direction, max_atom_disp)
        new_active_coords = active_coords + step
        if min_interatomic_dist > 0:
            too_close = batched_min_interatomic_distance(new_active_coords) < min_interatomic_dist
            if bool(too_close.any().item()):
                backed_off = active_coords + 0.5 * step
                new_active_coords = torch.where(
                    too_close[:, None, None],
                    backed_off,
                    new_active_coords,
                )
        coords = coords.clone()
        coords[active] = new_active_coords.detach()
        steps_taken[active] += 1
    final = batched_diagnostics(vectorfield, coords, force_criterion=force_criterion)
    converged = (final["n_neg"] == 1) & (final["force_value"] < force_threshold)
    elapsed = time.perf_counter() - start
    return coords, final, steps_taken, converged, elapsed


scalar_coords, scalar_final, scalar_steps, scalar_conv, scalar_elapsed = run_scalar(initial)
batched_coords, batched_final, batched_steps, batched_conv, batched_elapsed = run_batched(initial)

scalar_energy = torch.tensor([float(d["energy"]) for d in scalar_final], dtype=torch.float64)
scalar_fmax = torch.tensor([float(d["force_max"]) for d in scalar_final], dtype=torch.float64)
scalar_nneg = torch.tensor([int(d["n_neg"]) for d in scalar_final], dtype=torch.long)

print(
    "comparison_config "
    + f"n_samples={n_samples} n_steps={n_steps} n_atoms={n_atoms} seed={seed}",
    flush=True,
)
print(
    "timing "
    + f"sequential_s={scalar_elapsed:.6f} batched_s={batched_elapsed:.6f} "
    + f"speedup={scalar_elapsed / batched_elapsed:.3f}",
    flush=True,
)
print(
    "max_abs_diff "
    + f"coords={torch.max(torch.abs(scalar_coords - batched_coords)).item():.3e} "
    + f"energy={torch.max(torch.abs(scalar_energy - batched_final['energy'])).item():.3e} "
    + f"fmax={torch.max(torch.abs(scalar_fmax - batched_final['force_max'])).item():.3e}",
    flush=True,
)
print(
    "exact_matches "
    + f"n_neg={bool(torch.equal(scalar_nneg, batched_final['n_neg']))} "
    + f"steps={bool(torch.equal(scalar_steps, batched_steps))} "
    + f"converged={bool(torch.equal(scalar_conv, batched_conv))}",
    flush=True,
)
print(
    "step_summary "
    + f"sequential_min={int(scalar_steps.min().item())} "
    + f"sequential_max={int(scalar_steps.max().item())} "
    + f"batched_min={int(batched_steps.min().item())} "
    + f"batched_max={int(batched_steps.max().item())}",
    flush=True,
)
PY
