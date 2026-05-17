#!/usr/bin/env python
"""Standalone runner for projected GAD-L-BFGS.

This treats the projected GAD vector field as a root-finding/pseudo-gradient
problem. In the GAD phase, the pseudo-gradient is

    g_gad = -D_gad

where

    D_gad = F_i - 2 <F_i, v> v

in the Eckart-projected, mass-weighted internal subspace. Limited-memory BFGS is
used for every optimization step until the TS convergence criterion is met.

Usage:
  uv run python scripts/hybrid_gad_lbfgs_runner.py \
      --gad-dt 5e-3 --trust-radius 0.01 --lbfgs-history 10 \
      --noise 0.01 --n-samples 287 --n-steps 1000 \
      --output-dir /lustre07/scratch/memoozd/gadplus/runs/hybrid_gad_lbfgs/<cell>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gadplus.calculator.hip import load_hip_calculator, make_hip_predict_fn
from gadplus.data.transition1x import Transition1xDataset, UsePos
from gadplus.paths import hip_checkpoint_path, transition1x_h5_path
from gadplus.projection import atomic_nums_to_symbols, vib_eig
from gadplus.projection.eckart import build_vibrational_basis_torch
from gadplus.projection.masses import get_mass_weights_torch


def fmax(forces: torch.Tensor) -> float:
    f = forces.reshape(-1, 3)
    return float(torch.linalg.vector_norm(f, dim=1).max().item())


def fnorm(forces: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(forces.reshape(-1)).item())


def masses_from_z(z: torch.Tensor, device=None, dtype=torch.float64) -> torch.Tensor:
    symbols = atomic_nums_to_symbols(z)
    masses, _, _, _ = get_mass_weights_torch(symbols, device=device, dtype=dtype)
    return masses


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _internal_mass_weighted_state(
    force_cart: torch.Tensor,
    hessian_cart: torch.Tensor,
    coords: torch.Tensor,
    masses: torch.Tensor,
) -> dict:
    """Convert Cartesian force/Hessian to Eckart-projected mass-weighted space."""
    force_shape = force_cart.shape
    dtype = force_cart.dtype
    device = force_cart.device

    F_x = force_cart.reshape(-1)
    H_x = _symmetrize(hessian_cart.to(dtype=dtype, device=device))
    coords = coords.to(dtype=dtype, device=device)
    masses = masses.to(dtype=dtype, device=device)

    n = F_x.numel()
    if H_x.shape != (n, n):
        raise ValueError("hessian_cart must have shape (force_cart.numel(), force_cart.numel()).")
    if coords.shape[-1] != 3 or coords.numel() != n:
        raise ValueError("coords must have shape (N, 3), with 3*N == force_cart.numel().")
    if masses.shape != (coords.shape[0],):
        raise ValueError("masses must have shape (N,).")

    U_int, U_ext, _ = build_vibrational_basis_torch(coords, masses)
    sqrt_m3 = torch.sqrt(masses).repeat_interleave(3)
    inv_sqrt_m3 = 1.0 / sqrt_m3

    F_q = inv_sqrt_m3 * F_x
    H_q = inv_sqrt_m3[:, None] * H_x * inv_sqrt_m3[None, :]
    H_q = _symmetrize(H_q)
    F_i = U_int.T @ F_q
    H_i = U_int.T @ H_q @ U_int
    H_i = _symmetrize(H_i)

    return {
        "force_shape": force_shape,
        "F_i": F_i,
        "H_i": H_i,
        "U_int": U_int,
        "U_ext": U_ext,
        "inv_sqrt_m3": inv_sqrt_m3,
    }


def _internal_vector_to_cartesian(vec_i: torch.Tensor, state: dict) -> torch.Tensor:
    vec_q = state["U_int"] @ vec_i
    vec_x = state["inv_sqrt_m3"] * vec_q
    return vec_x.reshape(state["force_shape"])


def _internal_step_to_cartesian(
    step_i: torch.Tensor,
    state: dict,
    trust_radius: float | None = None,
) -> torch.Tensor:
    step_x = _internal_vector_to_cartesian(step_i, state)
    if trust_radius is None:
        return step_x

    radius = torch.as_tensor(trust_radius, dtype=step_x.dtype, device=step_x.device)
    norm = torch.linalg.vector_norm(step_x)
    scale = torch.minimum(
        torch.ones((), dtype=step_x.dtype, device=step_x.device),
        radius / (norm + torch.finfo(step_x.dtype).eps),
    )
    return scale * step_x


def info_scalar(info: dict, key: str, default=None) -> float | None:
    value = info.get(key, default)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def info_int(info: dict, key: str, default=0) -> int:
    value = info.get(key, default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return int(default)
        return int(value.detach().reshape(-1)[0].cpu().item())
    return int(value)


def n_neg_eckart(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
) -> tuple[int, float, float]:
    """Eckart-projected n_neg + eig0 + eig1 (vibrational only)."""
    syms = atomic_nums_to_symbols(atomic_nums)
    evals, _, _ = vib_eig(hessian, coords, syms, purify=False)
    evals_sorted = torch.sort(evals).values
    n_neg = int((evals_sorted < 0).sum().item())
    eig0 = float(evals_sorted[0].item()) if evals_sorted.numel() > 0 else 0.0
    eig1 = float(evals_sorted[1].item()) if evals_sorted.numel() > 1 else 0.0
    return n_neg, eig0, eig1


def _bool_arg(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def _mass_weighted_coords(coords: torch.Tensor, state: dict) -> torch.Tensor:
    sqrt_m3 = 1.0 / state["inv_sqrt_m3"]
    return sqrt_m3 * coords.reshape(-1)


def _clamp_optional(
    value: torch.Tensor,
    min_value: float | None,
    max_value: float | None,
) -> torch.Tensor:
    if min_value is not None:
        value = torch.maximum(
            value,
            torch.as_tensor(min_value, dtype=value.dtype, device=value.device),
        )
    if max_value is not None:
        value = torch.minimum(
            value,
            torch.as_tensor(max_value, dtype=value.dtype, device=value.device),
        )
    return value


class LBFGSMemory:
    """Limited-memory inverse-Hessian state for the GAD pseudo-gradient."""

    def __init__(
        self,
        history_size: int,
        initial_scale: float,
        curvature_tol: float,
        reset_overlap: float,
        reset_on_index_change: bool,
        h0_min: float | None,
        h0_max: float | None,
    ):
        self.history_size = max(0, int(history_size))
        self.initial_scale = float(initial_scale)
        self.curvature_tol = float(curvature_tol)
        self.reset_overlap = float(reset_overlap)
        self.reset_on_index_change = bool(reset_on_index_change)
        self.h0_min = h0_min
        self.h0_max = h0_max
        self.s_list: list[torch.Tensor] = []
        self.y_list: list[torch.Tensor] = []
        self.rho_list: list[torch.Tensor] = []
        self.prev_q: torch.Tensor | None = None
        self.prev_g: torch.Tensor | None = None
        self.prev_mode: torch.Tensor | None = None
        self.prev_num_negative_modes: int | None = None

    def reset(self, clear_previous: bool = True):
        self.s_list.clear()
        self.y_list.clear()
        self.rho_list.clear()
        if clear_previous:
            self.prev_q = None
            self.prev_g = None
            self.prev_mode = None
            self.prev_num_negative_modes = None

    @property
    def n_pairs(self) -> int:
        return len(self.s_list)

    def update_with_current(
        self,
        q: torch.Tensor,
        g: torch.Tensor,
        mode: torch.Tensor,
        num_negative_modes: int,
    ) -> dict:
        """Update memory using the previous and current pseudo-gradient states."""
        q = q.detach().clone()
        g = g.detach().clone()
        mode_norm = torch.linalg.vector_norm(mode).clamp_min(
            torch.finfo(mode.dtype).eps
        )
        mode = (mode / mode_norm).detach().clone()

        stats = {
            "lbfgs_pair_status": "init",
            "lbfgs_reset_reason": "",
            "lbfgs_mode_overlap": torch.nan,
            "lbfgs_sTy": torch.nan,
            "lbfgs_curvature_threshold": torch.nan,
        }

        if self.prev_q is None:
            self.prev_q = q
            self.prev_g = g
            self.prev_mode = mode
            self.prev_num_negative_modes = int(num_negative_modes)
            return stats

        reset_reason = ""
        if self.prev_q.shape != q.shape or self.prev_g.shape != g.shape:
            reset_reason = "dimension_change"
        else:
            mode_overlap = torch.abs(torch.dot(self.prev_mode, mode))
            stats["lbfgs_mode_overlap"] = mode_overlap
            if mode_overlap < self.reset_overlap:
                reset_reason = "mode_overlap"
            elif (
                self.reset_on_index_change
                and self.prev_num_negative_modes is not None
                and int(num_negative_modes) != self.prev_num_negative_modes
            ):
                reset_reason = "index_change"

        if reset_reason:
            self.reset(clear_previous=False)
            self.prev_q = q
            self.prev_g = g
            self.prev_mode = mode
            self.prev_num_negative_modes = int(num_negative_modes)
            stats["lbfgs_pair_status"] = "reset"
            stats["lbfgs_reset_reason"] = reset_reason
            return stats

        s = q - self.prev_q
        y = g - self.prev_g
        sTy = torch.dot(s, y)
        s_norm = torch.linalg.vector_norm(s)
        y_norm = torch.linalg.vector_norm(y)
        threshold = self.curvature_tol * s_norm * y_norm
        stats["lbfgs_sTy"] = sTy
        stats["lbfgs_curvature_threshold"] = threshold

        if self.history_size > 0 and sTy > threshold:
            if len(self.s_list) == self.history_size:
                self.s_list.pop(0)
                self.y_list.pop(0)
                self.rho_list.pop(0)
            self.s_list.append(s.detach().clone())
            self.y_list.append(y.detach().clone())
            self.rho_list.append((1.0 / sTy).detach().clone())
            stats["lbfgs_pair_status"] = "accepted"
        else:
            stats["lbfgs_pair_status"] = "skipped_curvature"

        self.prev_q = q
        self.prev_g = g
        self.prev_mode = mode
        self.prev_num_negative_modes = int(num_negative_modes)
        return stats

    def direction(self, g: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Return an L-BFGS search direction for minimizing the pseudo-gradient."""
        g = g.detach()
        stats = {
            "lbfgs_h0_scale": torch.as_tensor(
                self.initial_scale, dtype=g.dtype, device=g.device
            ),
            "lbfgs_descent_dot": torch.nan,
            "lbfgs_fallback": False,
        }

        if not self.s_list:
            p = -self.initial_scale * g
            stats["lbfgs_descent_dot"] = torch.dot(p, g)
            return p, stats

        q = g.clone()
        alphas: list[torch.Tensor] = []
        for s, y, rho in zip(
            reversed(self.s_list),
            reversed(self.y_list),
            reversed(self.rho_list),
        ):
            alpha = rho * torch.dot(s, q)
            q = q - alpha * y
            alphas.append(alpha)

        s_last = self.s_list[-1]
        y_last = self.y_list[-1]
        yTy = torch.dot(y_last, y_last)
        eps = torch.finfo(g.dtype).eps
        if yTy > eps:
            h0 = torch.dot(s_last, y_last) / yTy
            h0 = _clamp_optional(h0, self.h0_min, self.h0_max)
        else:
            h0 = torch.as_tensor(self.initial_scale, dtype=g.dtype, device=g.device)
        stats["lbfgs_h0_scale"] = h0

        r = h0 * q
        for s, y, rho, alpha in zip(
            self.s_list,
            self.y_list,
            self.rho_list,
            reversed(alphas),
        ):
            beta = rho * torch.dot(y, r)
            r = r + s * (alpha - beta)

        p = -r
        descent_dot = torch.dot(p, g)
        stats["lbfgs_descent_dot"] = descent_dot
        if descent_dot >= 0.0:
            self.reset(clear_previous=False)
            p = -self.initial_scale * g
            stats["lbfgs_descent_dot"] = torch.dot(p, g)
            stats["lbfgs_fallback"] = True

        return p, stats


def projected_gad_lbfgs_step(
    force_cart: torch.Tensor,
    hessian_cart: torch.Tensor,
    coords: torch.Tensor,
    masses: torch.Tensor,
    lbfgs_memory: LBFGSMemory,
    target_mode: int = 0,
    min_curvature: float = 1.0e-8,
    trust_radius: float | None = None,
):
    """Projected GAD pseudo-gradient L-BFGS step."""
    state = _internal_mass_weighted_state(
        force_cart=force_cart,
        hessian_cart=hessian_cart,
        coords=coords,
        masses=masses,
    )

    F_i = state["F_i"]
    H_i = state["H_i"]
    eigvals, eigvecs = torch.linalg.eigh(H_i)

    if not (0 <= target_mode < eigvals.numel()):
        raise ValueError("target_mode is outside the internal-mode spectrum.")

    inertia_tol = min_curvature
    negative_modes = eigvals < -inertia_tol
    zero_modes = eigvals.abs() <= inertia_tol
    num_negative_modes = torch.sum(negative_modes)
    num_zero_modes = torch.sum(zero_modes)
    num_positive_modes = torch.sum(eigvals > inertia_tol)
    hessian_has_clear_index1 = (
        (num_negative_modes == 1)
        & (num_zero_modes == 0)
        & negative_modes[target_mode]
    )
    force_norm_internal = torch.linalg.vector_norm(F_i)

    v = eigvecs[:, target_mode]
    gad_dir_i = F_i - 2.0 * torch.dot(F_i, v) * v

    # Store memory in the fixed full mass-weighted Cartesian frame rather than
    # in the moving Eckart basis coordinates.
    gad_dir_q = state["U_int"] @ gad_dir_i
    gad_grad_q = -gad_dir_q
    coords_q = _mass_weighted_coords(coords, state)
    mode_q = state["U_int"] @ v

    update_stats = lbfgs_memory.update_with_current(
        q=coords_q,
        g=gad_grad_q,
        mode=mode_q,
        num_negative_modes=int(num_negative_modes.detach().cpu().item()),
    )
    step_q, direction_stats = lbfgs_memory.direction(gad_grad_q)

    # Project the quasi-Newton proposal back into the current internal space
    # before returning to Cartesian coordinates.
    step_i = state["U_int"].T @ step_q
    step_cart = _internal_step_to_cartesian(
        step_i=step_i,
        state=state,
        trust_radius=trust_radius,
    )
    direction_cart = _internal_vector_to_cartesian(
        vec_i=step_i,
        state=state,
    )

    info = {
        "method": "projected_gad_lbfgs",
        "internal_eigvals": eigvals,
        "target_eigval": eigvals[target_mode],
        "num_external_modes": state["U_ext"].shape[1],
        "num_internal_modes": state["U_int"].shape[1],
        "num_negative_modes": num_negative_modes,
        "num_zero_modes": num_zero_modes,
        "num_positive_modes": num_positive_modes,
        "hessian_has_clear_index1": hessian_has_clear_index1,
        "direction_cart": direction_cart,
        "direction_norm_cart": torch.linalg.vector_norm(direction_cart),
        "force_norm_internal": force_norm_internal,
        "step_norm_internal": torch.linalg.vector_norm(step_i),
        "step_norm_cart": torch.linalg.vector_norm(step_cart),
        "lbfgs_pairs": lbfgs_memory.n_pairs,
        **update_stats,
        **direction_stats,
    }
    return step_cart, info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gad-dt",
        type=float,
        default=5.0e-3,
        help="Initial inverse scale for L-BFGS; pure-GAD fallback step is dt * D_gad.",
    )
    p.add_argument("--trust-radius", type=float, default=0.01)
    p.add_argument("--min-curvature", type=float, default=1.0e-8)
    p.add_argument("--lbfgs-history", type=int, default=10)
    p.add_argument("--lbfgs-curvature-tol", type=float, default=1.0e-10)
    p.add_argument("--lbfgs-reset-overlap", type=float, default=0.5)
    p.add_argument("--lbfgs-reset-on-index-change", type=_bool_arg, default=True)
    p.add_argument("--lbfgs-h0-min", type=float, default=None)
    p.add_argument("--lbfgs-h0-max", type=float, default=None)
    p.add_argument(
        "--noise",
        type=float,
        required=True,
        help="Gaussian noise stddev in Angstrom (e.g. 0.01 = 10pm)",
    )
    p.add_argument("--n-samples", type=int, default=287)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--force-threshold",
        type=float,
        default=0.01,
        help="fmax convergence criterion (with n_neg=1)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_pm = int(round(args.noise * 1000))

    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = str(hip_checkpoint_path())
    h5 = str(transition1x_h5_path())

    calculator = load_hip_calculator(ckpt, device=device)
    predict_fn = make_hip_predict_fn(calculator)
    print(f"HIP loaded on {device}")

    dataset = Transition1xDataset(
        h5,
        split=args.split,
        max_samples=args.n_samples,
        transform=UsePos("pos_transition"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    method_tag = "gad_lbfgs_eckart"
    method_tag += (
        f"_dt{args.gad_dt:g}_tr{args.trust_radius:g}"
        f"_m{args.lbfgs_history}"
    )
    run_id = f"{method_tag}_{noise_pm}pm_{uuid.uuid4().hex[:8]}"
    summary_path = out_dir / f"summary_{method_tag}_{noise_pm}pm.parquet"

    torch.manual_seed(args.seed)
    noise_vecs = {}
    for i in range(len(dataset)):
        s = dataset[i]
        noise_vecs[i] = torch.randn_like(s.pos) * args.noise

    rows = []
    t_total = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]
        coords_ts = sample.pos.to(device)
        z = sample.z.to(device)
        formula = getattr(sample, "formula", f"sample_{i}")
        coords = (coords_ts + noise_vecs[i].to(device)).double()
        atomic_nums = z
        masses = masses_from_z(atomic_nums, device=coords.device, dtype=coords.dtype)
        lbfgs_memory = LBFGSMemory(
            history_size=args.lbfgs_history,
            initial_scale=args.gad_dt,
            curvature_tol=args.lbfgs_curvature_tol,
            reset_overlap=args.lbfgs_reset_overlap,
            reset_on_index_change=args.lbfgs_reset_on_index_change,
            h0_min=args.lbfgs_h0_min,
            h0_max=args.lbfgs_h0_max,
        )

        traj_rows = []
        t0 = time.time()
        converged = False
        converged_step = None
        final_force_max = float("nan")
        final_force_norm = float("nan")
        final_n_neg = -1
        final_eig0 = 0.0
        final_eig1 = 0.0
        final_energy = float("nan")
        final_method_used = ""
        final_step_norm_cart = float("nan")
        final_force_norm_internal = float("nan")
        final_target_eigval = float("nan")
        final_lbfgs_pairs = 0
        final_lbfgs_pair_status = ""
        final_lbfgs_reset_reason = ""
        final_lbfgs_mode_overlap = float("nan")
        final_lbfgs_sTy = float("nan")
        final_lbfgs_descent_dot = float("nan")
        final_lbfgs_fallback = False
        n_steps_actual = 0

        for step_idx in range(args.n_steps):
            out = predict_fn(
                coords,
                atomic_nums,
                do_hessian=True,
                require_grad=False,
            )
            E = out["energy"]
            F = out["forces"]
            H = out["hessian"]
            F = F.reshape(-1, 3).double()
            H = H.reshape(F.numel(), F.numel()).double()

            fmax_v = fmax(F)
            fnorm_v = fnorm(F)
            E_v = float(E.item()) if hasattr(E, "item") else float(E)
            n_neg, eig0, eig1 = n_neg_eckart(H, coords, atomic_nums)

            traj_rows.append(
                {
                    "sample_id": i,
                    "step": step_idx,
                    "energy": E_v,
                    "force_max": fmax_v,
                    "force_norm": fnorm_v,
                    "n_neg": n_neg,
                    "eig0": eig0,
                    "eig1": eig1,
                    "step_method": None,
                    "step_norm_cart": None,
                    "force_norm_internal": None,
                    "target_eigval": None,
                    "lbfgs_pairs": None,
                    "lbfgs_pair_status": None,
                    "lbfgs_reset_reason": None,
                    "lbfgs_mode_overlap": None,
                    "lbfgs_sTy": None,
                    "lbfgs_descent_dot": None,
                    "lbfgs_fallback": None,
                }
            )

            if n_neg == 1 and fmax_v < args.force_threshold:
                converged = True
                converged_step = step_idx
                final_force_max = fmax_v
                final_force_norm = fnorm_v
                final_n_neg = n_neg
                final_eig0 = eig0
                final_eig1 = eig1
                final_energy = E_v
                n_steps_actual = step_idx + 1
                break

            step, info = projected_gad_lbfgs_step(
                force_cart=F,
                hessian_cart=H,
                coords=coords.double(),
                masses=masses,
                lbfgs_memory=lbfgs_memory,
                target_mode=0,
                min_curvature=args.min_curvature,
                trust_radius=args.trust_radius,
            )
            used = info.get("method", "?")
            step_norm_cart = info_scalar(
                info,
                "step_norm_cart",
                default=torch.linalg.vector_norm(step),
            )
            force_norm_internal = info_scalar(info, "force_norm_internal")
            target_eigval = info_scalar(info, "target_eigval")
            lbfgs_pairs = info_int(info, "lbfgs_pairs")
            lbfgs_pair_status = str(info.get("lbfgs_pair_status", ""))
            lbfgs_reset_reason = str(info.get("lbfgs_reset_reason", ""))
            lbfgs_mode_overlap = info_scalar(info, "lbfgs_mode_overlap")
            lbfgs_sTy = info_scalar(info, "lbfgs_sTy")
            lbfgs_descent_dot = info_scalar(info, "lbfgs_descent_dot")
            lbfgs_fallback = bool(info.get("lbfgs_fallback", False))

            traj_rows[-1].update(
                {
                    "step_method": used,
                    "step_norm_cart": step_norm_cart,
                    "force_norm_internal": force_norm_internal,
                    "target_eigval": target_eigval,
                    "lbfgs_pairs": lbfgs_pairs,
                    "lbfgs_pair_status": lbfgs_pair_status,
                    "lbfgs_reset_reason": lbfgs_reset_reason,
                    "lbfgs_mode_overlap": lbfgs_mode_overlap,
                    "lbfgs_sTy": lbfgs_sTy,
                    "lbfgs_descent_dot": lbfgs_descent_dot,
                    "lbfgs_fallback": lbfgs_fallback,
                }
            )

            step = step.reshape_as(coords)
            coords = (coords + step).detach()

            final_force_max = fmax_v
            final_force_norm = fnorm_v
            final_n_neg = n_neg
            final_eig0 = eig0
            final_eig1 = eig1
            final_energy = E_v
            final_method_used = used
            final_step_norm_cart = step_norm_cart
            final_force_norm_internal = force_norm_internal
            final_target_eigval = target_eigval
            final_lbfgs_pairs = lbfgs_pairs
            final_lbfgs_pair_status = lbfgs_pair_status
            final_lbfgs_reset_reason = lbfgs_reset_reason
            final_lbfgs_mode_overlap = (
                float("nan") if lbfgs_mode_overlap is None else lbfgs_mode_overlap
            )
            final_lbfgs_sTy = float("nan") if lbfgs_sTy is None else lbfgs_sTy
            final_lbfgs_descent_dot = (
                float("nan") if lbfgs_descent_dot is None else lbfgs_descent_dot
            )
            final_lbfgs_fallback = lbfgs_fallback
            n_steps_actual = step_idx + 1

        wall = time.time() - t0
        traj_path = out_dir / f"traj_{method_tag}_{noise_pm}pm_{run_id[-8:]}_{i}.parquet"
        if traj_rows:
            pd.DataFrame(traj_rows).to_parquet(traj_path)

        rows.append(
            {
                "sample_id": i,
                "formula": formula,
                "method": method_tag,
                "noise_pm": noise_pm,
                "n_steps_setting": args.n_steps,
                "converged": converged,
                "converged_step": converged_step,
                "total_steps": n_steps_actual,
                "final_force_max": final_force_max,
                "final_force_norm": final_force_norm,
                "final_step_norm_cart": final_step_norm_cart,
                "final_force_norm_internal": final_force_norm_internal,
                "final_target_eigval": final_target_eigval,
                "final_n_neg": final_n_neg,
                "final_eig0": final_eig0,
                "final_eig1": final_eig1,
                "final_energy": final_energy,
                "wall_time_s": wall,
                "last_step_method": final_method_used,
                "trust_radius": args.trust_radius,
                "gad_dt": args.gad_dt,
                "lbfgs_history": args.lbfgs_history,
                "lbfgs_curvature_tol": args.lbfgs_curvature_tol,
                "lbfgs_reset_overlap": args.lbfgs_reset_overlap,
                "lbfgs_reset_on_index_change": args.lbfgs_reset_on_index_change,
                "final_lbfgs_pairs": final_lbfgs_pairs,
                "final_lbfgs_pair_status": final_lbfgs_pair_status,
                "final_lbfgs_reset_reason": final_lbfgs_reset_reason,
                "final_lbfgs_mode_overlap": final_lbfgs_mode_overlap,
                "final_lbfgs_sTy": final_lbfgs_sTy,
                "final_lbfgs_descent_dot": final_lbfgs_descent_dot,
                "final_lbfgs_fallback": final_lbfgs_fallback,
            }
        )

        status = "CONV" if converged else "FAIL"
        print(
            f"  [{i:3d}] {formula:>12s} | {status} | n_neg={final_n_neg} "
            f"fmax={final_force_max:.4f} steps={n_steps_actual} wall={wall:.1f}s "
            f"last_method={final_method_used} lbfgs_pairs={final_lbfgs_pairs}",
            flush=True,
        )

    pd.DataFrame(rows).to_parquet(summary_path)
    print(f"\nWrote {summary_path} ({len(rows)} rows)")
    print(f"Total wall: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
