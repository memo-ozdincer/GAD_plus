"""Strictly pointwise, scale-covariant smooth ``lambda_2`` GAD.

This module implements a closed-form map that uses only the coordinates,
gradient, and Hessian at the current point.  It has no line search, adaptive
radius, rejected trial, mode-tracking state, accumulated quasi-Newton model,
or global pseudo-potential.

Two spectral objects define the map:

``rho_beta(H)``
    A normalized matrix soft-min.  It approaches the projector onto the
    lowest eigendirection but is basis-invariant at a degeneracy.

``w(H)``
    A dimensionless sigmoid gate on the exact second ordered vibrational
    eigenvalue.  The ordered eigenvalue is continuous at crossings, although
    generally not differentiable there.  Thus the complete field is C0 at
    the remaining spectral boundaries; it never needs an arbitrary choice of
    eigenvector inside a degenerate subspace.

A pointwise Levenberg regularizer supplies an intrinsic step bound in closed
form.  It is not an adaptive trust-region algorithm: its radius is recomputed
from the current geometry and never carried between iterations.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from gadplus.core.convergence import (
    force_max,
    force_mean,
    force_value_from_criterion,
    is_ts_converged,
)
from gadplus.core.types import PredictFn
from gadplus.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig


@dataclass(frozen=True)
class IntrinsicGADConfig:
    """Configuration for the pointwise smooth-index GAD map.

    Only two dimensionless algorithmic scales remain:

    ``spectral_temperature``
        Resolution of the soft lowest-mode density and ``lambda_2`` gate.
        The Hessian is normalized by its RMS vibrational eigenvalue, so this
        parameter is invariant to energy-unit or energy-scale changes.

    ``step_fraction``
        Maximum mass-weighted RMS step as a fraction of the current inverse-
        RMS pair distance.  This geometric length contracts smoothly near a
        pair collision and scales with a uniform rescaling of the geometry.
    """

    max_steps: int = 1000
    force_threshold: float = 0.01
    force_criterion: str = "fmax"
    index_threshold: float = 1.0e-4
    spectral_temperature: float = 0.01
    step_fraction: float = 0.05
    gate_variant: str = "lambda2"
    purify_hessian: bool = False
    record_history: bool = True

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.force_threshold <= 0:
            raise ValueError("force_threshold must be positive")
        if self.index_threshold < 0:
            raise ValueError("index_threshold must be nonnegative")
        if self.spectral_temperature <= 0:
            raise ValueError("spectral_temperature must be positive")
        if self.step_fraction <= 0:
            raise ValueError("step_fraction must be positive")
        if self.gate_variant not in {
            "lambda2",
            "alignment",
            "competitive",
            "competitive_subspace",
            "guard",
            "gad",
        }:
            raise ValueError(
                "gate_variant must be 'lambda2', 'alignment', 'competitive', "
                "'competitive_subspace', 'guard', or 'gad'"
            )


@dataclass(frozen=True)
class IntrinsicGADStep:
    """Diagnostics for one pointwise evaluation and its closed-form step."""

    iteration: int
    energy: float
    force_max: float
    n_neg: int
    eig0: float
    eig1: float
    gate_weight: float
    spectral_scale: float
    geometric_length: float
    local_radius: float
    regularizer: float
    step_rms: float


@dataclass(frozen=True)
class IntrinsicGADObservation:
    """Read-only snapshot emitted after each coherent local evaluation.

    The callback receiving this object is observational: all tensors are
    detached CPU clones and no value returned by the callback enters the
    optimizer map. ``step_cart`` is ``None`` for a terminal evaluation.
    """

    evaluation: int
    iteration: int
    wall_time_s: float
    energy: float
    forces: torch.Tensor
    coords: torch.Tensor
    eigenvalues: torch.Tensor
    n_neg: int
    lambda2_gate: float
    effective_gate: float
    spectral_scale: float
    low_mode_weights: torch.Tensor
    negative_mode_weights: torch.Tensor
    gradient_coefficients: torch.Tensor
    soft_activity: float
    extra_negative_activity: float
    activity_fraction: float
    spectral_entropy: float
    lowest_reflection: float
    geometric_length: float | None
    local_radius: float | None
    regularizer: float | None
    step_mw_rms: float | None
    step_cart: torch.Tensor | None
    terminal: bool


@dataclass(frozen=True)
class IntrinsicGADResult:
    """Result and pointwise diagnostics from :func:`run_intrinsic_gad`."""

    converged: bool
    converged_step: int | None
    total_steps: int
    n_evaluations: int
    final_coords: torch.Tensor
    final_energy: float
    final_n_neg: int
    final_force_norm: float
    final_force_max: float
    final_eig0: float
    final_eig1: float
    final_gate_weight: float
    wall_time_s: float
    failure_type: str | None = None
    history: tuple[IntrinsicGADStep, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SpectralPolicy:
    """Dimensionless spectral quantities defining the gated reflection."""

    scale: torch.Tensor
    gate: torch.Tensor
    low_mode_weights: torch.Tensor
    negative_mode_weights: torch.Tensor
    lowest_normalized_eigenvalue: torch.Tensor
    temperature: float


@dataclass(frozen=True)
class GateDiagnostics:
    """Pointwise scalar diagnostics used to interpret a spectral gate."""

    lambda2_gate: torch.Tensor
    effective_gate: torch.Tensor
    soft_activity: torch.Tensor
    extra_negative_activity: torch.Tensor
    activity_fraction: torch.Tensor
    spectral_entropy: torch.Tensor
    reflection_weights: torch.Tensor
    lowest_reflection: torch.Tensor


@dataclass(frozen=True)
class _State:
    coords: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    evals: torch.Tensor
    modes_mw: torch.Tensor
    gradient_mw: torch.Tensor
    policy: SpectralPolicy
    gradient_coefficients: torch.Tensor
    gate_diagnostics: GateDiagnostics
    effective_gate: torch.Tensor
    n_neg: int
    force_norm: float
    force_max: float


def smooth_spectral_policy(
    eigenvalues: torch.Tensor,
    temperature: float,
) -> SpectralPolicy:
    r"""Return the basis-invariant low-mode density and ``lambda_2`` gate.

    For RMS spectral scale ``s`` and dimensionless temperature ``tau``,

    .. math::

        p_i = \frac{\exp[-\lambda_i/(\tau s)]}
                    {\sum_j\exp[-\lambda_j/(\tau s)]},
        \qquad
        w = \sigma\!\left(\frac{\lambda_2}{\tau s}\right).

    In matrix form the soft projector is
    ``rho = V diag(p) V.T = exp(-H/(tau*s)) / tr(exp(...))``.  It is analytic
    in ``H`` for finite ``tau`` and invariant under rotations within a
    degenerate eigenspace.  ``lambda_2`` itself is continuous but only
    piecewise differentiable when ordered eigenvalues meet.
    """

    evals = eigenvalues.to(torch.float64).reshape(-1)
    if evals.numel() == 0:
        raise ValueError("at least one vibrational eigenvalue is required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scale = torch.sqrt(torch.mean(evals.square()))
    numerical_floor = torch.finfo(evals.dtype).eps * torch.maximum(
        evals.abs().amax(),
        torch.ones((), dtype=evals.dtype, device=evals.device),
    )
    scale = torch.maximum(scale, numerical_floor)
    normalized = evals / scale
    low_weights = torch.softmax(-normalized / temperature, dim=0)
    negative_weights = torch.sigmoid(-normalized / temperature)
    lambda2 = normalized[1] if normalized.numel() > 1 else normalized[0]
    gate = torch.sigmoid(lambda2 / temperature)
    return SpectralPolicy(
        scale=scale,
        gate=gate,
        low_mode_weights=low_weights,
        negative_mode_weights=negative_weights,
        lowest_normalized_eigenvalue=normalized[0],
        temperature=float(temperature),
    )


def effective_gate_weight(
    gradient_coefficients: torch.Tensor,
    policy: SpectralPolicy,
    variant: str = "lambda2",
) -> torch.Tensor:
    r"""Return the local gate for one of the experimental policies.

    ``alignment`` rescues ascent in a high-index region in proportion to the
    fraction of gradient activity in the soft-low-mode density.  The more
    selective ``competitive`` policy compares that activity only with
    activity in additional negative modes; stable-mode gradient components
    therefore do not suppress a targeted escape.

    ``guard`` applies the competitive gate together with a parameter-free
    (beyond the existing spectral temperature) bell-shaped activation at
    ``lambda_1=0``.  It is an experimental index-boundary safeguard, not a
    guarantee that a finite step cannot enter a convex basin.

    Both extensions reduce exactly to the maintained ``lambda2`` gate when
    it is fully active, are energy-scale invariant, and use no history.  The
    quotient is defined as zero at an exactly stationary point, where the
    step itself is zero and the gate value cannot affect the map.
    """

    return gate_diagnostics(gradient_coefficients, policy, variant).effective_gate


def _base_gate_variant(variant: str) -> str:
    """Return the scalar-gate policy underlying a reflection variant."""

    return "competitive" if variant == "competitive_subspace" else variant


def relative_soft_subspace_weights(policy: SpectralPolicy) -> torch.Tensor:
    r"""Return the parameter-free relative soft-subspace filter.

    With the trace-normalized soft density ``p``, define

    .. math::

        \tilde p_i = p_i / \max_j p_j
        = \exp[-(\lambda_i-\lambda_1)/(\tau s_H)].

    It gives weight one to every exactly degenerate lowest mode, while an
    isolated lowest mode is exactly ordinary one-mode GAD.  The use of the
    ordered minimum is continuous but piecewise smooth at eigenvalue
    crossings, the same regularity already present in the ``lambda_2`` gate.
    """

    weights = policy.low_mode_weights
    return weights / weights.amax().clamp_min(torch.finfo(weights.dtype).tiny)


def reflection_weights(policy: SpectralPolicy, variant: str) -> torch.Tensor:
    """Return the weights used in ``I-2wP`` for a local policy."""

    if variant == "competitive_subspace":
        return relative_soft_subspace_weights(policy)
    return policy.low_mode_weights


def gate_diagnostics(
    gradient_coefficients: torch.Tensor,
    policy: SpectralPolicy,
    variant: str = "lambda2",
) -> GateDiagnostics:
    """Return the gate and the local activities from the same calculation.

    Keeping these values together prevents observability code from silently
    reimplementing a policy with slightly different numerical conventions.
    """

    if variant not in {
        "lambda2",
        "alignment",
        "competitive",
        "competitive_subspace",
        "guard",
        "gad",
    }:
        raise ValueError(f"unknown gate variant: {variant}")
    base_variant = _base_gate_variant(variant)
    coeffs = gradient_coefficients.to(torch.float64).reshape(-1)
    activity = coeffs.square()
    soft = torch.sum(policy.low_mode_weights * activity)
    extra_negative_weights = policy.negative_mode_weights * (1.0 - policy.low_mode_weights).square()
    extra_negative = torch.sum(extra_negative_weights * activity)
    competitive_denominator = soft + extra_negative
    competitive_fraction = torch.where(
        competitive_denominator > 0,
        soft / competitive_denominator.clamp_min(torch.finfo(coeffs.dtype).tiny),
        torch.zeros_like(competitive_denominator),
    ).clamp(0.0, 1.0)

    if base_variant == "gad":
        effective = torch.ones_like(policy.gate)
    elif base_variant == "lambda2":
        effective = policy.gate
    else:
        if base_variant == "alignment":
            denominator = torch.sum(activity)
            fraction = torch.where(
                denominator > 0,
                soft / denominator.clamp_min(torch.finfo(coeffs.dtype).tiny),
                torch.zeros_like(denominator),
            ).clamp(0.0, 1.0)
        else:
            fraction = competitive_fraction
        competitive = policy.gate + (1.0 - policy.gate) * fraction
        if base_variant == "guard":
            z1 = policy.lowest_normalized_eigenvalue / policy.temperature
            minimum_boundary = torch.sigmoid(z1)
            boundary_guard = 4.0 * minimum_boundary * (1.0 - minimum_boundary)
            effective = 1.0 - (1.0 - competitive) * (1.0 - boundary_guard)
        else:
            effective = competitive

    weights = policy.low_mode_weights
    if weights.numel() > 1:
        entropy = -torch.sum(
            weights * torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        ) / math.log(weights.numel())
    else:
        entropy = torch.zeros_like(policy.gate)
    policy_reflection_weights = reflection_weights(policy, variant)
    lowest_reflection = 1.0 - 2.0 * effective * policy_reflection_weights[0]
    return GateDiagnostics(
        lambda2_gate=policy.gate,
        effective_gate=effective,
        soft_activity=soft,
        extra_negative_activity=extra_negative,
        activity_fraction=competitive_fraction,
        spectral_entropy=entropy,
        reflection_weights=policy_reflection_weights,
        lowest_reflection=lowest_reflection,
    )


def inverse_rms_pair_length(coords: torch.Tensor) -> torch.Tensor:
    r"""Return ``(mean_{i<j} r_ij^-2)^-1/2`` at the current geometry.

    Unlike a hard minimum-distance guard, this length is smooth wherever no
    atoms coincide.  It is permutation and rigid-motion invariant, scales
    linearly with the coordinates, and tends to zero at a pair collision.
    """

    xyz = coords.reshape(-1, 3).to(torch.float64)
    if xyz.shape[0] < 2:
        raise ValueError("intrinsic GAD requires at least two atoms")
    distances = torch.pdist(xyz)
    if bool((distances <= 0).any().item()):
        raise ValueError("intrinsic GAD is undefined for coincident atoms")
    return torch.rsqrt(torch.mean(distances.reciprocal().square()))


def pointwise_step_coefficients(
    gradient_coefficients: torch.Tensor,
    eigenvalues: torch.Tensor,
    policy: SpectralPolicy,
    radius_mw: float,
    gate_variant: str = "lambda2",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the closed-form gated step in the Hessian eigenbasis.

    With ``b_i = (1 - 2 w r_i) g_i`` and
    ``mu = ||b|| / R``, the step is

    .. math::

        a_i = -\frac{b_i}{\sqrt{\lambda_i^2 + \mu^2}}.

    Since every denominator is at least ``mu``, ``||a|| <= R``.  This is an
    algebraic consequence, not an a posteriori cap.  Multiplying the PES by a
    positive constant multiplies ``b``, ``lambda``, and ``mu`` equally and
    therefore leaves the step unchanged.
    """

    if radius_mw <= 0:
        raise ValueError("radius_mw must be positive")
    coeffs = gradient_coefficients.to(torch.float64).reshape(-1)
    evals = eigenvalues.to(dtype=coeffs.dtype, device=coeffs.device).reshape(-1)
    if coeffs.shape != evals.shape or coeffs.shape != policy.low_mode_weights.shape:
        raise ValueError("gradient, eigenvalue, and spectral-weight shapes must agree")

    gate = effective_gate_weight(coeffs, policy, gate_variant)
    reflection = 1.0 - 2.0 * gate * reflection_weights(policy, gate_variant)
    modified_gradient = reflection * coeffs
    modified_norm = torch.linalg.vector_norm(modified_gradient)
    if float(modified_norm.item()) == 0.0:
        return torch.zeros_like(modified_gradient), torch.zeros_like(modified_norm)

    regularizer = modified_norm / radius_mw
    denominator = torch.sqrt(evals.square() + regularizer.square())
    step = -modified_gradient / denominator
    return step, regularizer


def _evaluate(
    predict_fn: PredictFn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list[str],
    cfg: IntrinsicGADConfig,
) -> _State:
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    energy = torch.as_tensor(
        out["energy"],
        dtype=torch.float64,
        device=coords.device,
    ).reshape(())
    forces = torch.as_tensor(
        out["forces"],
        dtype=torch.float64,
        device=coords.device,
    ).reshape(-1, 3)
    hessian = torch.as_tensor(
        out["hessian"],
        dtype=torch.float64,
        device=coords.device,
    ).reshape(coords.numel(), coords.numel())
    hessian = 0.5 * (hessian + hessian.T)
    if not (
        bool(torch.isfinite(energy).item())
        and bool(torch.isfinite(forces).all().item())
        and bool(torch.isfinite(hessian).all().item())
    ):
        raise FloatingPointError("calculator returned a non-finite local evaluation")

    evals, modes_mw, _ = vib_eig(
        hessian,
        coords,
        atomsymbols,
        purify=cfg.purify_hessian,
    )
    _, _, _, inv_sqrt_mass = get_mass_weights(
        atomsymbols,
        device=coords.device,
    )
    gradient_cart = -forces.reshape(-1)
    gradient_mw = inv_sqrt_mass * gradient_cart
    policy = smooth_spectral_policy(evals, cfg.spectral_temperature)
    gradient_coeffs = modes_mw.T @ gradient_mw
    diagnostics = gate_diagnostics(gradient_coeffs, policy, cfg.gate_variant)
    return _State(
        coords=coords,
        energy=energy,
        forces=forces,
        evals=evals,
        modes_mw=modes_mw,
        gradient_mw=gradient_mw,
        policy=policy,
        gradient_coefficients=gradient_coeffs,
        gate_diagnostics=diagnostics,
        effective_gate=diagnostics.effective_gate,
        n_neg=int((evals < -cfg.index_threshold).sum().item()),
        force_norm=force_mean(forces),
        force_max=force_max(forces),
    )


def _make_result(
    state: _State,
    *,
    converged: bool,
    steps: int,
    started: float,
    failure_type: str | None,
    history: list[IntrinsicGADStep],
) -> IntrinsicGADResult:
    return IntrinsicGADResult(
        converged=converged,
        converged_step=steps if converged else None,
        total_steps=steps,
        n_evaluations=steps + 1,
        final_coords=state.coords.detach().cpu(),
        final_energy=float(state.energy.item()),
        final_n_neg=state.n_neg,
        final_force_norm=state.force_norm,
        final_force_max=state.force_max,
        final_eig0=float(state.evals[0].item()),
        final_eig1=(float(state.evals[1].item()) if state.evals.numel() > 1 else math.nan),
        final_gate_weight=float(state.effective_gate.item()),
        wall_time_s=time.time() - started,
        failure_type=failure_type,
        history=tuple(history),
    )


def run_intrinsic_gad(
    predict_fn: PredictFn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    cfg: IntrinsicGADConfig | None = None,
    observer: Callable[[IntrinsicGADObservation], None] | None = None,
) -> IntrinsicGADResult:
    r"""Iterate the strictly pointwise smooth-index GAD map.

    In mass-weighted vibrational coordinates, each iteration is

    .. math::

        g_i &= v_i^T M^{-1/2} g,\\
        b_i &= (1 - 2 w p_i)g_i,\\
        R(q) &= \eta\,\ell(q)\sqrt{\sum_a m_a},\\
        \mu(q) &= \|b\|/R(q),\\
        a_i &= -b_i/\sqrt{\lambda_i^2+\mu(q)^2},\\
        q^+ &= q + M^{-1/2}Va.

    Every quantity on the right is evaluated at the same current point.  The
    returned history is diagnostic output only and never affects the map.
    Likewise, ``observer`` receives detached CPU snapshots and has no return
    channel into the update.
    """

    cfg = cfg or IntrinsicGADConfig()
    started = time.time()
    coords = coords0.detach().clone().to(torch.float64).reshape(-1, 3)
    atomic_nums = atomic_nums.to(device=coords.device)
    atomsymbols = atomic_nums_to_symbols(atomic_nums)
    masses, _, _, inv_sqrt_mass = get_mass_weights(
        atomsymbols,
        device=coords.device,
    )
    mass_total = float(masses.sum().item())
    history: list[IntrinsicGADStep] = []

    for iteration in range(cfg.max_steps + 1):
        state = _evaluate(predict_fn, coords, atomic_nums, atomsymbols, cfg)

        force_value = force_value_from_criterion(
            state.forces,
            cfg.force_criterion,
        )
        converged = is_ts_converged(
            state.n_neg,
            force_value,
            cfg.force_threshold,
            criterion=cfg.force_criterion,
        )
        terminal = converged or iteration == cfg.max_steps
        if terminal and observer is not None:
            diagnostics = state.gate_diagnostics
            observer(
                IntrinsicGADObservation(
                    evaluation=iteration,
                    iteration=iteration,
                    wall_time_s=time.time() - started,
                    energy=float(state.energy.item()),
                    forces=state.forces.detach().cpu().clone(),
                    coords=state.coords.detach().cpu().clone(),
                    eigenvalues=state.evals.detach().cpu().clone(),
                    n_neg=state.n_neg,
                    lambda2_gate=float(diagnostics.lambda2_gate.item()),
                    effective_gate=float(diagnostics.effective_gate.item()),
                    spectral_scale=float(state.policy.scale.item()),
                    low_mode_weights=state.policy.low_mode_weights.detach().cpu().clone(),
                    negative_mode_weights=(
                        state.policy.negative_mode_weights.detach().cpu().clone()
                    ),
                    gradient_coefficients=(state.gradient_coefficients.detach().cpu().clone()),
                    soft_activity=float(diagnostics.soft_activity.item()),
                    extra_negative_activity=float(diagnostics.extra_negative_activity.item()),
                    activity_fraction=float(diagnostics.activity_fraction.item()),
                    spectral_entropy=float(diagnostics.spectral_entropy.item()),
                    lowest_reflection=float(diagnostics.lowest_reflection.item()),
                    geometric_length=None,
                    local_radius=None,
                    regularizer=None,
                    step_mw_rms=None,
                    step_cart=None,
                    terminal=True,
                )
            )
        if converged:
            return _make_result(
                state,
                converged=True,
                steps=iteration,
                started=started,
                failure_type=None,
                history=history,
            )
        if iteration == cfg.max_steps:
            return _make_result(
                state,
                converged=False,
                steps=iteration,
                started=started,
                failure_type="max_steps",
                history=history,
            )

        geometric_length = inverse_rms_pair_length(state.coords)
        local_radius = cfg.step_fraction * float(geometric_length.item())
        radius_mw = local_radius * math.sqrt(mass_total)
        gradient_coeffs = state.gradient_coefficients
        step_coeffs, regularizer = pointwise_step_coefficients(
            gradient_coeffs,
            state.evals,
            state.policy,
            radius_mw,
            cfg.gate_variant,
        )
        step_mw = state.modes_mw @ step_coeffs
        step_cart = inv_sqrt_mass * step_mw
        step_rms = float(torch.linalg.vector_norm(step_mw).item()) / math.sqrt(mass_total)

        if observer is not None:
            diagnostics = state.gate_diagnostics
            observer(
                IntrinsicGADObservation(
                    evaluation=iteration,
                    iteration=iteration,
                    wall_time_s=time.time() - started,
                    energy=float(state.energy.item()),
                    forces=state.forces.detach().cpu().clone(),
                    coords=state.coords.detach().cpu().clone(),
                    eigenvalues=state.evals.detach().cpu().clone(),
                    n_neg=state.n_neg,
                    lambda2_gate=float(diagnostics.lambda2_gate.item()),
                    effective_gate=float(diagnostics.effective_gate.item()),
                    spectral_scale=float(state.policy.scale.item()),
                    low_mode_weights=state.policy.low_mode_weights.detach().cpu().clone(),
                    negative_mode_weights=(
                        state.policy.negative_mode_weights.detach().cpu().clone()
                    ),
                    gradient_coefficients=(state.gradient_coefficients.detach().cpu().clone()),
                    soft_activity=float(diagnostics.soft_activity.item()),
                    extra_negative_activity=float(diagnostics.extra_negative_activity.item()),
                    activity_fraction=float(diagnostics.activity_fraction.item()),
                    spectral_entropy=float(diagnostics.spectral_entropy.item()),
                    lowest_reflection=float(diagnostics.lowest_reflection.item()),
                    geometric_length=float(geometric_length.item()),
                    local_radius=local_radius,
                    regularizer=float(regularizer.item()),
                    step_mw_rms=step_rms,
                    step_cart=step_cart.reshape_as(state.coords).detach().cpu().clone(),
                    terminal=False,
                )
            )

        if cfg.record_history:
            history.append(
                IntrinsicGADStep(
                    iteration=iteration,
                    energy=float(state.energy.item()),
                    force_max=state.force_max,
                    n_neg=state.n_neg,
                    eig0=float(state.evals[0].item()),
                    eig1=(float(state.evals[1].item()) if state.evals.numel() > 1 else math.nan),
                    gate_weight=float(state.effective_gate.item()),
                    spectral_scale=float(state.policy.scale.item()),
                    geometric_length=float(geometric_length.item()),
                    local_radius=local_radius,
                    regularizer=float(regularizer.item()),
                    step_rms=step_rms,
                )
            )

        coords = (state.coords + step_cart.reshape_as(state.coords)).detach()

    raise AssertionError("unreachable")
