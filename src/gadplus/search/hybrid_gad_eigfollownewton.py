"""Use GAD far from the saddle and eigenvector-following Newton near the saddle."""

import torch


def _symmetrize(H: torch.Tensor) -> torch.Tensor:
    return 0.5 * (H + H.transpose(-1, -2))


def _trust_limit(step: torch.Tensor, trust_radius: float | None) -> torch.Tensor:
    if trust_radius is None:
        return step

    r = torch.as_tensor(trust_radius, dtype=step.dtype, device=step.device)
    norm = torch.linalg.vector_norm(step)
    eps = torch.finfo(step.dtype).eps

    scale = torch.minimum(
        torch.ones((), dtype=step.dtype, device=step.device),
        r / (norm + eps),
    )
    return scale * step


def gad_direction_from_force(
    force: torch.Tensor,
    hessian: torch.Tensor,
    target_mode: int = 0,
):
    """
    Original GAD direction from force and Hessian.

    Uses

        dx/dt = F - 2 <F, v> v,

    where F = -grad V and v is the selected Hessian eigenvector.
    """

    force_shape = force.shape
    F = force.reshape(-1)
    H = _symmetrize(hessian)

    if H.shape != (F.numel(), F.numel()):
        raise ValueError("hessian must have shape (force.numel(), force.numel()).")

    eigvals, eigvecs = torch.linalg.eigh(H)
    v = eigvecs[:, target_mode]

    direction = F - 2.0 * torch.dot(F, v) * v

    info = {
        "eigvals": eigvals,
        "mode": v.reshape(force_shape),
    }

    return direction.reshape(force_shape), info


def index1_saddle_step_from_force(
    force: torch.Tensor,
    hessian: torch.Tensor,
    target_mode: int = 0,
    min_curvature: float = 1.0e-6,
    trust_radius: float | None = None,
    zero_mode_cutoff: float | None = None,
):
    """
    Stabilized eigenvector-following Newton step for an index-1 saddle.

    Input:
        force   = -grad V, shape (n,) or any shape that flattens to n
        hessian = grad^2 V, shape (n, n)

    Output:
        step such that x_next = x + step

    The target mode is treated as the ascent mode.
    All other modes are treated as descent modes.
    """

    force_shape = force.shape
    F = force.reshape(-1)
    H = _symmetrize(hessian)

    if H.shape != (F.numel(), F.numel()):
        raise ValueError("hessian must have shape (force.numel(), force.numel()).")

    eigvals, eigvecs = torch.linalg.eigh(H)

    lam_abs = eigvals.abs().clamp_min(min_curvature)

    # Positive denominators give descent-like motion.
    # Negative denominator on target_mode gives ascent-like motion.
    denom = lam_abs.clone()
    denom[target_mode] = -lam_abs[target_mode]

    # Optional: project out translational/rotational or other near-zero modes.
    if zero_mode_cutoff is not None:
        zero_mask = eigvals.abs() < zero_mode_cutoff
        zero_mask[target_mode] = False
        denom = torch.where(
            zero_mask,
            torch.full_like(denom, torch.inf),
            denom,
        )

    F_eig = eigvecs.T @ F
    step_eig = F_eig / denom
    step = eigvecs @ step_eig

    step = _trust_limit(step, trust_radius)

    inertia_tol = min_curvature
    info = {
        "eigvals": eigvals,
        "denom": denom,
        "num_negative_modes": torch.sum(eigvals < -inertia_tol),
        "num_zero_modes": torch.sum(eigvals.abs() <= inertia_tol),
        "num_positive_modes": torch.sum(eigvals > inertia_tol),
        "target_eigval": eigvals[target_mode],
        "step_norm": torch.linalg.vector_norm(step),
    }

    return step.reshape(force_shape), info


def hybrid_gad_newton_step_from_force(
    force: torch.Tensor,
    hessian: torch.Tensor,
    target_mode: int = 0,
    gad_dt: float = 1.0e-2,
    switch_force: float = 1.0e-3,
    min_curvature: float = 1.0e-6,
    trust_radius: float | None = None,
    zero_mode_cutoff: float | None = None,
):
    """
    Use GAD far from the saddle and eigenvector-following Newton near the saddle.

    Criterion:

        ||F|| > switch_force  -> GAD step
        ||F|| <= switch_force -> eigenvector-following Newton step
    """

    F = force.reshape(-1)
    force_norm = torch.linalg.vector_norm(F).detach().item()

    if force_norm > switch_force:
        direction, info = gad_direction_from_force(
            force,
            hessian,
            target_mode=target_mode,
        )
        step = gad_dt * direction.reshape(-1)
        step = _trust_limit(step, trust_radius)
        info["method"] = "gad"
        info["force_norm"] = torch.as_tensor(
            force_norm, dtype=force.dtype, device=force.device
        )
        return step.reshape_as(force), info

    step, info = index1_saddle_step_from_force(
        force,
        hessian,
        target_mode=target_mode,
        min_curvature=min_curvature,
        trust_radius=trust_radius,
        zero_mode_cutoff=zero_mode_cutoff,
    )
    info["method"] = "eigenvector_following_newton"
    info["force_norm"] = torch.as_tensor(
        force_norm, dtype=force.dtype, device=force.device
    )
    return step, info


if __name__ == "__main__":
    # x:       shape (n, 3)
    # force:   shape (n,) or, for example, (natoms, 3)
    # hessian: shape (n, n), where n = force.numel()

    x = torch.randn(10, 3)
    force = torch.randn(10)
    hessian = torch.randn(10, 10)

    force = force.double()
    hessian = hessian.double()

    step, info = hybrid_gad_newton_step_from_force(
        force,
        hessian,
        target_mode=0,          # lowest Hessian eigenmode
        gad_dt=1.0e-2,
        switch_force=1.0e-3,
        min_curvature=1.0e-6,
        trust_radius=0.05,
    )

    x_next = x + step.reshape_as(x)