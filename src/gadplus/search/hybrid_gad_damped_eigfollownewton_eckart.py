"""Use GAD far from the saddle and damped eigenvector-following Newton near the saddle."""

import torch

MASS_AMU = {
    1: 1.008,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    15: 30.974,
    16: 32.065,
    17: 35.453,
    35: 79.904,
    53: 126.904,
}


def masses_from_z(z, device=None, dtype=torch.float64):
    if device is None:
        device = z.device if isinstance(z, torch.Tensor) else None

    vals = []
    for a in torch.as_tensor(z).detach().cpu().tolist():
        a = int(a)
        if a not in MASS_AMU:
            raise KeyError(f"No mass available for atomic number Z={a}.")
        vals.append(MASS_AMU[a])

    return torch.tensor(vals, dtype=dtype, device=device)


def eckart_internal_basis(coords, masses, tol=None):
    """
    Builds the internal-coordinate basis U_int in mass-weighted Cartesian space.

    coords: shape (N, 3)
    masses: shape (N,)

    Returns:
        P:     projector onto internal space, shape (3N, 3N)
        U_int: orthonormal internal basis, shape (3N, 3N-r)
        U_ext: orthonormal external basis, shape (3N, r)

    For nonlinear molecules, r = 6.
    For linear molecules, r = 5.
    """

    dtype = coords.dtype
    device = coords.device

    if tol is None:
        tol = 1.0e-10 if dtype == torch.float64 else 1.0e-6

    coords = coords.to(dtype=dtype, device=device)
    masses = masses.to(dtype=dtype, device=device)

    N = coords.shape[0]
    dim = 3 * N

    sqrt_m = torch.sqrt(masses)

    com = (coords * masses[:, None]).sum(dim=0) / masses.sum()
    r = coords - com

    cols = []

    # Translations: sqrt(m_i) e_alpha
    for a in range(3):
        c = torch.zeros((N, 3), dtype=dtype, device=device)
        c[:, a] = sqrt_m
        cols.append(c.reshape(-1))

    # Rotations: sqrt(m_i) (omega x r_i)
    axes = torch.eye(3, dtype=dtype, device=device)
    for a in range(3):
        omega = axes[a].expand_as(r)
        rot = torch.cross(omega, r, dim=1)
        cols.append((rot * sqrt_m[:, None]).reshape(-1))

    B = torch.stack(cols, dim=1)

    # Drop zero columns, then remove linear dependencies.
    norms = torch.linalg.vector_norm(B, dim=0)
    keep = norms > tol * norms.max().clamp_min(torch.tensor(1.0, dtype=dtype, device=device))
    B = B[:, keep] / norms[keep]

    U, S, _ = torch.linalg.svd(B, full_matrices=False)

    rank = int(
        torch.sum(
            S > tol * S.max().clamp_min(torch.tensor(1.0, dtype=dtype, device=device))
        ).item()
    )

    U_ext = U[:, :rank]

    # Complete orthonormal basis. The remaining columns span the internal space.
    Q, _ = torch.linalg.qr(U_ext, mode="complete")
    U_ext = Q[:, :rank]
    U_int = Q[:, rank:]

    I = torch.eye(dim, dtype=dtype, device=device)
    P = I - U_ext @ U_ext.T
    P = 0.5 * (P + P.T)

    return P, U_int, U_ext


def _symmetrize(H):
    return 0.5 * (H + H.transpose(-1, -2))


def _cartesian_trust_limit(step_x, trust_radius):
    if trust_radius is None:
        return step_x

    r = torch.as_tensor(trust_radius, dtype=step_x.dtype, device=step_x.device)
    norm = torch.linalg.vector_norm(step_x)

    scale = torch.minimum(
        torch.ones((), dtype=step_x.dtype, device=step_x.device),
        r / (norm + torch.finfo(step_x.dtype).eps),
    )

    return scale * step_x


def _internal_mass_weighted_state(force_cart, hessian_cart, coords, masses):
    """
    Converts Cartesian force/Hessian into the Eckart-projected,
    mass-weighted internal subspace.

    Assumes:
        force_cart = -grad_x V
        hessian_cart = grad_x^2 V

    Returns internal force F_i and internal Hessian H_i.
    """

    force_shape = force_cart.shape
    dtype = force_cart.dtype
    device = force_cart.device

    F_x = force_cart.reshape(-1)
    H_x = _symmetrize(hessian_cart.to(dtype=dtype, device=device))

    coords = coords.to(dtype=dtype, device=device)
    masses = masses.to(dtype=dtype, device=device)

    n = F_x.numel()

    if H_x.shape != (n, n):
        raise ValueError(
            "hessian_cart must have shape "
            "(force_cart.numel(), force_cart.numel())."
        )

    if coords.shape[-1] != 3 or coords.numel() != n:
        raise ValueError(
            "coords must have shape (N, 3), with 3*N == force_cart.numel()."
        )

    if masses.shape != (coords.shape[0],):
        raise ValueError("masses must have shape (N,).")

    _, U_int, U_ext = eckart_internal_basis(coords, masses)

    sqrt_m3 = torch.sqrt(masses).repeat_interleave(3)
    inv_sqrt_m3 = 1.0 / sqrt_m3

    # Cartesian -> mass-weighted coordinates.
    F_q = inv_sqrt_m3 * F_x
    H_q = inv_sqrt_m3[:, None] * H_x * inv_sqrt_m3[None, :]
    H_q = _symmetrize(H_q)

    # Restrict to internal Eckart subspace.
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


def _internal_vector_to_cartesian(vec_i, state):
    """
    Converts an internal mass-weighted vector to Cartesian coordinates.
    """

    vec_q = state["U_int"] @ vec_i
    vec_x = state["inv_sqrt_m3"] * vec_q

    return vec_x.reshape(state["force_shape"])


def _internal_step_to_cartesian(step_i, state, trust_radius=None):
    """
    Converts an internal mass-weighted step back to Cartesian coordinates.
    """

    step_x = _internal_vector_to_cartesian(step_i, state)
    step_x = _cartesian_trust_limit(step_x, trust_radius)

    return step_x


def damped_eigenfollowing_step(
    F_i,
    eigvals,
    eigvecs,
    target_mode=0,
    min_curvature=1.0e-8,
    trust_radius=None,
    trust_norm_fn=None,
    max_iter=80,
):
    """
    Damped index-1 eigenvector-following step in the internal,
    mass-weighted subspace.

    F_i:      internal force, shape (m,)
    eigvals:  Hessian eigenvalues, shape (m,)
    eigvecs:  Hessian eigenvectors, shape (m, m)

    Returns:
        step_i, mu
    """

    F_eig = eigvecs.T @ F_i

    curv = eigvals.abs().clamp_min(min_curvature)

    sign = torch.ones_like(curv)
    sign[target_mode] = -1.0

    def step_for_mu(mu):
        step_eig = sign * F_eig / (curv + mu)
        return eigvecs @ step_eig

    def step_norm(step):
        if trust_norm_fn is None:
            return torch.linalg.vector_norm(step)
        return trust_norm_fn(step)

    step = step_for_mu(torch.zeros((), dtype=F_i.dtype, device=F_i.device))

    if trust_radius is None:
        return step, torch.zeros((), dtype=F_i.dtype, device=F_i.device)

    delta = torch.as_tensor(trust_radius, dtype=F_i.dtype, device=F_i.device)

    if step_norm(step) <= delta:
        return step, torch.zeros((), dtype=F_i.dtype, device=F_i.device)

    lo = torch.zeros((), dtype=F_i.dtype, device=F_i.device)
    hi = torch.ones((), dtype=F_i.dtype, device=F_i.device)

    while step_norm(step_for_mu(hi)) > delta:
        hi = 2.0 * hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if step_norm(step_for_mu(mid)) > delta:
            lo = mid
        else:
            hi = mid

    mu = hi
    step = step_for_mu(mu)
    return step, mu


def projected_gad_step(
    force_cart,
    hessian_cart,
    coords,
    masses,
    target_mode=0,
    gad_dt=1.0e-2,
    trust_radius=None,
):
    """
    Eckart-projected GAD step.

    Assumes:
        force_cart = -grad_x V
        hessian_cart = grad_x^2 V

    Computes GAD in the internal mass-weighted subspace:

        d_i = F_i - 2 <F_i, v> v

    and returns a Cartesian step:

        x_next = x + step_cart
    """

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

    v = eigvecs[:, target_mode]

    # GAD direction in internal mass-weighted coordinates.
    gad_dir_i = F_i - 2.0 * torch.dot(F_i, v) * v

    step_i = gad_dt * gad_dir_i

    direction_cart = _internal_vector_to_cartesian(
        vec_i=gad_dir_i,
        state=state,
    )
    step_cart = _internal_step_to_cartesian(
        step_i=step_i,
        state=state,
        trust_radius=trust_radius,
    )

    info = {
        "method": "projected_gad",
        "internal_eigvals": eigvals,
        "target_eigval": eigvals[target_mode],
        "target_mode_vec_internal": v,
        "num_external_modes": state["U_ext"].shape[1],
        "num_internal_modes": state["U_int"].shape[1],
        "direction_cart": direction_cart,
        "direction_norm_cart": torch.linalg.vector_norm(direction_cart),
        "force_norm_internal": torch.linalg.vector_norm(F_i),
        "step_norm_internal": torch.linalg.vector_norm(step_i),
        "step_norm_cart": torch.linalg.vector_norm(step_cart),
    }

    return step_cart, info


def projected_index1_newton_step(
    force_cart,
    hessian_cart,
    coords,
    masses,
    target_mode=0,
    min_curvature=1.0e-8,
    trust_radius=None,
):
    """
    Eckart-projected damped eigenvector-following Newton step.

    Assumes:
        force_cart = -grad_x V
        hessian_cart = grad_x^2 V
    """

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

    def cartesian_step_norm(step_i):
        step_x = _internal_step_to_cartesian(
            step_i=step_i,
            state=state,
            trust_radius=None,
        )
        return torch.linalg.vector_norm(step_x)

    step_i, mu = damped_eigenfollowing_step(
        F_i=F_i,
        eigvals=eigvals,
        eigvecs=eigvecs,
        target_mode=target_mode,
        min_curvature=min_curvature,
        trust_radius=trust_radius,
        trust_norm_fn=cartesian_step_norm,
    )

    direction_cart = _internal_vector_to_cartesian(
        vec_i=step_i,
        state=state,
    )
    step_cart = _internal_step_to_cartesian(
        step_i=step_i,
        state=state,
        trust_radius=None,
    )

    info = {
        "method": "projected_damped_eigenvector_following_newton",
        "internal_eigvals": eigvals,
        "target_eigval": eigvals[target_mode],
        "damping_mu": mu,
        "num_external_modes": state["U_ext"].shape[1],
        "num_internal_modes": state["U_int"].shape[1],
        "direction_cart": direction_cart,
        "direction_norm_cart": torch.linalg.vector_norm(direction_cart),
        "force_norm_internal": torch.linalg.vector_norm(F_i),
        "step_norm_internal": torch.linalg.vector_norm(step_i),
        "step_norm_cart": torch.linalg.vector_norm(step_cart),
    }

    return step_cart, info


def projected_hybrid_gad_newton_step(
    force_cart,
    hessian_cart,
    coords,
    masses,
    target_mode=0,
    gad_dt=1.0e-2,
    switch_based_on_hessian_eigval=False,
    switch_force=1.0e-3,
    min_curvature=1.0e-8,
    trust_radius=None,
):
    """
    Uses projected GAD far from the saddle and projected damped eigenvector-following Newton near the saddle.

    By default, the switch criterion uses the internal mass-weighted force norm.
    If switch_based_on_hessian_eigval is True, GAD is used until the internal
    mass-weighted Hessian clearly has index 1.
    """

    state = _internal_mass_weighted_state(
        force_cart=force_cart,
        hessian_cart=hessian_cart,
        coords=coords,
        masses=masses,
    )

    F_i = state["F_i"]
    H_i = state["H_i"]

    force_norm_internal = torch.linalg.vector_norm(F_i)
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

    if switch_based_on_hessian_eigval:
        use_newton = bool(hessian_has_clear_index1.detach().cpu().item())
    else:
        use_newton = bool((force_norm_internal <= switch_force).detach().cpu().item())

    if not use_newton:
        v = eigvecs[:, target_mode]

        # Projected GAD direction.
        direction_i = F_i - 2.0 * torch.dot(F_i, v) * v
        step_i = gad_dt * direction_i

        step_cart = _internal_step_to_cartesian(
            step_i=step_i,
            state=state,
            trust_radius=trust_radius,
        )
        damping_mu = torch.zeros((), dtype=F_i.dtype, device=F_i.device)
        method = "projected_gad"

    else:
        def cartesian_step_norm(step_i):
            step_x = _internal_step_to_cartesian(
                step_i=step_i,
                state=state,
                trust_radius=None,
            )
            return torch.linalg.vector_norm(step_x)

        step_i, damping_mu = damped_eigenfollowing_step(
            F_i=F_i,
            eigvals=eigvals,
            eigvecs=eigvecs,
            target_mode=target_mode,
            min_curvature=min_curvature,
            trust_radius=trust_radius,
            trust_norm_fn=cartesian_step_norm,
        )
        direction_i = step_i

        step_cart = _internal_step_to_cartesian(
            step_i=step_i,
            state=state,
            trust_radius=None,
        )
        method = "projected_damped_eigenvector_following_newton"

    direction_cart = _internal_vector_to_cartesian(
        vec_i=direction_i,
        state=state,
    )
    info = {
        "method": method,
        "internal_eigvals": eigvals,
        "target_eigval": eigvals[target_mode],
        "damping_mu": damping_mu,
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
    }

    return step_cart, info


if __name__ == "__main__":
    # coords:  shape (natoms, 3)
    # force:   shape (natoms, 3)
    # hessian: shape (3 * natoms, 3 * natoms)
    # z:       shape (natoms,)

    natoms = 10
    coords = torch.randn(natoms, 3)
    force = torch.randn(natoms, 3)
    hessian = torch.randn(3 * natoms, 3 * natoms)
    z = torch.tensor([6] * natoms)  # carbon atoms

    coords = coords.double()
    force = force.double()
    hessian = _symmetrize(hessian.double())

    masses = masses_from_z(z, device=coords.device, dtype=coords.dtype)

    step, info = projected_hybrid_gad_newton_step(
        force_cart=force,
        hessian_cart=hessian,
        coords=coords,
        masses=masses,
        target_mode=0,
        gad_dt=1.0e-2,
        switch_based_on_hessian_eigval=False,
        switch_force=1.0e-3,
        min_curvature=1.0e-8,
        trust_radius=0.05,
    )

    coords_next = coords + step
