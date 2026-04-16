"""IRC (Intrinsic Reaction Coordinate) validation using Sella.

After finding a TS, run IRC forward and backward to verify that it connects
the intended reactant and product. Uses Sella's IRC optimizer with HIP
via the ASE adapter.

Validation outputs include:
1. RMSD-based endpoint matching (legacy compatibility)
2. Bond-topology matching via graph isomorphism (permutation-invariant)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import natural_cutoffs, neighbor_list

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional runtime dependency
    nx = None

from gadplus.projection import Z_TO_SYMBOL
from gadplus.geometry.alignment import aligned_rmsd_by_element


@dataclass
class IRCResult:
    """Result of IRC validation."""
    intended: bool              # Both reactant and product matched
    half_intended: bool         # Only one endpoint matched
    topology_intended: bool     # Graph match for both endpoints (direction-agnostic)
    topology_half_intended: bool
    forward_coords: Optional[np.ndarray]    # Final geometry from forward IRC
    reverse_coords: Optional[np.ndarray]    # Final geometry from reverse IRC
    rmsd_to_reactant: Optional[float]
    rmsd_to_product: Optional[float]
    forward_rmsd_to_reactant: Optional[float]
    forward_rmsd_to_product: Optional[float]
    reverse_rmsd_to_reactant: Optional[float]
    reverse_rmsd_to_product: Optional[float]
    forward_graph_matches_reactant: bool
    forward_graph_matches_product: bool
    reverse_graph_matches_reactant: bool
    reverse_graph_matches_product: bool
    # Endpoint spectral diagnostics (filled in when predict_fn is passed to
    # score_endpoints). `n_neg_vib` is the count of negative eigenvalues in
    # the Eckart-projected vibrational Hessian at the endpoint. A true
    # minimum has n_neg_vib == 0. `min_vib_eig` is the smallest vibrational
    # eigenvalue (signed) — negative values indicate the endpoint is still
    # on a saddle or ridge.
    forward_n_neg_vib: Optional[int] = None
    reverse_n_neg_vib: Optional[int] = None
    forward_min_vib_eig: Optional[float] = None
    reverse_min_vib_eig: Optional[float] = None
    error: Optional[str] = None
    topology_error: Optional[str] = None


def coords_to_bond_graph(
    coords: np.ndarray | torch.Tensor,
    atomic_nums: torch.Tensor,
    cutoff_scale: float = 1.2,
):
    """Build an element-labeled bond graph from coordinates.

    Bonds are detected with ASE neighbor lists using atom-wise cutoffs from
    covalent radii scaled by `cutoff_scale`.
    """
    if nx is None:
        raise ImportError("networkx is required for topology validation")

    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy().reshape(-1, 3)
    else:
        coords_np = np.asarray(coords, dtype=float).reshape(-1, 3)

    nums = atomic_nums.detach().cpu().numpy().astype(int).reshape(-1)
    if coords_np.shape[0] != nums.shape[0]:
        raise ValueError("coords and atomic_nums have inconsistent atom counts")

    atoms = Atoms(numbers=nums.tolist(), positions=coords_np)
    cutoffs = natural_cutoffs(atoms, mult=cutoff_scale)
    i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)

    graph = nx.Graph()
    for i, z in enumerate(nums.tolist()):
        graph.add_node(int(i), Z=int(z))

    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        if i < j:
            graph.add_edge(int(i), int(j))

    return graph


def bond_graphs_match(graph1, graph2) -> bool:
    """Check element-aware graph isomorphism between two molecular graphs."""
    if nx is None:
        raise ImportError("networkx is required for topology validation")
    if graph1 is None or graph2 is None:
        return False
    return bool(
        nx.is_isomorphic(
            graph1,
            graph2,
            node_match=lambda a, b: a.get("Z") == b.get("Z"),
        )
    )


def _to_numpy_coords(coords: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if coords is None:
        return None
    return coords.detach().cpu().numpy().reshape(-1, 3)


def _coords_rmsd(coords_a: np.ndarray, coords_b: np.ndarray, atomic_nums: torch.Tensor) -> float:
    nums = atomic_nums.detach().cpu().numpy().astype(int)
    return float(aligned_rmsd_by_element(coords_a, coords_b, nums))


def _min_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
    vals = [x for x in (a, b) if x is not None]
    return min(vals) if vals else None


def _endpoint_spectral(
    coords: np.ndarray,
    atomic_nums: torch.Tensor,
    predict_fn,
) -> tuple[Optional[int], Optional[float]]:
    """Count negative vibrational eigenvalues and return min eigenvalue
    at an endpoint geometry. Returns ``(None, None)`` on failure so that a
    bad HIP call doesn't propagate into the run.
    """
    from gadplus.projection import atomic_nums_to_symbols, vib_eig

    try:
        device = atomic_nums.device if hasattr(atomic_nums, "device") else "cpu"
        coords_t = torch.tensor(
            np.asarray(coords, dtype=float).reshape(-1, 3),
            dtype=torch.float32, device=device,
        )
        out = predict_fn(coords_t, atomic_nums, do_hessian=True, require_grad=False)
        hess = out["hessian"]
        if isinstance(hess, torch.Tensor):
            hess_t = hess.detach()
        else:
            hess_t = torch.tensor(hess)
        atomsymbols = atomic_nums_to_symbols(atomic_nums)
        evals_vib, _, _ = vib_eig(hess_t, coords_t, atomsymbols, purify=False)
        n_neg = int((evals_vib < 0).sum().cpu().item())
        min_eig = float(evals_vib[0].cpu().item())
        return n_neg, min_eig
    except Exception:
        return None, None


def score_endpoints(
    forward_coords: Optional[np.ndarray],
    reverse_coords: Optional[np.ndarray],
    atomic_nums: torch.Tensor,
    reactant_coords: Optional[torch.Tensor] = None,
    product_coords: Optional[torch.Tensor] = None,
    rmsd_threshold: float = 0.3,
    error: Optional[str] = None,
    predict_fn=None,
) -> IRCResult:
    """Score IRC endpoints against reference reactant/product geometries.

    Applies two direction-agnostic tests in parallel:
    - Kabsch+Hungarian RMSD match below `rmsd_threshold`.
    - Element-labeled bond-graph isomorphism.

    If `predict_fn` is provided, also computes the Eckart-projected
    vibrational eigenvalue spectrum at each endpoint — specifically
    `n_neg_vib` (count of negative eigenvalues; 0 = true minimum) and
    `min_vib_eig` (signed smallest eigenvalue). This distinguishes
    "unintended because we landed at a different minimum" from
    "unintended because IRC never reached a minimum".

    Produces the full `IRCResult` used by the Parquet writer. Shared by
    every IRC integrator in the codebase.
    """
    reactant_np = _to_numpy_coords(reactant_coords)
    product_np = _to_numpy_coords(product_coords)

    fr_rmsd = _coords_rmsd(forward_coords, reactant_np, atomic_nums) if (forward_coords is not None and reactant_np is not None) else None
    rr_rmsd = _coords_rmsd(reverse_coords, reactant_np, atomic_nums) if (reverse_coords is not None and reactant_np is not None) else None
    fp_rmsd = _coords_rmsd(forward_coords, product_np, atomic_nums) if (forward_coords is not None and product_np is not None) else None
    rp_rmsd = _coords_rmsd(reverse_coords, product_np, atomic_nums) if (reverse_coords is not None and product_np is not None) else None

    rmsd_to_reactant = _min_optional(fr_rmsd, rr_rmsd)
    rmsd_to_product = _min_optional(fp_rmsd, rp_rmsd)

    found_reactant = (
        (fr_rmsd is not None and fr_rmsd < rmsd_threshold)
        or (rr_rmsd is not None and rr_rmsd < rmsd_threshold)
    )
    found_product = (
        (fp_rmsd is not None and fp_rmsd < rmsd_threshold)
        or (rp_rmsd is not None and rp_rmsd < rmsd_threshold)
    )

    intended = found_reactant and found_product
    half_intended = (found_reactant or found_product) and not intended

    forward_graph_matches_reactant = False
    forward_graph_matches_product = False
    reverse_graph_matches_reactant = False
    reverse_graph_matches_product = False
    topology_error = None

    try:
        reactant_graph = (
            coords_to_bond_graph(reactant_np, atomic_nums) if reactant_np is not None else None
        )
        product_graph = (
            coords_to_bond_graph(product_np, atomic_nums) if product_np is not None else None
        )
        forward_graph = (
            coords_to_bond_graph(forward_coords, atomic_nums) if forward_coords is not None else None
        )
        reverse_graph = (
            coords_to_bond_graph(reverse_coords, atomic_nums) if reverse_coords is not None else None
        )

        forward_graph_matches_reactant = bond_graphs_match(forward_graph, reactant_graph)
        forward_graph_matches_product = bond_graphs_match(forward_graph, product_graph)
        reverse_graph_matches_reactant = bond_graphs_match(reverse_graph, reactant_graph)
        reverse_graph_matches_product = bond_graphs_match(reverse_graph, product_graph)
    except Exception as exc:
        topology_error = str(exc)

    topology_intended = (
        (forward_graph_matches_reactant and reverse_graph_matches_product)
        or (forward_graph_matches_product and reverse_graph_matches_reactant)
    )
    topology_half_intended = (
        (forward_graph_matches_reactant
         or forward_graph_matches_product
         or reverse_graph_matches_reactant
         or reverse_graph_matches_product)
        and not topology_intended
    )

    forward_n_neg = None
    reverse_n_neg = None
    forward_min_eig = None
    reverse_min_eig = None
    if predict_fn is not None:
        if forward_coords is not None:
            forward_n_neg, forward_min_eig = _endpoint_spectral(
                forward_coords, atomic_nums, predict_fn,
            )
        if reverse_coords is not None:
            reverse_n_neg, reverse_min_eig = _endpoint_spectral(
                reverse_coords, atomic_nums, predict_fn,
            )

    return IRCResult(
        intended=intended,
        half_intended=half_intended,
        topology_intended=topology_intended,
        topology_half_intended=topology_half_intended,
        forward_coords=forward_coords,
        reverse_coords=reverse_coords,
        rmsd_to_reactant=rmsd_to_reactant,
        rmsd_to_product=rmsd_to_product,
        forward_rmsd_to_reactant=fr_rmsd,
        forward_rmsd_to_product=fp_rmsd,
        reverse_rmsd_to_reactant=rr_rmsd,
        reverse_rmsd_to_product=rp_rmsd,
        forward_graph_matches_reactant=forward_graph_matches_reactant,
        forward_graph_matches_product=forward_graph_matches_product,
        reverse_graph_matches_reactant=reverse_graph_matches_reactant,
        reverse_graph_matches_product=reverse_graph_matches_product,
        forward_n_neg_vib=forward_n_neg,
        reverse_n_neg_vib=reverse_n_neg,
        forward_min_vib_eig=forward_min_eig,
        reverse_min_vib_eig=reverse_min_eig,
        error=error,
        topology_error=topology_error,
    )


def run_irc_validation(
    ts_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    predict_fn,
    reactant_coords: Optional[torch.Tensor] = None,
    product_coords: Optional[torch.Tensor] = None,
    rmsd_threshold: float = 0.3,
    max_steps: int = 100,
) -> IRCResult:
    """Baseline: vanilla Sella IRC with BFGS-updated Hessian, Cartesian coords.

    Identical wiring to what the codebase has always used. Kept as the
    control condition for comparison against `irc_sella_hip.run_irc_sella_hip`
    and `irc_rigorous.run_irc_rigorous`.
    """
    try:
        from sella import IRC
    except ImportError:
        return score_endpoints(
            forward_coords=None,
            reverse_coords=None,
            atomic_nums=atomic_nums,
            reactant_coords=reactant_coords,
            product_coords=product_coords,
            rmsd_threshold=rmsd_threshold,
            error="Sella not installed",
        )

    from gadplus.calculator.ase_adapter import HipASECalculator

    coords_np = ts_coords.detach().cpu().numpy().reshape(-1, 3)
    nums = atomic_nums.detach().cpu().tolist()
    symbols = [Z_TO_SYMBOL.get(int(z), "X") for z in nums]

    atoms = Atoms(symbols=symbols, positions=coords_np)
    atoms.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=atomic_nums)

    optimizer_kwargs = {"dx": 0.1, "eta": 1e-4, "gamma": 0.4}

    endpoints = {}
    for direction in ["forward", "reverse"]:
        try:
            atoms_copy = atoms.copy()
            atoms_copy.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=atomic_nums)
            irc = IRC(atoms=atoms_copy, **optimizer_kwargs)
            irc.run(fmax=0.01, steps=max_steps, direction=direction)
            endpoints[direction] = atoms_copy.positions.copy()
        except Exception:
            endpoints[direction] = None

    return score_endpoints(
        forward_coords=endpoints.get("forward"),
        reverse_coords=endpoints.get("reverse"),
        atomic_nums=atomic_nums,
        reactant_coords=reactant_coords,
        product_coords=product_coords,
        rmsd_threshold=rmsd_threshold,
        predict_fn=predict_fn,
    )
