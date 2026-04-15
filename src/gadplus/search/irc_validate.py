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
    forward_graph_matches_reactant: bool
    forward_graph_matches_product: bool
    reverse_graph_matches_reactant: bool
    reverse_graph_matches_product: bool
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


def _coords_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((coords_a - coords_b) ** 2)))


def _min_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
    vals = [x for x in (a, b) if x is not None]
    return min(vals) if vals else None


def run_irc_validation(
    ts_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    predict_fn,
    reactant_coords: Optional[torch.Tensor] = None,
    product_coords: Optional[torch.Tensor] = None,
    rmsd_threshold: float = 0.3,
    max_steps: int = 100,
) -> IRCResult:
    """Run IRC forward and backward from a converged TS.

    Args:
        ts_coords: (N, 3) TS coordinates.
        atomic_nums: (N,) atomic numbers.
        predict_fn: PredictFn for energy/forces.
        reactant_coords: (N, 3) reference reactant geometry.
        product_coords: (N, 3) reference product geometry.
        rmsd_threshold: RMSD threshold for matching endpoints.
        max_steps: Maximum IRC steps per direction.

    Returns:
        IRCResult with validation status and final geometries.
    """
    try:
        from sella import IRC
    except ImportError:
        return IRCResult(
            intended=False, half_intended=False,
            topology_intended=False, topology_half_intended=False,
            forward_coords=None, reverse_coords=None,
            rmsd_to_reactant=None, rmsd_to_product=None,
            forward_graph_matches_reactant=False,
            forward_graph_matches_product=False,
            reverse_graph_matches_reactant=False,
            reverse_graph_matches_product=False,
            error="Sella not installed",
        )

    from gadplus.calculator.ase_adapter import HipASECalculator

    # Build ASE Atoms from TS geometry
    coords_np = ts_coords.detach().cpu().numpy().reshape(-1, 3)
    nums = atomic_nums.detach().cpu().tolist()
    symbols = [Z_TO_SYMBOL.get(int(z), "X") for z in nums]

    atoms = Atoms(symbols=symbols, positions=coords_np)
    atoms.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=atomic_nums)

    optimizer_kwargs = {"dx": 0.1, "eta": 1e-4, "gamma": 0.4}

    # Run IRC in both directions
    endpoints = {}
    for direction in ["forward", "reverse"]:
        try:
            atoms_copy = atoms.copy()
            atoms_copy.calc = HipASECalculator(predict_fn=predict_fn, atomic_nums=atomic_nums)
            irc = IRC(atoms=atoms_copy, **optimizer_kwargs)
            irc.run(fmax=0.01, steps=max_steps, direction=direction)
            endpoints[direction] = atoms_copy.positions.copy()
        except Exception as e:
            endpoints[direction] = None

    forward_coords = endpoints.get("forward")
    reverse_coords = endpoints.get("reverse")

    # Compare endpoints to known reactant/product
    reactant_np = _to_numpy_coords(reactant_coords)
    product_np = _to_numpy_coords(product_coords)

    fr_rmsd = _coords_rmsd(forward_coords, reactant_np) if (forward_coords is not None and reactant_np is not None) else None
    rr_rmsd = _coords_rmsd(reverse_coords, reactant_np) if (reverse_coords is not None and reactant_np is not None) else None
    fp_rmsd = _coords_rmsd(forward_coords, product_np) if (forward_coords is not None and product_np is not None) else None
    rp_rmsd = _coords_rmsd(reverse_coords, product_np) if (reverse_coords is not None and product_np is not None) else None

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

    # Topology-based matching (direction-agnostic: forward/reverse can swap)
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

    return IRCResult(
        intended=intended,
        half_intended=half_intended,
        topology_intended=topology_intended,
        topology_half_intended=topology_half_intended,
        forward_coords=forward_coords,
        reverse_coords=reverse_coords,
        rmsd_to_reactant=rmsd_to_reactant,
        rmsd_to_product=rmsd_to_product,
        forward_graph_matches_reactant=forward_graph_matches_reactant,
        forward_graph_matches_product=forward_graph_matches_product,
        reverse_graph_matches_reactant=reverse_graph_matches_reactant,
        reverse_graph_matches_product=reverse_graph_matches_product,
        topology_error=topology_error,
    )
