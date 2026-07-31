#!/usr/bin/env python3
"""A self-contained visual introduction to the GADplus gate evolution.

Run:

    python examples/gad_evolution_toy.py --output gad_evolution_toy.png

The model is the analytic four-well potential

    E(x, y) = (x^2 - 1)^2 + (y^2 - 1)^2.

It has four minima, four index-one edge saddles, and an index-two centre.
The plot compares ordinary GAD, the lambda_2 gate, the competitive gate, and
the final competitive-subspace reflection using exactly the same local data:
coordinates, gradient, and Hessian.  The fifth panel is a scalar diagnostic
showing why relative soft-subspace weights avoid a diluted reflection near a
nearly degenerate soft spectrum.

This is an explanatory toy, not a molecular benchmark.  It uses unit masses
and two Cartesian degrees of freedom so all formulae remain readable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Policy:
    name: str
    color: str
    gate: str
    subspace: bool = False


POLICIES = (
    Policy("ordinary GAD", "#7F7F7F", "gad"),
    Policy(r"smooth $\lambda_2$ gate", "#CC79A7", "lambda2"),
    Policy("competitive gate", "#0072B2", "competitive"),
    Policy("final competitive-subspace", "#009E73", "competitive", True),
)


def energy(q: np.ndarray) -> float:
    x, y = q
    return float((x * x - 1.0) ** 2 + (y * y - 1.0) ** 2)


def gradient(q: np.ndarray) -> np.ndarray:
    x, y = q
    return np.array((4.0 * x * (x * x - 1.0), 4.0 * y * (y * y - 1.0)))


def hessian(q: np.ndarray) -> np.ndarray:
    x, y = q
    return np.diag((12.0 * x * x - 4.0, 12.0 * y * y - 4.0))


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def local_step(q: np.ndarray, policy: Policy, *, temperature: float = 0.18, radius: float = 0.085) -> np.ndarray:
    """Return the closed-form pointwise step for one policy.

    This is the identity-mass, no-rigid-mode specialization of the documented
    intrinsic update.  The soft-min density is retained even in two dimensions
    so the final relative-subspace correction is visible.
    """

    eigenvalues, vectors = np.linalg.eigh(hessian(q))
    c = vectors.T @ gradient(q)
    scale = max(float(np.sqrt(np.mean(eigenvalues**2))), 1.0e-12)
    z = eigenvalues / (temperature * scale)
    p = np.exp(-z - np.max(-z))
    p /= p.sum()
    negative = sigmoid(-z)
    w2 = float(sigmoid(z[1]))
    soft = float(np.sum(p * c**2))
    extra = float(np.sum(negative * (1.0 - p) ** 2 * c**2))
    fraction = soft / (soft + extra) if soft + extra > 0.0 else 0.0
    if policy.gate == "gad":
        w = 1.0
    elif policy.gate == "lambda2":
        w = w2
    else:
        w = w2 + (1.0 - w2) * fraction
    reflection = p / p.max() if policy.subspace else p
    if policy.gate == "gad":
        reflection = np.array((1.0, 0.0))
    b = (1.0 - 2.0 * w * reflection) * c
    mu = np.linalg.norm(b) / radius if np.linalg.norm(b) > 0.0 else 1.0
    a = -b / np.sqrt(eigenvalues**2 + mu**2)
    return vectors @ a


def trajectory(start: np.ndarray, policy: Policy, *, steps: int = 260) -> np.ndarray:
    points = [np.asarray(start, dtype=float)]
    for _ in range(steps):
        delta = local_step(points[-1], policy)
        points.append(points[-1] + delta)
    return np.asarray(points)


def field_diagnostics(lam1: float, lam2: float, c: np.ndarray, temperature: float = 0.18) -> tuple[float, float, float, float]:
    """Return p1, competitive gate, rank-one and subspace multipliers."""

    values = np.array((lam1, lam2), dtype=float)
    scale = max(float(np.sqrt(np.mean(values**2))), 1.0e-12)
    z = values / (temperature * scale)
    p = np.exp(-z - np.max(-z)); p /= p.sum()
    w2 = float(sigmoid(z[1]))
    soft = float(np.sum(p * c**2))
    extra = float(np.sum(sigmoid(-z) * (1.0 - p) ** 2 * c**2))
    chi = soft / (soft + extra) if soft + extra else 0.0
    w = w2 + (1.0 - w2) * chi
    return float(p[0]), w, float(1.0 - 2.0 * w * p[0]), float(1.0 - 2.0 * w)


def make_figure(output: str | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    grid = np.linspace(-1.55, 1.55, 360)
    xx, yy = np.meshgrid(grid, grid)
    zz = (xx**2 - 1.0) ** 2 + (yy**2 - 1.0) ** 2
    starts = np.array(((0.34, 0.28), (-0.34, 0.28), (0.30, -0.37), (-0.28, -0.35)))
    figure, axes = plt.subplots(1, 5, figsize=(19, 3.8), constrained_layout=True)
    for axis, policy in zip(axes[:4], POLICIES):
        axis.contour(xx, yy, zz, levels=np.linspace(0.08, 5.0, 13), colors="#BDBDBD", linewidths=0.7)
        axis.scatter((0, 0, 1, -1, 1, -1, 0, 0, 0), (0, 1, 0, 0, 0, 0, -1, 0, 0), s=10, color="#222222", zorder=2)
        for start in starts:
            path = trajectory(start, policy)
            axis.plot(path[:, 0], path[:, 1], color=policy.color, alpha=0.88, lw=1.6)
            axis.scatter(*start, color=policy.color, s=18, zorder=3)
        axis.set(title=policy.name, xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), aspect="equal", xlabel="x", ylabel="y")
        axis.axhline(0.0, color="#DDDDDD", lw=0.6); axis.axvline(0.0, color="#DDDDDD", lw=0.6)

    axis = axes[4]
    gaps = np.linspace(0.0, 1.2, 200)
    rank_one, subspace, weights = [], [], []
    for gap in gaps:
        p1, _, m_rank, m_sub = field_diagnostics(-1.0, -1.0 + gap, np.array((1.0, 0.05)))
        weights.append(p1); rank_one.append(m_rank); subspace.append(m_sub)
    axis.plot(gaps, weights, color="#CC79A7", label=r"soft density $p_1$")
    axis.plot(gaps, rank_one, color="#0072B2", label=r"rank-one multiplier $1-2wp_1$")
    axis.plot(gaps, subspace, color="#009E73", label=r"subspace multiplier $1-2w$")
    axis.axhline(0.0, color="#222222", lw=0.8, ls="--")
    axis.set(title="Why the final reflection helps", xlabel=r"$\lambda_2-\lambda_1$", ylabel="weight / multiplier", ylim=(-1.05, 1.05))
    axis.legend(fontsize=7, loc="lower right")
    figure.suptitle("From ordinary GAD to the final local competitive-subspace rule", fontsize=14)
    if output:
        figure.savefig(output, dpi=220, bbox_inches="tight")
        print(f"wrote {output}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="gad_evolution_toy.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    make_figure(args.output, args.show)


if __name__ == "__main__":
    main()
