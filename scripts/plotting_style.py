"""Shared plotting style for repository analysis scripts."""
from __future__ import annotations

from collections.abc import Iterable

from matplotlib.colors import to_hex
import seaborn as sns


def apply_plot_style() -> None:
    """Apply the repo-wide Seaborn theme for static matplotlib figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    sns.set_context("talk")


def palette(n_colors: int = 10):
    """Return colors from the currently active Seaborn palette."""
    return sns.color_palette(n_colors=n_colors)


def palette_color(index: int, n_colors: int = 10):
    """Return one color from the currently active Seaborn palette."""
    colors = palette(n_colors=n_colors)
    return colors[index % len(colors)]


def palette_hex(index: int, n_colors: int = 10) -> str:
    """Return one active-palette color as a hex string."""
    return to_hex(palette_color(index, n_colors=n_colors))


def palette_map(names: Iterable[str]):
    """Map names to colors from the currently active Seaborn palette."""
    names = list(names)
    colors = palette(n_colors=max(len(names), 1))
    return dict(zip(names, colors))
