"""Shared matplotlib styling for cross-family benchmark and recovery figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ("Qwen2.5", "Llama 3.x", "Gemma3")
COLORS = {"Qwen2.5": "#4c72b0", "Llama 3.x": "#dd8452", "Gemma3": "#8172b3"}
MARKERS = {"Qwen2.5": "o", "Llama 3.x": "s", "Gemma3": "D"}
FAMILY_CMAPS = {"Qwen2.5": "Blues", "Llama 3.x": "Oranges", "Gemma3": "Purples"}
FAMILY_LINE_STYLES = {"Qwen2.5": "-", "Llama 3.x": "--", "Gemma3": "-."}
CURRICULA = [
    (True, "-", True, 1.0, "Biased curriculum"),
    (False, "--", False, 0.55, "Unbiased curriculum"),
]
CHANCE_COLOR = "#888888"
REFERENCE_GREEN = "#55a868"


def normalize_family(name: str) -> str:
    if name in COLORS:
        return name
    lower = name.lower()
    if "qwen" in lower:
        return "Qwen2.5"
    if "llama" in lower:
        return "Llama 3.x"
    if "gemma" in lower:
        return "Gemma3"
    return name


def family_label(model_name: str) -> str:
    if model_name.startswith("qwen2.5-"):
        return "Qwen2.5"
    if model_name.startswith("llama"):
        return "Llama 3.x"
    if model_name.startswith("gemma"):
        return "Gemma3"
    return model_name.split("-")[0]


def display_size(size_b: float) -> str:
    return f"{size_b:g}B"


def family_model_colors(fam: str, n: int) -> list:
    """Return n shades of the family base color (light → dark)."""
    fam = normalize_family(fam)
    if n <= 0:
        return []
    if n == 1:
        return [COLORS[fam]]
    cmap = plt.get_cmap(FAMILY_CMAPS[fam])
    return [cmap(x) for x in np.linspace(0.40, 0.90, n)]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axes(ax) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
