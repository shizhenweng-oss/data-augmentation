"""Matplotlib helpers for binary layouts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_binary_layout(layout: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(layout, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_side_by_side(
    original: np.ndarray,
    candidate: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Original")
    axes[1].imshow(candidate, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Candidate")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
