"""Loading/saving and conversion utilities for binary layout matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_npy_binary(path: Path) -> np.ndarray:
    """Load a .npy array and enforce 2D binary uint8 (0/1)."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D array, got shape={arr.shape}")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{path} contains non-binary values.")
    return arr.astype(np.uint8)


def npy_to_nested_list(arr: np.ndarray) -> List[List[int]]:
    return arr.astype(int).tolist()


def nested_list_to_npy(grid: List[List[int]]) -> np.ndarray:
    arr = np.array(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape={arr.shape}")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("Grid contains values other than 0/1.")
    return arr.astype(np.uint8)


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.uint8))


def save_binary_png(path: Path, arr: np.ndarray, title: str | None = None) -> None:
    """Save a binary matrix as a grayscale image (1=black, 0=white)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(arr, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_layouts(layouts_dir: Path) -> Dict[str, np.ndarray]:
    """Load all .npy layouts from a directory, keyed by file stem."""
    out: Dict[str, np.ndarray] = {}
    for p in sorted(layouts_dir.glob("*.npy")):
        if "_viz" in p.stem:
            continue
        out[p.stem] = load_npy_binary(p)
    if not out:
        raise FileNotFoundError(f"No .npy layout files found in {layouts_dir}")
    return out
