"""General helper utilities."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        ensure_dir(path)


def slugify_model_name(model: str) -> str:
    out = model.replace("/", "_")
    out = re.sub(r"[^a-zA-Z0-9_]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out.lower()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compact_matrix_json(matrix: Any) -> str:
    return json.dumps(matrix, separators=(",", ":"))


def matrix_to_bit_rows(arr: np.ndarray) -> List[str]:
    """Convert binary matrix to compact list of bit-rows like '0101...'."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("matrix_to_bit_rows requires binary matrix values.")
    return ["".join("1" if v else "0" for v in row.tolist()) for row in arr.astype(np.uint8)]


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """Extract first balanced JSON object from raw model text."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model response.")

    depth = 0
    end = None
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise ValueError("No balanced JSON object found in model response.")
    return json.loads(text[start:end])


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def generate_mock_variants(layout: np.ndarray, n_variants: int) -> Dict[str, Any]:
    """Local random-flip variants for dry-run testing (no API calls)."""
    variants: List[Dict[str, Any]] = []
    h, w = layout.shape
    for i in range(n_variants):
        cand = layout.copy()
        rng = np.random.default_rng(seed=10_000 + i)
        n_flips = max(25, (h * w) // 250)
        ys = rng.integers(0, h, size=n_flips)
        xs = rng.integers(0, w, size=n_flips)
        for y, x in zip(ys, xs):
            if cand[y, x] == 1:
                cand[y, x] = 1 if rng.random() < 0.7 else 0
            else:
                cand[y, x] = 1 if rng.random() < 0.2 else 0
        variants.append({"id": f"variant_{i+1}", "grid": cand.astype(int).tolist()})
    return {"variants": variants}
