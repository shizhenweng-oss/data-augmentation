"""Baseline candidate validation for binary layouts.

Only checks binary values and shape. Add your own domain-specific
constraints (connectivity, no-go regions, port touches, etc.) inside
`validate_candidate` or by composing additional check functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ValidationResult:
    is_valid: bool
    failures: List[str] = field(default_factory=list)
    details: Dict[str, float | int | str] = field(default_factory=dict)


def validate_candidate(
    *,
    candidate: np.ndarray,
    source_shape: Tuple[int, int],
) -> ValidationResult:
    """Baseline checks: binary values + exact shape. Extend as needed."""
    failures: List[str] = []
    details: Dict[str, float | int | str] = {}

    if not np.isin(candidate, [0, 1]).all():
        failures.append("non_binary_values")
    details["binary_ok"] = int(np.isin(candidate, [0, 1]).all())

    if candidate.shape != source_shape:
        failures.append("shape_mismatch")
    details["shape"] = f"{candidate.shape[0]}x{candidate.shape[1]}"
    details["expected_shape"] = f"{source_shape[0]}x{source_shape[1]}"

    return ValidationResult(is_valid=(len(failures) == 0), failures=failures, details=details)
