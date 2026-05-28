"""Baseline prompt construction for binary layout augmentation.

Customize these prompts for your domain by editing the strings below
(e.g., describe the semantic meaning of 1/0 in your layouts, add domain
constraints, supply a few-shot example, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List

from utils import compact_matrix_json


def build_system_prompt() -> str:
    """Baseline system prompt. Replace domain language for your use case."""
    return (
        "You are a binary-layout augmentation assistant. "
        "You operate ONLY on 2D binary matrix data, not images.\n\n"
        "Hard constraints:\n"
        "- Matrices are binary: values must be 0 or 1 only.\n"
        "- Output matrices must have the same shape as the input examples.\n"
        "- Produce diverse but plausible variants of the input layouts.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- No prose.\n"
        "- No markdown.\n"
        "- No code fences.\n"
        "- Return a JSON object with key 'variants'.\n"
        "- Prefer sparse edit operations relative to a chosen source example.\n"
        "- Do NOT return a full grid unless absolutely necessary."
    )


def build_user_prompt(
    *,
    example_layouts_bits: List[Dict[str, Any]],
    n_variants: int,
) -> str:
    """Baseline user prompt. Add domain-specific constraints/masks as needed."""
    payload: Dict[str, Any] = {
        "task": "generate_binary_layout_variants",
        "n_variants": n_variants,
        "preferred_output_format": {
            "variants": [
                {
                    "id": "variant_1",
                    "source_layout_id": example_layouts_bits[0]["id"],
                    "edits": [
                        {"r0": 10, "r1": 20, "c0": 30, "c1": 45, "value": 1},
                        {"points": [[5, 5], [5, 6]], "value": 0},
                    ],
                }
            ]
        },
        "indexing_rules": {
            "indices": "0-based",
            "r1_c1": "exclusive",
            "bounds": "must stay within matrix shape",
        },
        "constraints": [
            "binary values only",
            "same grid shape as input",
            "diverse routing relative to all examples",
        ],
    }

    return (
        "Return JSON only.\n"
        "No text explanation.\n"
        "No markdown.\n"
        "No code fences.\n\n"
        "These are example 2D binary layouts. Generate NEW variants.\n"
        "Binary semantics: 1 = foreground, 0 = background.\n\n"
        "IMPORTANT: Return sparse edits relative to one source example layout.\n"
        "Do not return full-grid matrices unless sparse edits are impossible.\n"
        "Keep edits concise (typically <= 40 edit objects per variant).\n\n"
        f"n_variants: {n_variants}\n"
        f"example_layouts_bit_rows: {compact_matrix_json(example_layouts_bits)}\n\n"
        "Preferred output format:\n"
        "{\"variants\":[{\"id\":\"variant_1\",\"source_layout_id\":\"<id>\","
        "\"edits\":[{\"r0\":10,\"r1\":20,\"c0\":30,\"c1\":45,\"value\":1}]}]}\n"
        "Alternative fallback format (only if needed):\n"
        "{\"variants\":[{\"id\":\"variant_1\",\"grid\":[[0,1,0],[1,1,0]]}]}\n\n"
        "Additional JSON context:\n"
        f"{compact_matrix_json(payload)}"
    )
