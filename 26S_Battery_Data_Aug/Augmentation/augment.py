"""Main entrypoint for binary layout augmentation via OpenRouter."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import Config, load_config
from data_io import load_layouts, nested_list_to_npy, save_npy
from openrouter_client import OpenRouterClient
from prompting import build_system_prompt, build_user_prompt
from utils import (
    ensure_dirs,
    generate_mock_variants,
    matrix_to_bit_rows,
    save_json,
    set_reproducibility,
    slugify_model_name,
)
from validators import ValidationResult, validate_candidate
from visualize import plot_binary_layout, plot_side_by_side


def _parse_args(cfg: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM binary-layout augmentation.")
    parser.add_argument("--dry-run", action="store_true", help="Use local mocked responses only.")
    parser.add_argument(
        "--variants-per-layout",
        type=int,
        default=cfg.variants_per_layout,
        help="Number of variants requested from each model.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=cfg.default_models,
        help="OpenRouter model names to test.",
    )
    return parser.parse_args()


def _prepare_output_dirs(cfg: Config, model_slug: str, request_id: str) -> Dict[str, Path]:
    d = {
        "valid": cfg.outputs_root / "valid" / model_slug / request_id,
        "invalid": cfg.outputs_root / "invalid" / model_slug / request_id,
        "reports": cfg.outputs_root / "reports" / model_slug / request_id,
        "visualizations": cfg.outputs_root / "visualizations" / model_slug / request_id,
        "raw": cfg.outputs_root / "raw_responses" / model_slug / request_id,
    }
    ensure_dirs(d.values())
    return d


def _save_candidate_artifacts(
    *,
    candidate: np.ndarray,
    layout: np.ndarray,
    result: ValidationResult,
    variant_id: str,
    dirs: Dict[str, Path],
    model_name: str,
    request_id: str,
) -> None:
    target_root = dirs["valid"] if result.is_valid else dirs["invalid"]
    save_npy(target_root / f"{variant_id}.npy", candidate)
    plot_binary_layout(
        candidate,
        target_root / f"{variant_id}.png",
        title=f"{request_id} | {model_name} | {variant_id} | {'VALID' if result.is_valid else 'INVALID'}",
    )
    plot_side_by_side(
        layout,
        candidate,
        target_root / f"{variant_id}_compare.png",
        title=f"{request_id} | {variant_id} | original vs candidate",
    )


def _extract_variants(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError("Model JSON missing 'variants' list.")
    cleaned: List[Dict[str, Any]] = []
    for i, item in enumerate(variants, start=1):
        if not isinstance(item, dict):
            continue
        vid = str(item.get("id", f"variant_{i}"))
        cleaned.append({"id": vid, **{k: item[k] for k in ("grid", "edits", "source_layout_id") if k in item}})
    if not cleaned:
        raise ValueError("No parseable variants in model response.")
    return cleaned


def _apply_sparse_edits(base_layout: np.ndarray, edits: List[Dict[str, Any]]) -> np.ndarray:
    """Apply rect + point edit ops to a base binary matrix."""
    out = base_layout.copy()
    h, w = out.shape
    for op in edits:
        if not isinstance(op, dict):
            continue
        value = int(op.get("value", 1))
        if value not in (0, 1):
            continue

        if all(k in op for k in ("r0", "r1", "c0", "c1")):
            r0 = max(0, min(h, int(op["r0"])))
            r1 = max(0, min(h, int(op["r1"])))
            c0 = max(0, min(w, int(op["c0"])))
            c1 = max(0, min(w, int(op["c1"])))
            if r1 > r0 and c1 > c0:
                out[r0:r1, c0:c1] = value
            continue

        points = op.get("points")
        if isinstance(points, list):
            for p in points:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    continue
                y, x = int(p[0]), int(p[1])
                if 0 <= y < h and 0 <= x < w:
                    out[y, x] = value
    return out.astype(np.uint8)


def _variant_to_candidate(
    *,
    variant: Dict[str, Any],
    examples: Dict[str, np.ndarray],
    default_layout_id: str,
) -> np.ndarray:
    grid = variant.get("grid")
    if grid is not None:
        return nested_list_to_npy(grid)

    source_layout_id = str(variant.get("source_layout_id", default_layout_id))
    base = examples.get(source_layout_id, examples[default_layout_id])
    edits = variant.get("edits", [])
    if not isinstance(edits, list):
        raise ValueError("Variant 'edits' must be a list.")
    return _apply_sparse_edits(base, edits)


def run() -> None:
    cfg = load_config()
    args = _parse_args(cfg)
    set_reproducibility(cfg.random_seed)

    ensure_dirs(
        [
            cfg.outputs_root / "valid",
            cfg.outputs_root / "invalid",
            cfg.outputs_root / "reports",
            cfg.outputs_root / "visualizations",
            cfg.outputs_root / "raw_responses",
        ]
    )

    layouts = load_layouts(cfg.layouts_dir)
    system_prompt = build_system_prompt()

    client = None
    if not args.dry_run:
        client = OpenRouterClient(
            api_key=cfg.api_key(),
            base_url=cfg.openrouter_base_url,
            timeout_seconds=cfg.request_timeout_seconds,
            max_retries=cfg.max_retries,
            backoff_seconds=cfg.retry_backoff_seconds,
        )

    all_reports: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dry_run": bool(args.dry_run),
        "models": {},
    }

    layout_ids = sorted(layouts.keys())
    reference_layout_id = layout_ids[0]
    reference_layout = layouts[reference_layout_id]

    example_layouts_bits = [
        {"id": lid, "rows": matrix_to_bit_rows(layouts[lid])} for lid in layout_ids
    ]

    for model_name in args.models:
        model_slug = slugify_model_name(model_name)
        request_id = "from_examples"
        dirs = _prepare_output_dirs(cfg, model_slug, request_id)
        all_reports["models"][model_name] = {}

        for lid in layout_ids:
            plot_binary_layout(
                layouts[lid],
                dirs["visualizations"] / f"{lid}_example.png",
                title=f"{lid} | example layout",
            )

        user_prompt = build_user_prompt(
            example_layouts_bits=example_layouts_bits,
            n_variants=args.variants_per_layout,
        )

        if args.dry_run:
            response_json = generate_mock_variants(reference_layout, args.variants_per_layout)
            save_json(
                dirs["raw"] / f"{request_id}_{model_slug}_dry_run_response.json",
                response_json,
            )
        else:
            assert client is not None
            response_json = client.chat_completion(
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                raw_save_path=dirs["raw"] / f"{request_id}_{model_slug}_raw.txt",
            )

        model_layout_report: Dict[str, Any] = {
            "request_id": request_id,
            "model": model_name,
            "examples": layout_ids,
            "variants": [],
        }

        try:
            variants = _extract_variants(response_json)
        except Exception as exc:
            model_layout_report["error"] = f"parse_failed: {exc}"
            all_reports["models"][model_name][request_id] = model_layout_report
            save_json(dirs["reports"] / "report.json", model_layout_report)
            continue

        for idx, variant in enumerate(variants, start=1):
            variant_id = str(variant.get("id", f"variant_{idx}"))
            try:
                cand = _variant_to_candidate(
                    variant=variant,
                    examples=layouts,
                    default_layout_id=reference_layout_id,
                )
                validation = validate_candidate(
                    candidate=cand,
                    source_shape=reference_layout.shape,
                )
            except Exception as exc:
                validation = ValidationResult(
                    is_valid=False,
                    failures=["parse_or_conversion_error"],
                    details={"error": str(exc)},
                )
                cand = np.zeros_like(reference_layout, dtype=np.uint8)

            _save_candidate_artifacts(
                candidate=cand,
                layout=reference_layout,
                result=validation,
                variant_id=variant_id,
                dirs=dirs,
                model_name=model_name,
                request_id=request_id,
            )
            model_layout_report["variants"].append(
                {
                    "id": variant_id,
                    "is_valid": validation.is_valid,
                    "failures": validation.failures,
                    "details": validation.details,
                }
            )

        all_reports["models"][model_name][request_id] = model_layout_report
        save_json(dirs["reports"] / "report.json", model_layout_report)

    save_json(cfg.outputs_root / "run_report.json", all_reports)
    print(f"Done. Outputs written to: {cfg.outputs_root}")


if __name__ == "__main__":
    run()
