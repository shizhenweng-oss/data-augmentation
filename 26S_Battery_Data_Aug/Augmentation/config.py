"""Configuration for LLM-based binary layout augmentation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Config:
    """Runtime configuration."""

    # API
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    api_key_env_var: str = "OPENROUTER_API_KEY"
    # NOTE: free-tier model availability on OpenRouter changes frequently.
    # Verify these slugs at https://openrouter.ai/models?max_price=0 before running.
    default_models: List[str] = field(
        default_factory=lambda: [
            "deepseek/deepseek-chat-v3.1:free",
            "google/gemma-3-27b-it:free",
        ]
    )

    # Generation
    temperature: float = 0.2
    max_tokens: int = 20000
    variants_per_layout: int = 4
    request_timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    layouts_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "data" / "layouts"
    )
    outputs_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs"
    )

    # Run behavior
    random_seed: int = 42
    dry_run: bool = False

    def api_key(self) -> str:
        """Read API key from environment. Set OPENROUTER_API_KEY=<your_api_key_here>."""
        key = os.getenv(self.api_key_env_var, "").strip()
        if not key:
            raise RuntimeError(
                f"Missing API key. Set environment variable {self.api_key_env_var} "
                f"(e.g. export {self.api_key_env_var}=<your_api_key_here>)."
            )
        return key


def load_config() -> Config:
    return Config()
