"""OpenRouter chat-completions client with retry and JSON extraction."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from utils import extract_first_json_object, save_json


@dataclass
class OpenRouterClient:
    api_key: str
    base_url: str
    timeout_seconds: int = 120
    max_retries: int = 3
    backoff_seconds: float = 2.0

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        raw_save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=body,
                    timeout=self.timeout_seconds,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {resp.status_code} from OpenRouter: {resp.text[:2000]}"
                    )
                payload = resp.json()
                text = self._extract_message_text(payload)

                if raw_save_path is not None:
                    raw_save_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_save_path.write_text(text)
                    save_json(raw_save_path.with_suffix(".meta.json"), payload)

                return extract_first_json_object(text)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)
                else:
                    break

        raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")

    @staticmethod
    def _extract_message_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            raise ValueError("No choices in OpenRouter response.")
        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks)
        return json.dumps(content)
