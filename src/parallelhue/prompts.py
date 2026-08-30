"""Loading per-worker prompts from a JSON prompt file."""
from __future__ import annotations

import json
from pathlib import Path


class PromptFileError(ValueError):
    """Raised when a prompt file is missing, malformed, or empty."""


def load_prompt_file(path: str) -> list[str]:
    """Load a JSON array of non-empty prompt strings from ``path``.

    Every element must be a non-empty string; anything else is rejected
    rather than silently dropped.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptFileError(f"cannot read prompt file {path!r}: {exc.strerror or exc}") from exc
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptFileError(f"prompt file {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(values, list):
        raise PromptFileError(f"prompt file {path!r} must be a JSON array of prompt strings")
    if not values:
        raise PromptFileError(f"prompt file {path!r} must contain at least one prompt string")
    return values
