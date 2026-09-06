"""Loading and selecting per-worker prompts."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


class PromptFileError(ValueError):
    """Raised when a prompt file is missing, malformed, or invalid."""


class PromptSelectionError(ValueError):
    """Raised when prompt values cannot satisfy a requested selection."""


def _validate_prompt_values(
    values: object,
    *,
    source: str,
    error_type: type[ValueError],
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise error_type(f"{source} must be a sequence of prompt strings")
    entries = list(values)
    if not entries:
        raise error_type(f"{source} must contain at least one prompt string")
    if any(not isinstance(value, str) or not value for value in entries):
        raise error_type(f"{source} must contain only non-empty prompt strings")
    prompts = [value for value in entries if isinstance(value, str)]
    if len(set(prompts)) != len(prompts):
        raise error_type(f"{source} must contain distinct prompt strings")
    return prompts


def validate_prompt_values(values: Sequence[object], *, source: str = "prompts") -> list[str]:
    """Validate and copy a prompt sequence without changing its text."""
    return _validate_prompt_values(values, source=source, error_type=PromptSelectionError)


def select_prompts(
    values: Sequence[object],
    count: int,
    *,
    worker_index: int | None = None,
    source: str = "prompts",
) -> list[str]:
    """Select the first ``count`` prompts or one bounded worker entry."""
    prompts = validate_prompt_values(values, source=source)
    if count < 1:
        raise PromptSelectionError(f"{source} requires a positive prompt count")
    if len(prompts) < count:
        raise PromptSelectionError(f"{source} has fewer than {count} prompt strings")
    if worker_index is not None:
        if worker_index < 0 or worker_index >= len(prompts):
            raise PromptSelectionError(f"{source} has no prompt for worker index {worker_index}")
        return [prompts[worker_index]]
    return prompts[:count]


def load_prompt_file(path: str) -> list[str]:
    """Load a JSON array of non-empty, distinct prompt strings from ``path``."""
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
    return _validate_prompt_values(
        values,
        source=f"prompt file {path!r}",
        error_type=PromptFileError,
    )
