"""Profile for ordinary OpenAI-compatible vLLM deployments."""
from __future__ import annotations

from .base import BackendProfile


class GenericBackendProfile(BackendProfile):
    """Ordinary OpenAI-compatible servers need no extra summary details."""


PROFILE = GenericBackendProfile(
    name="generic",
    generation_counters=(
        "vllm:generation_tokens_total",
        "vllm:generation_tokens",
        "generation_tokens_total",
        "vllm:request_generation_tokens_sum",
        # llama.cpp / llama-server --metrics
        "llamacpp:tokens_predicted_total",
    ),
    accepted_counters=(),
    draft_counters=(),
    drafts_counters=(),
    uses_speculative_decoding=False,
)
