"""Profile for vLLM MTP speculative decoding."""
from __future__ import annotations

from .base import BackendProfile


class MTPBackendProfile(BackendProfile):
    """MTP-specific presentation for the standard speculative counters."""

    def extra_summary_lines(
        self,
        before: dict[str, float],
        after: dict[str, float],
        makespan: float,
    ) -> list[str]:
        return ["speculative backend: MTP"]


PROFILE = MTPBackendProfile(
    name="mtp",
    generation_counters=(
        "vllm:generation_tokens_total",
        "vllm:generation_tokens",
        "generation_tokens_total",
        "vllm:request_generation_tokens_sum",
    ),
    accepted_counters=("vllm:spec_decode_num_accepted_tokens_total",),
    draft_counters=("vllm:spec_decode_num_draft_tokens_total",),
    drafts_counters=("vllm:spec_decode_num_drafts_total",),
)
