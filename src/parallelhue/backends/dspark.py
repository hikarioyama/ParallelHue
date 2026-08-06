"""Profile for dspark speculative decoding deployments."""
from __future__ import annotations

from .base import BackendProfile


class DSparkBackendProfile(BackendProfile):
    """dspark-specific presentation for speculative decoding metrics."""

    def extra_summary_lines(
        self,
        before: dict[str, float],
        after: dict[str, float],
        makespan: float,
    ) -> list[str]:
        return ["speculative backend: dspark"]


PROFILE = DSparkBackendProfile(
    name="dspark",
    generation_counters=(
        "vllm:generation_tokens_total",
        "vllm:generation_tokens",
        "generation_tokens_total",
        "vllm:request_generation_tokens_sum",
    ),
    accepted_counters=(
        "dspark:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    ),
    draft_counters=(
        "dspark:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
    ),
    drafts_counters=(
        "dspark:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_drafts_total",
    ),
    uses_speculative_decoding=True,
)
