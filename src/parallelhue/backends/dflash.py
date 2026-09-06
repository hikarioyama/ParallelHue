"""Profile for standard vLLM DFlash speculative-decoding counters."""
from .base import BackendProfile

PROFILE = BackendProfile(
    name="dflash",
    generation_counters=(
        "vllm:generation_tokens_total",
        "vllm:generation_tokens",
        "generation_tokens_total",
        "vllm:request_generation_tokens_sum",
    ),
    accepted_counters=("vllm:spec_decode_num_accepted_tokens_total",),
    draft_counters=("vllm:spec_decode_num_draft_tokens_total",),
    drafts_counters=("vllm:spec_decode_num_drafts_total",),
    uses_speculative_decoding=True,
)
