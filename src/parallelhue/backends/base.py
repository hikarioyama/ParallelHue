"""Shared definitions for backend-specific ParallelHue summaries."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendProfile:
    """The counters and presentation details used for one serving backend.

    Counter tuples are ordered by preference.  A profile may leave any of the
    speculative-decoding tuples empty when that backend does not expose it.

    ``uses_speculative_decoding`` gates stream coloring: ParallelHue only
    paints step/chunk colors when the serving path actually uses speculative
    decoding (MTP, DSpark, …). Ordinary decode stays monochrome.
    """

    name: str
    generation_counters: tuple[str, ...]
    accepted_counters: tuple[str, ...]
    draft_counters: tuple[str, ...]
    drafts_counters: tuple[str, ...]
    uses_speculative_decoding: bool = False

    def extra_summary_lines(
        self,
        before: dict[str, float],
        after: dict[str, float],
        makespan: float,
    ) -> list[str]:
        """Return backend-specific lines for the completed-run summary.

        ``before``, ``after``, and ``makespan`` let a backend derive lines
        from its own counters without depending on command-line internals.
        Generic OpenAI-compatible servers have no additional lines.
        """
        return []
