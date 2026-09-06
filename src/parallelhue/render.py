"""Exact speculative-provenance reconciliation and safe terminal rendering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import re
from threading import Lock
from typing import Generic, Iterable, TypeVar

from .protocol import (
    ProvenanceFrame,
    TextFrame,
    TokenRecord,
    TokenRole,
    TraceSpan,
    parse_request_id,
)

PALETTE = (46, 196, 27, 226)

# CSI/OSC are the useful terminal escape families; the final ESC fallback also
# prevents malformed or unterminated sequences from reaching a terminal.
_ANSI_RE = re.compile(
    r"(?:\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_])",
    re.DOTALL,
)
_BIDI = {
    0x061C,
    0x200E,
    0x200F,
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
}


def sanitize_terminal(text: str) -> str:
    """Remove terminal escapes, BiDi overrides, and unsafe control characters."""
    if type(text) is not str:
        raise TypeError("text must be a string")
    text = _ANSI_RE.sub("", text)
    # Newline, carriage return, and tab are retained as ordinary model
    # formatting. All other C0/C1 controls are terminal control injection.
    return "".join(
        char
        for char in text
        if (char in "\n\r\t" or (0x20 <= ord(char) != 0x7F and ord(char) < 0x7F) or ord(char) >= 0xA0)
        and ord(char) not in _BIDI
    )


def palette_color(step_id: int) -> int:
    """Return the deterministic four-color palette entry for a step id."""
    if type(step_id) is not int or step_id < 0:
        raise ValueError("step_id must be a non-negative integer")
    return PALETTE[step_id % len(PALETTE)]

class StepPalette:
    """Assign palette colors to verified steps in first-observed order."""

    __slots__ = ("_colors", "_next")

    def __init__(self) -> None:
        self._colors: dict[int, int] = {}
        self._next = 0

    def color_for(self, step_id: int) -> int:
        if type(step_id) is not int or step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        if step_id not in self._colors:
            self._colors[step_id] = PALETTE[self._next % len(PALETTE)]
            self._next += 1
        return self._colors[step_id]



def colorize(text: str, color: int | None = None, *, step_id: int | None = None) -> str:
    """Sanitize and wrap text in an xterm-256 foreground color.

    When ``NO_COLOR`` is set to any non-empty value, return sanitized plain text
    with no ANSI color wrapper (https://no-color.org/).
    """
    safe = sanitize_terminal(text)
    if os.environ.get("NO_COLOR", ""):
        return safe
    if step_id is not None:
        selected = palette_color(step_id)
    elif color is None:
        selected = PALETTE[0]
    elif type(color) is int and color in PALETTE:
        selected = color
    elif type(color) is int and color >= 0:
        selected = palette_color(color)
    else:
        raise ValueError("color must be a palette color or non-negative index")
    return f"\x1b[38;5;{selected}m{safe}\x1b[0m"


@dataclass(frozen=True, slots=True)
class RenderSegment:
    """One text span with role metadata and verified-step attribution."""

    text: str
    token_offset_start: int
    token_offset_end: int
    role: TokenRole | None
    ambiguous: bool = False
    step_id: int | None = None

@dataclass(frozen=True, slots=True)
class ReconciledEvent:
    """One joined event with provenance and verified-step render segments."""

    request_id: str
    source_sequence: int
    provenance_sequence: int
    token_offset: int
    step_id: int
    choice_index: int
    tokens: tuple[TokenRecord, ...]
    text: str
    segments: tuple[RenderSegment, ...]
    finished: bool

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token_id for token in self.tokens)

    @property
    def roles(self) -> tuple[TokenRole, ...]:
        return tuple(token.role for token in self.tokens)


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Joined events which exactly account for one coalesced SSE chunk."""

    request_id: str
    events: tuple[ReconciledEvent, ...]
    text: str
    token_ids: tuple[int, ...]

    @property
    def step_ids(self) -> tuple[int, ...]:
        return tuple(event.step_id for event in self.events)

    @property
    def tokens(self) -> tuple[TokenRecord, ...]:
        return tuple(token for event in self.events for token in event.tokens)

    @property
    def roles(self) -> tuple[TokenRole, ...]:
        return tuple(token.role for token in self.tokens)


class _RequestFrames:
    def __init__(self) -> None:
        self.tokens: dict[int, TokenRecord] = {}
        self.steps: dict[int, int] = {}
        self.text: deque[TextFrame] = deque()
        self.sequence = {"provenance": 0, "text": 0}
        self.offset = {"provenance": 0, "text": 0}
        self.terminal: dict[str, int] = {}
        self.render_offset = 0
        self.history_start = 0


class StepReconciler:
    """Join independent sources by absolute token identity, then verify SSE."""

    def __init__(self, max_events_per_request: int = 256, max_requests: int = 1024) -> None:
        if type(max_events_per_request) is not int or max_events_per_request <= 0:
            raise ValueError("max_events_per_request must be positive")
        if type(max_requests) is not int or max_requests <= 0:
            raise ValueError("max_requests must be positive")
        self.max_events_per_request = max_events_per_request
        self.max_requests = max_requests
        self._requests: dict[str, _RequestFrames] = {}
        self._failed: set[str] = set()
        self._complete: set[str] = set()
        self._needs_output: set[str] = set()

    @property
    def failed_requests(self) -> frozenset[str]:
        return frozenset(self._failed)

    def failed(self, request_id: str) -> bool:
        return request_id in self._failed

    def completed(self, request_id: str) -> bool:
        return request_id in self._complete

    def waiting_for_output(self, request_id: str) -> bool:
        """Whether verified telemetry needs more SSE bytes or token IDs."""
        return request_id in self._needs_output

    def push(self, frame: ProvenanceFrame | TextFrame) -> bool:
        if not isinstance(frame, (ProvenanceFrame, TextFrame)):
            raise TypeError("expected a provenance or text frame")
        request_id = frame.request_id
        if request_id in self._failed or request_id in self._complete:
            return False
        state = self._requests.get(request_id)
        if state is None:
            if len(self._requests) >= self.max_requests:
                self._fail(request_id)
                return False
            state = self._requests[request_id] = _RequestFrames()
        kind = frame.frame_type
        if (
            kind in state.terminal
            or frame.source_sequence != state.sequence[kind]
            or frame.token_offset != state.offset[kind]
        ):
            self._fail(request_id)
            return False
        state.sequence[kind] += 1
        state.offset[kind] += len(frame.token_ids)
        if frame.finished:
            state.terminal[kind] = state.offset[kind]
        if isinstance(frame, ProvenanceFrame):
            for offset, token_id, role in zip(
                range(frame.token_offset, state.offset[kind]), frame.token_ids, frame.roles,
            ):
                state.tokens[offset] = TokenRecord(token_id, "", role)
        else:
            for offset in range(frame.token_offset, state.offset[kind]):
                state.steps[offset] = frame.step_id
            state.text.append(frame)
        if (
            len(state.text) > self.max_events_per_request
            or len(state.tokens) > self.max_events_per_request * 256
            or len(state.steps) > self.max_events_per_request * 256
            or len(state.terminal) == 2 and state.terminal["text"] != state.terminal["provenance"]
        ):
            self._fail(request_id)
            return False
        return True

    def _join(self, state: _RequestFrames, frame: TextFrame) -> ReconciledEvent | None:
        end = frame.token_offset + len(frame.token_ids)
        if state.offset["provenance"] < end:
            return None
        if frame.finished and "provenance" not in state.terminal:
            return None
        records = tuple(state.tokens[index] for index in range(frame.token_offset, end))
        if tuple(record.token_id for record in records) != frame.token_ids:
            raise ValueError("token identity mismatch")
        segments = []
        encoded = frame.text.encode("utf-8")
        token_texts = [""] * len(records)
        for span in frame.trace:
            if span.token_offset_end > end:
                raise ValueError("trace refers to future tokens")
            roles = tuple(
                state.tokens[index].role
                for index in range(span.token_offset_start, span.token_offset_end)
            )
            step_ids = tuple(
                state.steps[index]
                for index in range(span.token_offset_start, span.token_offset_end)
            )
            role = roles[0] if all(item == roles[0] for item in roles) else None
            span_step_id = (
                step_ids[0] if all(item == step_ids[0] for item in step_ids) else None
            )
            text = encoded[span.byte_start:span.byte_end].decode("utf-8")
            segments.append(RenderSegment(
                text, span.token_offset_start, span.token_offset_end, role,
                span_step_id is None, span_step_id,
            ))
            if span.token_offset_end == span.token_offset_start + 1 and span.token_offset_start >= frame.token_offset:
                token_texts[span.token_offset_start - frame.token_offset] += text
        tokens = tuple(
            TokenRecord(record.token_id, text, record.role)
            for record, text in zip(records, token_texts)
        )
        return ReconciledEvent(
            frame.request_id, frame.source_sequence, state.sequence["provenance"] - 1,
            frame.token_offset, frame.step_id, frame.choice_index, tokens,
            frame.text, tuple(segments), frame.finished,
        )

    def reconcile_chunk(
        self, request_id: str, text: str, token_ids: Iterable[int] | None = None,
    ) -> Reconciliation | None:
        if type(request_id) is not str or type(text) is not str:
            raise TypeError("request_id and text must be strings")
        if request_id in self._failed or request_id in self._complete:
            return None
        self._needs_output.discard(request_id)
        if token_ids is None:
            self._fail(request_id)
            return None
        chunk_ids = tuple(token_ids)
        if any(type(token) is not int or token < 0 for token in chunk_ids):
            self._fail(request_id)
            return None
        state = self._requests.get(request_id)
        if state is None:
            return None
        ids: list[int] = []
        parts: list[str] = []
        events: list[ReconciledEvent] = []
        keep_from = state.history_start
        for frame in state.text:
            try:
                event = self._join(state, frame)
            except (KeyError, ValueError, UnicodeError):
                self._fail(request_id)
                return None
            if event is None:
                return None
            ids.extend(frame.token_ids)
            parts.append(frame.text)
            events.append(event)
            raw = "".join(parts)
            shared = min(len(ids), len(chunk_ids))
            if tuple(ids[:shared]) != chunk_ids[:shared] or not (text.startswith(raw) or raw.startswith(text)):
                self._fail(request_id)
                return None
            if len(ids) > len(chunk_ids) or len(raw) > len(text):
                self._needs_output.add(request_id)
                return None
            if frame.trace:
                keep_from = max(keep_from, frame.trace[-1].token_offset_start)
            if tuple(ids) == chunk_ids and raw == text:
                if frame.finished and (
                    len(state.text) != len(events)
                    or state.terminal.get("provenance") != frame.token_offset + len(frame.token_ids)
                ):
                    self._fail(request_id)
                    return None
                for _ in events:
                    state.text.popleft()
                state.render_offset = frame.token_offset + len(frame.token_ids)
                while state.history_start < min(keep_from, state.render_offset):
                    state.tokens.pop(state.history_start, None)
                    state.steps.pop(state.history_start, None)
                    state.history_start += 1
                if frame.finished:
                    self._complete.add(request_id)
                    self._requests.pop(request_id, None)
                return Reconciliation(request_id, tuple(events), text, chunk_ids)
            if frame.finished:
                self._fail(request_id)
                return None
        return None

    def _fail(self, request_id: str) -> None:
        self._failed.add(request_id)
        self._requests.pop(request_id, None)
        self._needs_output.discard(request_id)


T = TypeVar("T")


class RenderQueue(Generic[T]):
    """Small thread-safe nonblocking bounded queue with fail-closed overflow."""

    def __init__(self, maxsize: int = 256) -> None:
        if type(maxsize) is not int or maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._items: deque[T] = deque()
        self._lock = Lock()
        self._closed = False
        self._overflowed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    @property
    def failed(self) -> bool:
        return self._closed or self._overflowed

    def put_nowait(self, item: T) -> bool:
        with self._lock:
            if self._closed or self._overflowed or len(self._items) >= self.maxsize:
                self._overflowed = self._overflowed or not self._closed
                return False
            self._items.append(item)
            return True

    def get_nowait(self) -> T | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._items.clear()


BoundedRenderQueue = RenderQueue
NonBlockingRenderQueue = RenderQueue

__all__ = [
    "PALETTE",
    "StepPalette",
    "RenderSegment",
    "ReconciledEvent",
    "Reconciliation",
    "StepReconciler",
    "RenderQueue",
    "BoundedRenderQueue",
    "NonBlockingRenderQueue",
    "sanitize_terminal",
    "palette_color",
    "colorize",
]
