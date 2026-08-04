"""Exact event reconciliation and safe terminal rendering primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from threading import Lock
from typing import Generic, Iterable, TypeVar

from .protocol import StepEvent, parse_request_id

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


def colorize(text: str, color: int | None = None, *, step_id: int | None = None) -> str:
    """Sanitize and wrap text in an xterm-256 foreground color."""
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
    safe = sanitize_terminal(text)
    return f"\x1b[38;5;{selected}m{safe}\x1b[0m"


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Events which exactly account for one coalesced SSE chunk."""

    request_id: str
    events: tuple[StepEvent, ...]
    text: str
    token_ids: tuple[int, ...]

    @property
    def step_ids(self) -> tuple[int, ...]:
        return tuple(event.step_id for event in self.events)


class StepReconciler:
    """Bounded, deterministic, fail-closed exact event reconciler.

    Events are accepted only in per-request sequence order. A chunk is matched
    against a prefix of buffered complete events; no event is consumed until
    both concatenated token IDs and raw text match exactly.
    """

    def __init__(self, max_events_per_request: int = 256, max_requests: int = 1024) -> None:
        if type(max_events_per_request) is not int or max_events_per_request <= 0:
            raise ValueError("max_events_per_request must be positive")
        if type(max_requests) is not int or max_requests <= 0:
            raise ValueError("max_requests must be positive")
        self.max_events_per_request = max_events_per_request
        self.max_requests = max_requests
        self._buffers: dict[str, deque[StepEvent]] = {}
        self._next_sequence: dict[str, int] = {}
        self._failed: set[str] = set()
        self._complete: set[str] = set()

    @property
    def failed_requests(self) -> frozenset[str]:
        return frozenset(self._failed)

    def failed(self, request_id: str) -> bool:
        return request_id in self._failed

    def completed(self, request_id: str) -> bool:
        """Return true only after a terminal event was matched with no remainder."""
        return request_id in self._complete and not self._buffers.get(request_id)

    def push(self, event: StepEvent) -> bool:
        """Buffer one event, returning false after any fail-closed condition."""
        if not isinstance(event, StepEvent):
            raise TypeError("event must be StepEvent")
        request_id = event.request_id
        if request_id in self._failed or request_id in self._complete:
            return False
        try:
            parse_request_id(request_id)
        except ValueError:
            self._failed.add(request_id)
            return False
        if request_id not in self._buffers:
            if len(self._buffers) >= self.max_requests:
                self._failed.add(request_id)
                return False
            self._buffers[request_id] = deque()
            self._next_sequence[request_id] = 0
        if event.sequence != self._next_sequence[request_id]:
            self._fail(request_id)
            return False
        buffer = self._buffers[request_id]
        if any(buffered.finished for buffered in buffer):
            self._fail(request_id)
            return False
        if len(buffer) >= self.max_events_per_request:
            self._fail(request_id)
            return False
        buffer.append(event)
        self._next_sequence[request_id] += 1
        return True

    add = push
    feed = push

    def reconcile_chunk(
        self,
        request_id: str,
        text: str,
        token_ids: Iterable[int] | None = None,
    ) -> Reconciliation | None:
        """Match one SSE chunk against one or more contiguous buffered events."""
        if type(request_id) is not str or type(text) is not str:
            raise TypeError("request_id and text must be strings")
        if request_id in self._failed or request_id in self._complete:
            return None
        if token_ids is None:
            self._fail(request_id)
            return None
        try:
            chunk_ids = tuple(token_ids)
        except TypeError as exc:
            raise TypeError("token_ids must be iterable") from exc
        if any(type(token) is not int or token < 0 for token in chunk_ids):
            self._fail(request_id)
            return None
        buffer = self._buffers.get(request_id)
        if not buffer:
            return None
        ids: list[int] = []
        raw_parts: list[str] = []
        for count, event in enumerate(buffer, 1):
            ids.extend(event.token_ids)
            raw_parts.append(event.text)
            if tuple(ids) == chunk_ids and "".join(raw_parts) == text:
                matched = tuple(list(buffer)[:count])
                for _ in range(count):
                    buffer.popleft()
                result = Reconciliation(request_id, matched, text, chunk_ids)
                if not buffer and matched[-1].finished:
                    self._complete.add(request_id)
                    self._buffers.pop(request_id, None)
                return result
            # A prefix cannot recover once either side diverges. Waiting for
            # later events is valid only while both supplied values are longer.
            if len(ids) > len(chunk_ids) or len("".join(raw_parts)) > len(text):
                self._fail(request_id)
                return None
            if tuple(ids) == chunk_ids:
                self._fail(request_id)
                return None
        buffered_text = "".join(raw_parts)
        if ids and len(ids) >= len(chunk_ids) and len(buffered_text) >= len(text):
            self._fail(request_id)
            return None
        return None

    def reconcile(
        self,
        request_id: str,
        token_ids: Iterable[int],
        text: str,
    ) -> Reconciliation | None:
        """Compatibility spelling with token IDs before raw text."""
        return self.reconcile_chunk(request_id, text, token_ids)

    match = reconcile_chunk

    def _fail(self, request_id: str) -> None:
        self._failed.add(request_id)
        self._buffers.pop(request_id, None)
        self._next_sequence.pop(request_id, None)


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
    "Reconciliation",
    "StepReconciler",
    "RenderQueue",
    "BoundedRenderQueue",
    "NonBlockingRenderQueue",
    "sanitize_terminal",
    "palette_color",
    "colorize",
]
