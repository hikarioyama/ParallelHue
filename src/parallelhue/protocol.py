"""Shared wire protocol for exact scheduler-step telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

SCHEMA_VERSION = 1
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_REQUEST_ID_RE = re.compile(r"^ph1_([0-9a-f]{32})_([0-9]+)$")
_FIELDS = (
    "schema_version",
    "run_id",
    "request_id",
    "sequence",
    "step_id",
    "choice_index",
    "token_ids",
    "text",
    "finished",
)


class ProtocolError(ValueError):
    """Raised when an event or request identifier violates the wire contract."""


@dataclass(frozen=True, slots=True)
class StepEvent:
    """An immutable scheduler-step output for one request and choice."""

    schema_version: int
    run_id: str
    request_id: str
    sequence: int
    step_id: int
    choice_index: int
    token_ids: tuple[int, ...]
    text: str
    finished: bool

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ProtocolError("schema_version must be 1")
        if type(self.run_id) is not str or _RUN_ID_RE.fullmatch(self.run_id) is None:
            raise ProtocolError("run_id must be 32 lowercase hexadecimal characters")
        parsed_run, _ = parse_request_id(self.request_id)
        if parsed_run != self.run_id:
            raise ProtocolError("request_id run_id does not match run_id")
        for name in ("sequence", "step_id", "choice_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ProtocolError(f"{name} must be a non-negative integer")
        if not isinstance(self.token_ids, (tuple, list)):
            raise ProtocolError("token_ids must be a tuple or list of integers")
        token_ids = tuple(self.token_ids)
        if any(type(token) is not int or token < 0 for token in token_ids):
            raise ProtocolError("token_ids must contain non-negative integers")
        object.__setattr__(self, "token_ids", token_ids)
        if type(self.text) is not str:
            raise ProtocolError("text must be a string")
        if type(self.finished) is not bool:
            raise ProtocolError("finished must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy of this event."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "step_id": self.step_id,
            "choice_index": self.choice_index,
            "token_ids": list(self.token_ids),
            "text": self.text,
            "finished": self.finished,
        }


def parse_request_id(request_id: str) -> tuple[str, int]:
    """Parse ``ph1_<32-lowercase-hex-run-id>_<stream-index>``."""
    if type(request_id) is not str:
        raise ProtocolError("request_id must be a string")
    match = _REQUEST_ID_RE.fullmatch(request_id)
    if match is None:
        raise ProtocolError("invalid request_id")
    return match.group(1), int(match.group(2))


def encode_event(event: StepEvent) -> bytes:
    """Encode an event as deterministic compact UTF-8 JSON."""
    if not isinstance(event, StepEvent):
        raise TypeError("encode_event expects StepEvent")
    return json.dumps(
        event.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def decode_event(payload: bytes | bytearray | memoryview | str) -> StepEvent:
    """Decode and validate one UTF-8 JSON event, rejecting schema extensions."""
    if isinstance(payload, (bytes, bytearray, memoryview)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("event is not valid UTF-8") from exc
    if type(payload) is not str:
        raise TypeError("decode_event expects UTF-8 bytes or str")
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("event is not valid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != set(_FIELDS):
        raise ProtocolError("event has an invalid field set")
    try:
        return StepEvent(**{field: value[field] for field in _FIELDS})
    except (TypeError, ValueError, ProtocolError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("event fields have invalid types") from exc


__all__ = [
    "SCHEMA_VERSION",
    "ProtocolError",
    "StepEvent",
    "encode_event",
    "decode_event",
    "parse_request_id",
]
