"""Strict two-frame wire protocol for exact speculative provenance."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal, Mapping

SCHEMA_VERSION = 2
TokenRole = Literal["accepted_draft", "target", "bonus"]
_TOKEN_ROLES = frozenset(("accepted_draft", "target", "bonus"))
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_REQUEST_ID_RE = re.compile(r"^ph1_([0-9a-f]{32})_([0-9]+)$")
_FRAME_TYPES = frozenset(("provenance", "text"))
_PROVENANCE_FIELDS = (
    "frame_type",
    "schema_version",
    "run_id",
    "request_id",
    "source_sequence",
    "token_offset",
    "token_ids",
    "roles",
    "finished",
)
_TEXT_FIELDS = (
    "frame_type",
    "schema_version",
    "run_id",
    "request_id",
    "source_sequence",
    "token_offset",
    "token_ids",
    "text",
    "trace",
    "step_id",
    "choice_index",
    "finished",
)


class ProtocolError(ValueError):
    """Raised when a telemetry frame violates the wire contract."""


def _validate_envelope(
    *,
    frame_type: str,
    schema_version: int,
    run_id: str,
    request_id: str,
    source_sequence: int,
    token_offset: int,
) -> None:
    if frame_type not in _FRAME_TYPES:
        raise ProtocolError("frame_type is invalid")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ProtocolError("schema_version must be 2")
    if type(run_id) is not str or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ProtocolError("run_id must be 32 lowercase hexadecimal characters")
    parsed_run, _ = parse_request_id(request_id)
    if parsed_run != run_id:
        raise ProtocolError("request_id run_id does not match run_id")
    for name, value in (
        ("source_sequence", source_sequence),
        ("token_offset", token_offset),
    ):
        if type(value) is not int or value < 0:
            raise ProtocolError(f"{name} must be a non-negative integer")


def _token_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ProtocolError("token_ids must be a tuple or list")
    tokens = tuple(value)
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ProtocolError("token_ids must contain non-negative integers")
    return tokens


def _role_tuple(value: Any) -> tuple[TokenRole, ...]:
    if not isinstance(value, (tuple, list)):
        raise ProtocolError("roles must be a tuple or list")
    roles = tuple(value)
    if any(type(role) is not str or role not in _TOKEN_ROLES for role in roles):
        raise ProtocolError("roles contain an unknown provenance value")
    return roles  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TraceSpan:
    """UTF-8 text bytes attributed to an absolute token range.

    ``token_offset_start``/``token_offset_end`` are half-open and may cover
    multiple tokens when native incremental detokenization defers a code point
    until a later token.  Consumers must treat such a span as mixed when its
    roles disagree; it is never safe to assign it to the last token.
    """

    byte_start: int
    byte_end: int
    token_offset_start: int
    token_offset_end: int

    def __post_init__(self) -> None:
        for name in (
            "byte_start",
            "byte_end",
            "token_offset_start",
            "token_offset_end",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ProtocolError(f"{name} must be a non-negative integer")
        if self.byte_end <= self.byte_start:
            raise ProtocolError("trace byte range must be non-empty")
        if self.token_offset_end <= self.token_offset_start:
            raise ProtocolError("trace token range must be non-empty")

    def to_dict(self) -> dict[str, int]:
        return {
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "token_offset_start": self.token_offset_start,
            "token_offset_end": self.token_offset_end,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceFrame:
    """Scheduler-authoritative token IDs and roles, without text."""

    schema_version: int
    run_id: str
    request_id: str
    source_sequence: int
    token_offset: int
    token_ids: tuple[int, ...]
    roles: tuple[TokenRole, ...]
    finished: bool
    frame_type: str = "provenance"

    def __post_init__(self) -> None:
        _validate_envelope(
            frame_type=self.frame_type,
            schema_version=self.schema_version,
            run_id=self.run_id,
            request_id=self.request_id,
            source_sequence=self.source_sequence,
            token_offset=self.token_offset,
        )
        token_ids = _token_tuple(self.token_ids)
        roles = _role_tuple(self.roles)
        if len(token_ids) != len(roles):
            raise ProtocolError("token_ids and roles must have equal lengths")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "roles", roles)
        if type(self.finished) is not bool:
            raise ProtocolError("finished must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_type": self.frame_type,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "source_sequence": self.source_sequence,
            "token_offset": self.token_offset,
            "token_ids": list(self.token_ids),
            "roles": list(self.roles),
            "finished": self.finished,
        }


def _trace_tuple(value: Any) -> tuple[TraceSpan, ...]:
    if not isinstance(value, (tuple, list)):
        raise ProtocolError("trace must be a tuple or list")
    spans: list[TraceSpan] = []
    for item in value:
        if isinstance(item, TraceSpan):
            spans.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ProtocolError("trace entries must be objects")
        try:
            spans.append(TraceSpan(**{field: item[field] for field in (
                "byte_start",
                "byte_end",
                "token_offset_start",
                "token_offset_end",
            )}))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("trace entry has invalid fields") from exc
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class TextFrame:
    """Output-processor text and native-detokenizer attribution trace."""

    schema_version: int
    run_id: str
    request_id: str
    source_sequence: int
    token_offset: int
    token_ids: tuple[int, ...]
    text: str
    trace: tuple[TraceSpan, ...]
    step_id: int
    choice_index: int
    finished: bool
    frame_type: str = "text"

    def __post_init__(self) -> None:
        _validate_envelope(
            frame_type=self.frame_type,
            schema_version=self.schema_version,
            run_id=self.run_id,
            request_id=self.request_id,
            source_sequence=self.source_sequence,
            token_offset=self.token_offset,
        )
        token_ids = _token_tuple(self.token_ids)
        object.__setattr__(self, "token_ids", token_ids)
        if type(self.text) is not str:
            raise ProtocolError("text must be a string")
        for name in ("step_id", "choice_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ProtocolError(f"{name} must be a non-negative integer")
        if type(self.finished) is not bool:
            raise ProtocolError("finished must be a boolean")
        spans = _trace_tuple(self.trace)
        text_bytes = self.text.encode("utf-8")
        if not text_bytes and spans:
            raise ProtocolError("empty text cannot carry trace spans")
        cursor = 0
        for span in spans:
            if span.byte_start != cursor:
                raise ProtocolError("trace must cover text bytes contiguously")
            if span.byte_end > len(text_bytes):
                raise ProtocolError("trace exceeds UTF-8 text length")
            try:
                text_bytes[span.byte_start:span.byte_end].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProtocolError("trace must align with UTF-8 codepoint boundaries") from exc
            cursor = span.byte_end
        if cursor != len(text_bytes):
            raise ProtocolError("trace must cover every emitted text byte")
        object.__setattr__(self, "trace", spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_type": self.frame_type,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "source_sequence": self.source_sequence,
            "token_offset": self.token_offset,
            "token_ids": list(self.token_ids),
            "text": self.text,
            "trace": [span.to_dict() for span in self.trace],
            "step_id": self.step_id,
            "choice_index": self.choice_index,
            "finished": self.finished,
        }


@dataclass(frozen=True, slots=True)
class TokenRecord:
    """Client-derived token text/provenance record."""

    token_id: int
    text: str
    role: TokenRole

    def __post_init__(self) -> None:
        _token_tuple((self.token_id,))
        if type(self.text) is not str:
            raise ProtocolError("token text must be a string")
        _role_tuple((self.role,))




def parse_request_id(request_id: str) -> tuple[str, int]:
    """Parse ``ph1_<32-lowercase-hex-run-id>_<stream-index>``."""
    if type(request_id) is not str:
        raise ProtocolError("request_id must be a string")
    match = _REQUEST_ID_RE.fullmatch(request_id)
    if match is None:
        raise ProtocolError("invalid request_id")
    return match.group(1), int(match.group(2))


def _encode(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def encode_provenance(frame: ProvenanceFrame) -> bytes:
    if not isinstance(frame, ProvenanceFrame):
        raise TypeError("encode_provenance expects ProvenanceFrame")
    return _encode(frame.to_dict())


def encode_text(frame: TextFrame) -> bytes:
    if not isinstance(frame, TextFrame):
        raise TypeError("encode_text expects TextFrame")
    return _encode(frame.to_dict())


def _decode_payload(payload: bytes | bytearray | memoryview | str) -> Mapping[str, Any]:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("frame is not valid UTF-8") from exc
    if type(payload) is not str:
        raise TypeError("decode_frame expects UTF-8 bytes or str")
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("frame is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ProtocolError("frame must be a JSON object")
    return value


def decode_frame(payload: bytes | bytearray | memoryview | str) -> ProvenanceFrame | TextFrame:
    """Decode one strict provenance or text frame."""
    value = _decode_payload(payload)
    frame_type = value.get("frame_type")
    if frame_type == "provenance":
        fields = _PROVENANCE_FIELDS
        cls = ProvenanceFrame
    elif frame_type == "text":
        fields = _TEXT_FIELDS
        cls = TextFrame
    else:
        raise ProtocolError("frame_type is invalid")
    if set(value) != set(fields):
        raise ProtocolError("frame has an invalid field set")
    try:
        return cls(**{field: value[field] for field in fields})
    except (TypeError, ValueError, ProtocolError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("frame fields have invalid types") from exc




__all__ = [
    "SCHEMA_VERSION",
    "TokenRole",
    "ProtocolError",
    "TraceSpan",
    "ProvenanceFrame",
    "TextFrame",
    "TokenRecord",
    "encode_provenance",
    "encode_text",
    "decode_frame",
    "parse_request_id",
]
