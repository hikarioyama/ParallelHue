"""Standard-library OpenAI-compatible streaming client for ParallelHue.

The client deliberately treats scheduler telemetry as optional.  Only the exact
mode is allowed to claim scheduler-step coloring; ordinary SSE chunks are always
labeled as such.
"""
from __future__ import annotations

import json
import os
import queue
import secrets
import socket
import stat
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from .protocol import StepEvent, decode_event, parse_request_id
from .render import StepReconciler, colorize, sanitize_terminal

PALETTE = (46, 196, 27, 226)


class ClientError(RuntimeError):
    """Base class for client failures."""


class ExactTelemetryError(ClientError):
    """Exact mode could not prove a one-to-one telemetry match."""


@dataclass(frozen=True)
class ClientConfig:
    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    model: str = ""
    prompt: str = ""
    max_tokens: int = 128
    concurrency: int = 1
    api_key: str | None = None
    mode: str = "auto"
    socket_dir: str | None = None
    timeout: float = 60.0
    stream_interval: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"exact", "auto", "chunk"}:
            raise ValueError("mode must be exact, auto, or chunk")
        if self.max_tokens < 1 or self.concurrency < 1:
            raise ValueError("max_tokens and concurrency must be positive")
        if self.stream_interval != 1:
            raise ValueError("ParallelHue requires stream_interval=1")


@dataclass(frozen=True)
class StreamChunk:
    request_id: str
    sequence: int
    text: str
    token_ids: tuple[int, ...] = ()
    finished: bool = False
    mode: str = "SSE CHUNK MODE"
    color: int | None = None
    step_id: int | None = None
    raw_text: str = field(default="", repr=False)


class UnixTelemetryReceiver:
    """Bounded, run-scoped AF_UNIX datagram receiver.

    The socket is created with restrictive permissions and is removed on close.
    Datagrams are decoded in a background thread, while callers drain a bounded
    queue.  Overflow is retained as a fail-closed flag rather than silently
    losing scheduler steps.
    """

    def __init__(self, run_id: str, socket_dir: str | None = None, max_events: int = 2048):
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be positive")
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("run_id must be 32 lowercase hexadecimal characters")
        self.run_id = run_id
        self.socket_dir = os.path.abspath(socket_dir or os.environ.get("PARALLELHUE_SOCKET_DIR", "/tmp/parallelhue"))
        self.path = os.path.join(self.socket_dir, f"{run_id}.sock")
        self._events: queue.Queue[StepEvent] = queue.Queue(maxsize=max_events)
        self._mailboxes: dict[str, list[StepEvent]] = {}
        self._condition = threading.Condition()
        self._max_events = max_events
        self._overflow = False
        self._stop = threading.Event()
        self._ready = False
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
    @property
    def overflow(self) -> bool:
        return self._overflow

    @property
    def ready(self) -> bool:
        return self._ready

    def start(self) -> "UnixTelemetryReceiver":
        os.makedirs(self.socket_dir, mode=0o700, exist_ok=True)
        directory = os.lstat(self.socket_dir)
        if stat.S_ISLNK(directory.st_mode) or not stat.S_ISDIR(directory.st_mode):
            raise ClientError("telemetry socket directory must be a non-symlink directory")
        if directory.st_uid != os.getuid():
            raise ClientError("telemetry socket directory is not owned by current uid")
        os.chmod(self.socket_dir, 0o700)
        if os.path.lexists(self.path):
            if os.path.islink(self.path) or not stat.S_ISSOCK(os.lstat(self.path).st_mode):
                raise ClientError("telemetry socket path already exists and is not a socket")
            old = os.lstat(self.path)
            if old.st_uid != os.getuid():
                raise ClientError("telemetry socket is not owned by current uid")
            os.unlink(self.path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.bind(self.path)
            os.chmod(self.path, 0o600)
            ps = os.stat(self.path)
            if ps.st_uid != os.getuid():
                raise ClientError("telemetry socket is not owned by current uid")
            sock.settimeout(0.1)
        except Exception:
            sock.close()
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            raise
        self._sock = sock
        self._ready = True
        self._thread = threading.Thread(target=self._receive, name=f"parallelhue-telemetry-{self.run_id[:8]}", daemon=True)
        self._thread.start()
        return self

    def _receive(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data = self._sock.recv(1 << 20)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                event = decode_event(data)
                if event.run_id != self.run_id:
                    continue
                with self._condition:
                    total = sum(len(items) for items in self._mailboxes.values())
                    if total >= self._max_events:
                        self._overflow = True
                    else:
                        self._mailboxes.setdefault(event.request_id, []).append(event)
                    self._condition.notify_all()
            except (ValueError, TypeError, json.JSONDecodeError):
                with self._condition:
                    self._overflow = True
                    self._condition.notify_all()

    def drain(self, request_id: str | None = None) -> list[StepEvent]:
        with self._condition:
            if request_id is None:
                out = [event for mailbox in self._mailboxes.values() for event in mailbox]
                self._mailboxes.clear()
                return out
            return self._mailboxes.pop(request_id, [])

    def wait_for(self, request_id: str, timeout: float) -> list[StepEvent]:
        """Wait briefly for this request's mailbox without draining peers."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._mailboxes.get(request_id) and not self._overflow and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._mailboxes.pop(request_id, [])
    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sock = None
        self._ready = False
        try:
            if os.path.lexists(self.path) and not os.path.islink(self.path):
                os.unlink(self.path)
        except FileNotFoundError:
            pass

    def __enter__(self) -> "UnixTelemetryReceiver":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()


def _new_run_id() -> str:
    return secrets.token_hex(16)


def _extract_text(choice: Mapping[str, Any]) -> str:
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        parts = [
            value
            for field in ("reasoning", "reasoning_content", "content")
            if isinstance(value := delta.get(field), str)
        ]
        return "".join(parts)
    value = choice.get("text")
    return value if isinstance(value, str) else ""

def _extract_token_ids(payload: Mapping[str, Any], choice: Mapping[str, Any]) -> tuple[int, ...]:
    candidates = (choice.get("token_ids"), (choice.get("delta") or {}).get("token_ids"), payload.get("output_token_ids"), payload.get("token_ids"))
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and all(type(value) is int and value >= 0 for value in candidate):
            return tuple(candidate)
    return ()


def iter_sse(response: Any) -> Iterator[dict[str, Any]]:
    """Yield JSON SSE payloads, ignoring comments and terminating at [DONE]."""
    data_lines: list[str] = []
    for raw in response:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines.clear()
                if payload == "[DONE]":
                    return
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ClientError(f"invalid SSE JSON: {exc}") from exc
                if isinstance(value, dict):
                    yield value
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ClientError(f"invalid SSE JSON: {exc}") from exc
            if isinstance(value, dict):
                yield value


class ParallelHueClient:
    """Small synchronous client; concurrent streams use bounded worker queues."""

    def __init__(self, config: ClientConfig | None = None, opener: Any = urllib.request.urlopen):
        self.config = config or ClientConfig()
        self._opener = opener

    def _request(self, prompt: str, request_id: str) -> Any:
        endpoint = self.config.endpoint.rstrip("/")
        is_chat = endpoint.endswith("/chat/completions")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "return_token_ids": True,
            "request_id": request_id,
        }
        if is_chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-ParallelHue-Request-ID": request_id,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(self.config.endpoint, data=data, headers=headers, method="POST")
        try:
            return self._opener(req, timeout=self.config.timeout)
        except urllib.error.HTTPError as exc:
            body = sanitize_terminal(exc.read(4096).decode("utf-8", "replace"))
            raise ClientError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            reason = sanitize_terminal(str(exc.reason))
            raise ClientError(f"unable to reach endpoint: {reason}") from exc

    @staticmethod
    def _result_color(result: Any) -> tuple[int | None, int | None]:
        events = getattr(result, "events", None)
        if isinstance(events, (list, tuple)) and events:
            event = events[0]
            return getattr(event, "step_id", None), PALETTE[getattr(event, "choice_index", 0) % len(PALETTE)]
        if isinstance(result, (list, tuple)) and result:
            event = result[0]
            return getattr(event, "step_id", None), PALETTE[getattr(event, "choice_index", 0) % len(PALETTE)]
        event = getattr(result, "event", result)
        if event is not None:
            return getattr(event, "step_id", None), getattr(result, "color", None) or PALETTE[getattr(event, "choice_index", 0) % len(PALETTE)]
        return None, None

    def _stream_with_receiver(self, prompt: str, stream_index: int, run_id: str, receiver: UnixTelemetryReceiver | None) -> Iterator[StreamChunk]:
        request_id = f"ph1_{run_id}_{stream_index}"
        reconciler = StepReconciler() if self.config.mode in {"exact", "auto"} else None
        exact_enabled = self.config.mode in {"exact", "auto"}
        auto_downgraded = False
        seen = False
        terminal_matched = False
        sequence = 0
        response = self._request(prompt, request_id)

        def collect_events() -> None:
            nonlocal seen
            if receiver is None or reconciler is None or auto_downgraded:
                return
            events = receiver.drain(request_id)
            if not events:
                events = receiver.wait_for(request_id, min(0.1, max(0.01, self.config.timeout * 0.01)))
            for event in events:
                seen = True
                reconciler.feed(event)

        try:
            for payload in iter_sse(response):
                choices = payload.get("choices") or [{}]
                choice = choices[0] if isinstance(choices[0], dict) else {}
                raw_text = _extract_text(choice)
                token_ids = _extract_token_ids(payload, choice)
                finished = bool(choice.get("finish_reason"))
                # Role-only and final metadata frames carry no model output and
                # must never be presented to the exact reconciler.
                if not raw_text and not token_ids:
                    continue

                result = None
                telemetry_failed = False
                if exact_enabled and not auto_downgraded:
                    collect_events()
                    result = reconciler.reconcile_chunk(request_id, raw_text, token_ids) if reconciler is not None else None
                    for _ in range(4):
                        if result is not None or (reconciler is not None and reconciler.failed(request_id)):
                            break
                        if receiver is None or receiver.overflow:
                            break
                        collect_events()
                        result = reconciler.reconcile_chunk(request_id, raw_text, token_ids) if reconciler is not None else None
                    telemetry_failed = bool(reconciler and reconciler.failed(request_id))
                    if receiver is None or receiver.overflow or telemetry_failed or (result is None and self.config.mode == "auto"):
                        if self.config.mode == "exact":
                            raise ExactTelemetryError("SSE chunk does not exactly match scheduler telemetry")
                        auto_downgraded = True
                        result = None

                if self.config.mode == "exact":
                    if receiver is None or receiver.overflow or telemetry_failed or result is None:
                        raise ExactTelemetryError("exact telemetry is unavailable or has gaps")
                    events = tuple(result.events)
                    terminal_matched = terminal_matched or any(event.finished for event in events)
                    for event in events:
                        safe = sanitize_terminal(event.text)
                        yield StreamChunk(
                            request_id, sequence, colorize(safe, step_id=event.step_id) if safe else "",
                            tuple(event.token_ids), event.finished, "EXACT SCHEDULER STEP",
                            PALETTE[event.step_id % len(PALETTE)], event.step_id, event.text,
                        )
                        sequence += 1
                elif result is not None and not auto_downgraded:
                    events = tuple(result.events)
                    terminal_matched = terminal_matched or any(event.finished for event in events)
                    for event in events:
                        safe = sanitize_terminal(event.text)
                        yield StreamChunk(
                            request_id, sequence, colorize(safe, step_id=event.step_id) if safe else "",
                            tuple(event.token_ids), event.finished, "EXACT SCHEDULER STEP",
                            PALETTE[event.step_id % len(PALETTE)], event.step_id, event.text,
                        )
                        sequence += 1
                else:
                    safe = sanitize_terminal(raw_text)
                    yield StreamChunk(
                        request_id, sequence, colorize(safe, PALETTE[sequence % len(PALETTE)]) if safe else "",
                        tuple(token_ids), finished, "SSE CHUNK MODE", PALETTE[sequence % len(PALETTE)], None, raw_text,
                    )
                    sequence += 1
            if self.config.mode == "exact" and (
                receiver is None or receiver.overflow or not seen or not terminal_matched or reconciler is None or reconciler.failed(request_id)
            ):
                raise ExactTelemetryError("exact telemetry was absent or incomplete")
        finally:
            close = getattr(response, "close", None)
            if close:
                close()

    def stream(self, prompt: str | None = None, stream_index: int = 0) -> Iterator[StreamChunk]:
        run_id = _new_run_id()
        receiver = UnixTelemetryReceiver(run_id, self.config.socket_dir) if self.config.mode in {"exact", "auto"} else None
        try:
            if receiver is not None:
                try:
                    receiver.start()
                except BaseException:
                    failed_receiver = receiver
                    receiver = None
                    failed_receiver.close()
                    if self.config.mode == "exact":
                        raise
            yield from self._stream_with_receiver(prompt if prompt is not None else self.config.prompt, stream_index, run_id, receiver)
        finally:
            if receiver is not None:
                receiver.close()

    def stream_many(self, prompts: Sequence[str] | None = None) -> Iterator[StreamChunk]:
        values = list(prompts if prompts is not None else [self.config.prompt])
        if not values:
            return
        run_id = _new_run_id()
        receiver = UnixTelemetryReceiver(run_id, self.config.socket_dir) if self.config.mode in {"exact", "auto"} else None
        cancelled = threading.Event()
        try:
            if receiver is not None:
                try:
                    receiver.start()
                except BaseException:
                    failed_receiver = receiver
                    receiver = None
                    failed_receiver.close()
                    if self.config.mode == "exact":
                        raise

            events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=max(8, self.config.concurrency * 4))
            tasks: queue.Queue[tuple[int, str]] = queue.Queue()
            for index, prompt in enumerate(values):
                tasks.put((index, prompt))

            def put_event(kind: str, payload: object) -> bool:
                while not cancelled.is_set():
                    try:
                        events.put((kind, payload), timeout=0.05)
                        return True
                    except queue.Full:
                        continue
                return False

            def worker() -> None:
                failed = False
                try:
                    while not cancelled.is_set():
                        try:
                            index, prompt = tasks.get_nowait()
                        except queue.Empty:
                            return
                        for item in self._stream_with_receiver(prompt, index, run_id, receiver):
                            if not put_event("output", item):
                                return
                except BaseException as exc:
                    if not cancelled.is_set():
                        failed = put_event("error", exc)
                finally:
                    if failed or not cancelled.is_set():
                        put_event("done", None)
                    if failed:
                        cancelled.set()

            worker_count = min(self.config.concurrency, len(values))
            threads = [threading.Thread(target=worker, daemon=True, name=f"parallelhue-worker-{i}") for i in range(worker_count)]
            for thread in threads:
                thread.start()
            done = 0
            while done < worker_count:
                kind, payload = events.get()
                if kind == "output":
                    yield payload  # type: ignore[misc]
                elif kind == "error":
                    raise payload  # type: ignore[misc]
                elif kind == "done":
                    done += 1
        finally:
            cancelled.set()
            if receiver is not None:
                receiver.close()
            for thread in locals().get("threads", ()):
                thread.join(timeout=1)


def mode_label(mode: str) -> str:
    return {"exact": "EXACT SCHEDULER STEP", "auto": "AUTO", "chunk": "SSE CHUNK MODE"}[mode]
