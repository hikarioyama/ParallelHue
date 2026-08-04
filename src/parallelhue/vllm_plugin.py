"""Opt-in exact scheduler-step adapter for the private vLLM 0.26 API.

The two hooks in this module deliberately do very little: they snapshot plain
Python values into a bounded queue. Encoding and socket I/O happen on the
background dispatcher thread, never in vLLM's output path.
"""
from __future__ import annotations

import importlib
import os
import queue
import re
import socket
import stat
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping

Payload = tuple[int, str, str, int, int, int, tuple[int, ...], str, bool]

_MAX_TIMESTAMP_STEPS = 4096

_MAX_REQUEST_SEQUENCES = 4096
_MAX_STREAM_INDEX = 4095


_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_VERSION_RE = re.compile(r"^0\.26\.\d+(?:[.+-].*)?$")
_REQUEST_RE = re.compile(
    r"^(?:(?P<chat_prefix>chatcmpl-)?"
    r"(?P<request>ph1_(?P<run>[0-9a-f]{32})_(?P<stream>\d+))"
    r"|cmpl-(?P<completion>ph1_(?P<completion_run>[0-9a-f]{32})_"
    r"(?P<completion_stream>\d+))-(?P<prompt_index>\d+))$"
)
_ENABLED_VALUES = frozenset(("1", "true", "yes", "on", "enable", "enabled"))
_DISABLED_VALUES = frozenset(("0", "false", "no", "off", "disable", "disabled"))


class VllmPluginError(RuntimeError):
    """Raised when exact mode was requested but its private contract is absent."""


_GLOBAL_STEP_LOCK = threading.Lock()
_GLOBAL_NEXT_STEP_ID = 0
_GLOBAL_STEPS: OrderedDict[Any, int] = OrderedDict()


class _Dispatcher:
    """Bounded, best-effort datagram sender."""

    def __init__(self, socket_dir: Path, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError("queue size must be positive")
        self.socket_dir = socket_dir
        self.queue: queue.Queue[Payload] = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.sent = 0
        self.invalid_socket = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="parallelhue-vllm-dispatcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, payload: Payload) -> bool:
        """Enqueue without ever blocking vLLM."""
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            self.dropped += 1
            return False
        return True

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # Protocol import is intentionally deferred to this non-hot thread.
        from .protocol import StepEvent, encode_event

        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.setblocking(False)
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    event = StepEvent(
                        schema_version=item[0],
                        run_id=item[1],
                        request_id=item[2],
                        sequence=item[3],
                        step_id=item[4],
                        choice_index=item[5],
                        token_ids=item[6],
                        text=item[7],
                        finished=item[8],
                    )
                    encoded = encode_event(event)
                    destination = self.socket_dir / f"{item[1]}.sock"
                    if not validate_socket(destination, self.socket_dir):
                        self.invalid_socket += 1
                        continue
                    sock.sendto(encoded, os.fspath(destination))
                    self.sent += 1
                except (OSError, ValueError, TypeError):
                    self.invalid_socket += 1
                finally:
                    self.queue.task_done()

class VllmExactPlugin:
    """Owns hooks and dispatcher for one vLLM engine process."""

    def __init__(self, socket_dir: str | os.PathLike[str], queue_size: int = 256) -> None:
        self.socket_dir = Path(socket_dir)
        self.dispatcher = _Dispatcher(self.socket_dir, queue_size)
        self._local = threading.local()
        self._sequence_lock = threading.Lock()
        self._sequences: OrderedDict[str, int] = OrderedDict()
        self.invalid_requests = 0
        self.observed = 0
        self._patched = False

    def close(self) -> None:
        self.dispatcher.close()

    def step_for_timestamp(self, timestamp: Any) -> int:
        """Return one process-global ID for every occurrence of a timestamp."""
        try:
            hash(timestamp)
        except TypeError as exc:
            raise VllmPluginError("engine_core_timestamp must be hashable") from exc
        global _GLOBAL_NEXT_STEP_ID
        with _GLOBAL_STEP_LOCK:
            existing = _GLOBAL_STEPS.get(timestamp)
            if existing is not None:
                _GLOBAL_STEPS.move_to_end(timestamp)
                return existing
            step_id = _GLOBAL_NEXT_STEP_ID
            _GLOBAL_NEXT_STEP_ID += 1
            _GLOBAL_STEPS[timestamp] = step_id
            if len(_GLOBAL_STEPS) > _MAX_TIMESTAMP_STEPS:
                _GLOBAL_STEPS.popitem(last=False)
            return step_id

    def begin_process_outputs(self, timestamp: Any) -> int:
        step_id = self.step_for_timestamp(timestamp)
        self._local.timestamp = timestamp
        self._local.step_id = step_id
        return step_id

    def observe_request_output(self, request_output: Any) -> int:
        """Snapshot an unmerged RequestOutput, returning number of queued items."""
        request_id = getattr(request_output, "request_id", None)
        parsed = _parse_request_id(request_id)
        if parsed is None:
            self.invalid_requests += 1
            return 0
        canonical_request_id, run_id, _stream_index = parsed
        step_id = getattr(self._local, "step_id", None)
        if step_id is None:
            self.invalid_requests += 1
            return 0
        outputs = getattr(request_output, "outputs", None)
        if outputs is None:
            outputs = ()
        try:
            choices = tuple(outputs)
        except TypeError:
            choices = ()
        if len(choices) > 1:
            self.invalid_requests += 1
            return 0
        request_finished = _completion_state(getattr(request_output, "finished", False))
        terminal = request_finished
        with self._sequence_lock:
            if (
                canonical_request_id not in self._sequences
                and len(self._sequences) >= _MAX_REQUEST_SEQUENCES
            ):
                self.invalid_requests += 1
                return 0
            sequence = self._sequences.get(canonical_request_id, 0)
            self._sequences[canonical_request_id] = sequence + 1
            self._sequences.move_to_end(canonical_request_id)
            count = 0
            for fallback_index, output in enumerate(choices):
                choice_index = _nonnegative_int(
                    getattr(output, "index", fallback_index), fallback_index
                )
                token_ids = _token_tuple(getattr(output, "token_ids", ()))
                text = getattr(output, "text", "")
                if not isinstance(text, str):
                    text = str(text)
                output_finished = getattr(output, "finished", None)
                resolved_output_finished = _completion_state(output_finished)
                item_finished = request_finished or resolved_output_finished
                terminal = terminal or item_finished
                payload: Payload = (
                    1,
                    run_id,
                    canonical_request_id,
                    sequence,
                    step_id,
                    choice_index,
                    token_ids,
                    text,
                    item_finished,
                )
                self.observed += 1
                if self.dispatcher.submit(payload):
                    count += 1
            if terminal:
                self._sequences.pop(canonical_request_id, None)
        return count

    def patch(self, vllm_module: Any) -> "VllmExactPlugin":
        output_processor, collector = _resolve_capabilities(vllm_module)
        _patch_method(output_processor, "process_outputs", self._process_wrapper)
        _patch_method(collector, "put", self._put_wrapper)
        self._patched = True
        return self

    def _process_wrapper(self, original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        processor = args[0] if args else kwargs.get("self")
        if _stream_interval(processor) != 1:
            raise VllmPluginError("exact mode requires bound stream_interval=1")
        timestamp = _find_timestamp(args, kwargs)
        if timestamp is None:
            raise VllmPluginError("OutputProcessor.process_outputs lacks engine_core_timestamp")
        self.begin_process_outputs(timestamp)
        return original(*args, **kwargs)

    def _put_wrapper(self, original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        request_output = _find_request_output(args, kwargs)
        if request_output is not None:
            # Observe first; original collector semantics and return value are untouched.
            self.observe_request_output(request_output)
        return original(*args, **kwargs)


_PATCHED_CLASSES: dict[tuple[type[Any], str], VllmExactPlugin] = {}
_PATCH_LOCK = threading.RLock()


def _patch_method(owner: Any, name: str, wrapper_factory: Callable[..., Any]) -> None:
    with _PATCH_LOCK:
        current = getattr(owner, name, None)
        if current is None or not callable(current):
            raise VllmPluginError(f"vLLM capability {owner!r}.{name} is not callable")
        if getattr(current, "__parallelhue_vllm_wrapper__", False):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return wrapper_factory(current, *args, **kwargs)

        wrapped.__parallelhue_vllm_wrapper__ = True
        wrapped.__parallelhue_original__ = current
        setattr(owner, name, wrapped)


def _resolve_capabilities(module: Any) -> tuple[Any, Any]:
    candidates = [module]
    for dotted in (
        "vllm.v1.engine.output_processor",
        "vllm.engine.output_processor",
        "vllm.outputs",
        "vllm.engine.llm_engine",
    ):
        try:
            candidates.append(importlib.import_module(dotted))
        except ImportError:
            pass
    output_processor = collector = None
    for candidate in candidates:
        output_processor = output_processor or getattr(candidate, "OutputProcessor", None)
        collector = collector or getattr(candidate, "RequestOutputCollector", None)
    if output_processor is None or collector is None:
        raise VllmPluginError("vLLM 0.26 exact hooks are unavailable")
    if not callable(getattr(output_processor, "process_outputs", None)):
        raise VllmPluginError("OutputProcessor.process_outputs is unavailable")
    if not callable(getattr(collector, "put", None)):
        raise VllmPluginError("RequestOutputCollector.put is unavailable")
    return output_processor, collector


def _find_timestamp(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    for key in ("engine_core_timestamp", "timestamp"):
        if key in kwargs:
            return kwargs[key]
    for value in args:
        if hasattr(value, "engine_core_timestamp"):
            return getattr(value, "engine_core_timestamp")
    if len(args) >= 3 and isinstance(args[2], (int, float, str, bytes)):
        return args[2]
    if len(args) >= 2 and isinstance(args[1], (int, float, str, bytes)):
        return args[1]
    return None


def _find_request_output(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    for key in ("request_output", "output"):
        if key in kwargs and hasattr(kwargs[key], "request_id"):
            return kwargs[key]
    for value in args:
        if hasattr(value, "request_id") and hasattr(value, "outputs"):
            return value
    return None


def _parse_request_id(request_id: Any) -> tuple[str, str, int] | None:
    if not isinstance(request_id, str):
        return None
    match = _REQUEST_RE.fullmatch(request_id)
    if match is None:
        return None
    completion = match.group("completion")
    if completion is not None:
        stream = match.group("completion_stream")
        if len(stream) > 4 or int(stream) > _MAX_STREAM_INDEX:
            return None
        return completion, match.group("completion_run"), int(stream)
    canonical = match.group("request")
    stream = match.group("stream")
    if len(stream) > 4 or int(stream) > _MAX_STREAM_INDEX:
        return None
    return canonical, match.group("run"), int(stream)


def _completion_state(value: Any) -> bool:
    """Resolve vLLM completion flags while failing closed on malformed values."""
    if callable(value):
        try:
            value = value()
        except Exception:
            return False
    return type(value) is bool and value


def _nonnegative_int(value: Any, fallback: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return value if value >= 0 else fallback


def _token_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return ()


def validate_socket(path: str | os.PathLike[str], socket_dir: str | os.PathLike[str] | None = None) -> bool:
    """Validate a run socket without following symlinks."""
    path = Path(path)
    directory = Path(socket_dir) if socket_dir is not None else path.parent
    try:
        directory_stat = os.lstat(directory)
        socket_stat = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        return False
    if stat.S_ISLNK(socket_stat.st_mode) or not stat.S_ISSOCK(socket_stat.st_mode):
        return False
    if directory_stat.st_uid != os.getuid() or socket_stat.st_uid != os.getuid():
        return False
    if stat.S_IMODE(directory_stat.st_mode) != 0o700 or stat.S_IMODE(socket_stat.st_mode) != 0o600:
        return False
    if not _RUN_ID_RE.fullmatch(path.stem):
        return False
    try:
        return len(os.fspath(path).encode()) < 108
    except UnicodeEncodeError:
        return False


def supports_vllm_version(module: Any) -> bool:
    version = getattr(module, "__version__", None)
    if version is None:
        version_obj = getattr(module, "version", None)
        version = getattr(version_obj, "__version__", version_obj)
    return isinstance(version, str) and _VERSION_RE.fullmatch(version) is not None


def _stream_interval(processor: Any) -> int | None:
    """Read stream_interval from the bound OutputProcessor instance."""
    if processor is None:
        return None
    candidates = [processor]
    for name in ("scheduler_config", "engine_config", "config"):
        candidate = getattr(processor, name, None)
        if candidate is not None:
            candidates.append(candidate)
    for candidate in candidates:
        if hasattr(candidate, "stream_interval"):
            return getattr(candidate, "stream_interval")
    return None


def install(
    vllm_module: Any | None = None,
    *,
    enabled: bool | None = None,
    socket_dir: str | os.PathLike[str] | None = None,
    queue_size: int = 256,
    stream_interval: int | None = None,
) -> VllmExactPlugin | None:
    """Install exact hooks when explicitly enabled; disabled mode is a no-op."""
    if enabled is None:
        raw = os.environ.get("PARALLELHUE_VLLM_EXACT", "")
        enabled = raw.strip().lower() in _ENABLED_VALUES
        if raw.strip().lower() in _DISABLED_VALUES or not raw.strip():
            enabled = False
    if not enabled:
        return None
    if vllm_module is None:
        try:
            vllm_module = importlib.import_module("vllm")
        except ImportError as exc:
            raise VllmPluginError("exact mode requires vLLM 0.26.x") from exc
    if not supports_vllm_version(vllm_module):
        raise VllmPluginError("exact mode requires vLLM version 0.26.x")
    if socket_dir is None:
        socket_dir = os.environ.get("PARALLELHUE_SOCKET_DIR")
    if not socket_dir:
        raise VllmPluginError("exact mode requires PARALLELHUE_SOCKET_DIR")
    output_processor, collector = _resolve_capabilities(vllm_module)
    with _PATCH_LOCK:
        existing = _PATCHED_CLASSES.get((output_processor, "process_outputs"))
        if existing is not None:
            return existing
        plugin = VllmExactPlugin(socket_dir, queue_size)
        plugin.patch(vllm_module)
        _PATCHED_CLASSES[(output_processor, "process_outputs")] = plugin
        _PATCHED_CLASSES[(collector, "put")] = plugin
        return plugin


def register() -> VllmExactPlugin | None:
    """vLLM general-plugin entry point."""
    return install()


register_plugin = register
patch_vllm = install

__all__ = [
    "VllmExactPlugin",
    "VllmPluginError",
    "install",
    "patch_vllm",
    "register",
    "register_plugin",
    "supports_vllm_version",
    "validate_socket",
]
