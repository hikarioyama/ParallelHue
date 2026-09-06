"""Opt-in provenance telemetry adapter for supported vLLM V1 contracts.

The scheduler process emits token IDs and authoritative roles.  The output
processor process emits raw incremental text and a native-detokenizer trace.
They are intentionally separate frames: vLLM transports ``EngineCoreOutput``
between processes and does not provide a safe extension field for plugins.
The client joins frames by request ID, absolute token offset, and token IDs.
"""

from __future__ import annotations

import importlib
import inspect
from functools import wraps
import os
import queue
import re
import stat
import threading
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Callable, Mapping

Payload = tuple[Any, ...]

_MAX_TIMESTAMP_STEPS = 4096
_MAX_REQUEST_SEQUENCES = 4096
_MAX_STREAM_INDEX = 4095
_MAX_PENDING_TEXT = 4096

_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_VERSION_RE = re.compile(r"^0\.26\.\d+(?:[.+-].*)?$")
_SUPPORTED_FORK_VERSION = "0.1.dev20051+g487ecf187"
_REQUEST_RE = re.compile(
    r"^(?:(?P<chat_prefix>chatcmpl-)?"
    r"(?P<request>ph1_(?P<run>[0-9a-f]{32})_(?P<stream>\d+))"
    r"|cmpl-(?P<completion>ph1_(?P<completion_run>[0-9a-f]{32})_"
    r"(?P<completion_stream>\d+))-(?P<prompt_index>\d+))$"
)
_ENABLED_VALUES = frozenset(("1", "true", "yes", "on", "enable", "enabled"))
_DISABLED_VALUES = frozenset(("0", "false", "no", "off", "disable", "disabled"))
_DRAFT_METHOD = "dflash"


class VllmPluginError(RuntimeError):
    """Raised when exact mode was requested but its private contract is absent."""


_GLOBAL_STEP_LOCK = threading.Lock()
_GLOBAL_NEXT_STEP_ID = 0
_GLOBAL_STEPS: OrderedDict[Any, int] = OrderedDict()


class _Dispatcher:
    """Bounded, best-effort datagram sender kept off vLLM hot paths."""

    def __init__(self, socket_dir: Path, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError("queue size must be positive")
        self.socket_dir = socket_dir
        self.socket_uid = int(os.environ.get("PARALLELHUE_SOCKET_UID", str(os.getuid())))
        if self.socket_uid < 0 or os.getuid() not in (0, self.socket_uid):
            raise VllmPluginError("socket owner must be the current uid, unless the sender is root")
        self.queue: queue.Queue[tuple[str, Payload]] = queue.Queue(maxsize=maxsize)
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

    def submit(self, frame_type: str, payload: Payload) -> bool:
        """Enqueue without ever blocking vLLM."""
        try:
            self.queue.put_nowait((frame_type, payload))
        except queue.Full:
            self.dropped += 1
            return False
        return True

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        from .protocol import (
            ProvenanceFrame,
            TextFrame,
            TraceSpan,
            encode_provenance,
            encode_text,
        )

        import socket

        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.setblocking(False)
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    frame_type, item = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if frame_type == "provenance":
                        (
                            schema_version,
                            run_id,
                            request_id,
                            source_sequence,
                            token_offset,
                            token_ids,
                            roles,
                            finished,
                        ) = item
                        encoded = encode_provenance(
                            ProvenanceFrame(
                                schema_version=schema_version,
                                run_id=run_id,
                                request_id=request_id,
                                source_sequence=source_sequence,
                                token_offset=token_offset,
                                token_ids=token_ids,
                                roles=roles,
                                finished=finished,
                            )
                        )
                    elif frame_type == "text":
                        (
                            schema_version,
                            run_id,
                            request_id,
                            source_sequence,
                            token_offset,
                            step_id,
                            choice_index,
                            token_ids,
                            text,
                            trace,
                            finished,
                        ) = item
                        encoded = encode_text(
                            TextFrame(
                                schema_version=schema_version,
                                run_id=run_id,
                                request_id=request_id,
                                source_sequence=source_sequence,
                                token_offset=token_offset,
                                token_ids=token_ids,
                                text=text,
                                trace=tuple(TraceSpan(*span) for span in trace),
                                step_id=step_id,
                                choice_index=choice_index,
                                finished=finished,
                            )
                        )
                    else:
                        self.invalid_socket += 1
                        continue
                    run_id = item[1]
                    destination = self.socket_dir / f"{run_id}.sock"
                    if not validate_socket(destination, self.socket_dir, socket_uid=self.socket_uid):
                        self.invalid_socket += 1
                        continue
                    sock.sendto(encoded, os.fspath(destination))
                    self.sent += 1
                except (OSError, ValueError, TypeError, KeyError):
                    self.invalid_socket += 1
                finally:
                    self.queue.task_done()


class _PendingText:
    __slots__ = (
        "token_offset",
        "token_ids",
        "text",
        "trace",
        "step_id",
        "choice_index",
        "finished",
    )

    def __init__(
        self,
        token_offset: int,
        token_ids: tuple[int, ...],
        text: str,
        trace: tuple[tuple[int, int, int, int], ...],
        step_id: int,
        choice_index: int,
        finished: bool,
    ) -> None:
        self.token_offset = token_offset
        self.token_ids = token_ids
        self.text = text
        self.trace = trace
        self.step_id = step_id
        self.choice_index = choice_index
        self.finished = finished


class _TraceState:
    """Consume native decoder pieces once, retaining only un-emitted bytes."""

    def __init__(self, detokenizer: Any) -> None:
        self.detokenizer = detokenizer
        self.pieces: deque[tuple[bytes, int, int]] = deque()
        self.pending_start = 0
        self.failed = False

    def record_piece(self, token_id: int, piece: Any) -> None:
        if type(token_id) is not int or token_id < 0 or not isinstance(piece, str):
            self.failed = True
            return
        end = self.detokenizer.num_output_tokens()
        if end <= 0 or self.detokenizer.token_ids[-1] != token_id:
            self.failed = True
            return
        if piece:
            self.pieces.append((piece.encode("utf-8"), self.pending_start, end))
            self.pending_start = end
        if len(self.pieces) > _MAX_PENDING_TEXT:
            self.failed = True

    def trace_for(
        self, token_offset: int, token_ids: tuple[int, ...], text: str,
    ) -> tuple[tuple[int, int, int, int], ...] | None:
        if self.failed:
            return None
        all_ids = self.detokenizer.token_ids
        prompt_length = len(all_ids) - self.detokenizer.num_output_tokens()
        start = prompt_length + token_offset
        if tuple(all_ids[start:start + len(token_ids)]) != token_ids:
            self.failed = True
            return None
        wanted = text.encode("utf-8")
        cursor = 0
        spans = []
        while cursor < len(wanted):
            if not self.pieces:
                self.failed = True
                return None
            data, token_start, token_end = self.pieces.popleft()
            count = min(len(data), len(wanted) - cursor)
            if data[:count] != wanted[cursor:cursor + count]:
                self.failed = True
                return None
            spans.append((cursor, cursor + count, token_start, token_end))
            cursor += count
            if count < len(data):
                self.pieces.appendleft((data[count:], token_start, token_end))
        return tuple(spans)


class _ProvenancePlan:
    __slots__ = ("request_id", "token_offset", "token_ids", "roles")

    def __init__(
        self,
        request_id: str,
        token_offset: int,
        token_ids: tuple[int, ...],
        roles: tuple[str, ...],
    ) -> None:
        self.request_id = request_id
        self.token_offset = token_offset
        self.token_ids = token_ids
        self.roles = roles


class VllmExactPlugin:
    """Own hooks and dispatchers for one vLLM engine/output process."""

    def __init__(self, socket_dir: str | os.PathLike[str], queue_size: int = 256) -> None:
        self.socket_dir = Path(socket_dir)
        self.dispatcher = _Dispatcher(self.socket_dir, queue_size)
        self._local = threading.local()
        self._sequence_lock = threading.Lock()
        self._sequences: OrderedDict[str, int] = OrderedDict()
        self._text_sequences: OrderedDict[str, int] = OrderedDict()
        self._prov_sequences: OrderedDict[str, int] = OrderedDict()
        self._text_offsets: OrderedDict[str, int] = OrderedDict()
        self._pending_text: dict[str, deque[_PendingText]] = {}
        self._trace_states: dict[int, _TraceState] = {}
        self._request_trace_keys: dict[str, int] = {}
        self.invalid_requests = 0
        self.observed = 0
        self._patched = False
        self._scheduler_capable = False

    def close(self) -> None:
        self.dispatcher.close()

    def step_for_timestamp(self, timestamp: Any) -> int:
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

    def _canonical_request(self, request_id: Any) -> tuple[str, str, int] | None:
        parsed = _parse_request_id(request_id)
        if parsed is None:
            return None
        return parsed

    def _record_invalid(self) -> None:
        with self._sequence_lock:
            self.invalid_requests += 1

    def _register_request_state(self, request_state: Any) -> None:
        request_id = getattr(request_state, "external_req_id", None)
        parsed = self._canonical_request(request_id)
        if parsed is None:
            return
        detokenizer = getattr(request_state, "detokenizer", None)
        if detokenizer is not None and len(self._trace_states) < _MAX_REQUEST_SEQUENCES:
            self._trace_states[id(detokenizer)] = _TraceState(detokenizer)
            self._request_trace_keys[parsed[0]] = id(detokenizer)

    def _record_decode_piece(self, detokenizer: Any, token_id: int, piece: Any) -> None:
        trace = self._trace_states.get(id(detokenizer))
        if trace is not None:
            trace.record_piece(token_id, piece)

    def _capture_completion(self, request_state: Any, output: Any) -> None:
        request_id = getattr(request_state, "external_req_id", None)
        parsed = self._canonical_request(request_id)
        if parsed is None:
            self._record_invalid()
            return
        canonical_request_id, _run_id, _stream_index = parsed
        token_ids = _token_tuple(getattr(output, "token_ids", ()))
        offset = self._text_offsets.get(canonical_request_id, 0)
        trace_state = self._trace_states.get(id(getattr(request_state, "detokenizer", None)))
        text = getattr(output, "text", "")
        if not isinstance(text, str):
            self._record_invalid()
            return
        output_kind = getattr(request_state, "output_kind", None)
        if output_kind is not None:
            output_kind_name = getattr(output_kind, "name", str(output_kind)).upper()
            if "DELTA" not in output_kind_name:
                # Cumulative/full output cannot be joined to absolute token
                # offsets without re-decoding history; fail closed.
                self._record_invalid()
                return
        trace = trace_state.trace_for(offset, token_ids, text) if trace_state is not None else None
        if trace is None:
            self._record_invalid()
            return
        output_finished = bool(getattr(output, "finish_reason", None))
        pending = self._pending_text.setdefault(canonical_request_id, deque())
        if len(pending) >= _MAX_PENDING_TEXT:
            self._record_invalid()
            return
        pending.append(
            _PendingText(
                offset,
                token_ids,
                text,
                trace,
                getattr(self._local, "step_id", 0),
                _nonnegative_int(getattr(output, "index", 0), 0),
                output_finished,
            )
        )
        self._text_offsets[canonical_request_id] = offset + len(token_ids)

    def observe_request_output(self, request_output: Any) -> int:
        """Queue text frames from an unmerged RequestOutput."""
        request_id = getattr(request_output, "request_id", None)
        parsed = self._canonical_request(request_id)
        if parsed is None:
            self._record_invalid()
            return 0
        canonical_request_id, run_id, _stream_index = parsed
        outputs = getattr(request_output, "outputs", None)
        if outputs is None:
            outputs = ()
        try:
            choices = tuple(outputs)
        except TypeError:
            choices = ()
        if len(choices) > 1:
            self._record_invalid()
            return 0
        request_finished = _completion_state(getattr(request_output, "finished", False))
        pending = self._pending_text.get(canonical_request_id)
        if pending is None or not choices:
            self._record_invalid()
            return 0
        count = 0
        for output in choices:
            if not pending:
                self._record_invalid()
                break
            item = pending.popleft()
            observed_ids = _token_tuple(getattr(output, "token_ids", ()))
            observed_text = getattr(output, "text", "")
            if observed_ids != item.token_ids or observed_text != item.text:
                self._record_invalid()
                continue
            finished = request_finished or item.finished
            sequence = self._text_sequences.get(canonical_request_id, 0)
            self._text_sequences[canonical_request_id] = sequence + 1
            payload: Payload = (
                2,
                run_id,
                canonical_request_id,
                sequence,
                item.token_offset,
                item.step_id,
                item.choice_index,
                item.token_ids,
                item.text,
                item.trace,
                finished,
            )
            self.observed += 1
            if self.dispatcher.submit("text", payload):
                count += 1
            if finished:
                self._finish_request(canonical_request_id)
        return count

    def _finish_request(self, request_id: str) -> None:
        self._pending_text.pop(request_id, None)
        self._text_sequences.pop(request_id, None)
        self._prov_sequences.pop(request_id, None)
        self._text_offsets.pop(request_id, None)
        key = self._request_trace_keys.pop(request_id, None)
        if key is not None:
            self._trace_states.pop(key, None)

    def _classify_roles(
        self,
        scheduler: Any,
        request_id: str,
        draft_ids: Any,
        generated_ids: Any,
    ) -> tuple[str, ...] | None:
        generated = _token_tuple(generated_ids)
        if getattr(scheduler, "num_sampled_tokens_per_step", None) != 1:
            return None
        if getattr(scheduler, "adaptive_mtp_controller", None) is not None:
            return None
        spec_method = _spec_method(scheduler)
        if draft_ids is None:
            if spec_method not in ("", "none", _DRAFT_METHOD):
                return None
            return tuple("target" for _ in generated)
        drafts = _raw_token_tuple(draft_ids)
        if spec_method != _DRAFT_METHOD or drafts is None:
            return None
        # The scheduler counts accepted tokens from the sampler's committed
        # prefix, before stop truncation. Async drafting may use -1 placeholders;
        # those are scheduled slots, not evidence that a token was rejected.
        draft_count = len(drafts)
        if len(generated) == 0:
            return ()
        accepted_count = len(generated) - 1
        if accepted_count < 0 or accepted_count > draft_count:
            return None
        if any(drafts[index] >= 0 and generated[index] != drafts[index] for index in range(accepted_count)):
            return None
        final_role = "bonus" if draft_count > 0 and accepted_count == draft_count else "target"
        return tuple(["accepted_draft"] * accepted_count + [final_role])

    def _scheduler_plans(self, scheduler: Any, scheduler_output: Any, model_output: Any) -> dict[str, _ProvenancePlan]:
        sampled = getattr(model_output, "sampled_token_ids", None)
        req_ids = getattr(model_output, "req_ids", None)
        if not isinstance(sampled, (list, tuple)) or not isinstance(req_ids, (list, tuple)):
            return {}
        scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
        requests = getattr(scheduler, "requests", {})
        plans: dict[str, _ProvenancePlan] = {}
        for index, internal_id in enumerate(req_ids):
            if index >= len(sampled):
                self._record_invalid()
                continue
            request = requests.get(internal_id) if hasattr(requests, "get") else None
            external_id = getattr(request, "external_req_id", None) or internal_id
            parsed = self._canonical_request(external_id)
            if parsed is None:
                self._record_invalid()
                continue
            canonical, _run_id, _stream_index = parsed
            prior_ids = getattr(request, "_output_token_ids", None)
            if not isinstance(prior_ids, list):
                self._record_invalid()
                continue
            generated = _token_tuple(sampled[index])
            draft_ids = scheduled.get(internal_id) if hasattr(scheduled, "get") else None
            roles = self._classify_roles(scheduler, canonical, draft_ids, generated)
            if roles is None:
                self._record_invalid()
                continue
            plans[internal_id] = _ProvenancePlan(
                canonical,
                len(prior_ids),
                generated,
                roles,
            )
        return plans

    def observe_scheduler_output(
        self,
        scheduler: Any,
        scheduler_output: Any,
        model_output: Any,
        outputs: Any,
        plans: dict[str, _ProvenancePlan] | None = None,
    ) -> None:
        if plans is None:
            plans = self._scheduler_plans(scheduler, scheduler_output, model_output)
        if not isinstance(outputs, Mapping):
            self._record_invalid()
            return
        for engine_outputs in outputs.values():
            if isinstance(engine_outputs, (list, tuple)):
                output_items = engine_outputs
            else:
                output_items = getattr(engine_outputs, "outputs", ())
            for output in output_items:
                plan = plans.get(getattr(output, "request_id", None))
                if plan is None:
                    continue
                emitted = _token_tuple(getattr(output, "new_token_ids", ()))
                if len(emitted) > len(plan.token_ids) or emitted != plan.token_ids[: len(emitted)]:
                    self._record_invalid()
                    continue
                if not emitted and not bool(getattr(output, "finished", False)):
                    continue
                sequence = self._prov_sequences.get(plan.request_id, 0)
                self._prov_sequences[plan.request_id] = sequence + 1
                payload: Payload = (
                    2,
                    _run_id_from_request(plan.request_id),
                    plan.request_id,
                    sequence,
                    plan.token_offset,
                    emitted,
                    plan.roles[: len(emitted)],
                    bool(getattr(output, "finished", False)),
                )
                self.dispatcher.submit("provenance", payload)
                if bool(getattr(output, "finished", False)):
                    self._prov_sequences.pop(plan.request_id, None)


    def patch(self, vllm_module: Any, scheduler: Any | None = None) -> "VllmExactPlugin":
        output_processor, collector = _resolve_capabilities(vllm_module)
        _patch_method(output_processor, "process_outputs", self._process_wrapper)
        _patch_method(collector, "put", self._put_wrapper)
        request_state, detokenizers = _resolve_detokenizer_capabilities()
        if request_state is not None:
            _patch_method(request_state, "__init__", self._request_state_wrapper)
            _patch_method(request_state, "_new_completion_output", self._completion_wrapper)
        for detokenizer in detokenizers:
            _patch_method(detokenizer, "decode_next", self._decode_wrapper)
        if scheduler is not None:
            _patch_method(scheduler, "update_from_output", self._scheduler_wrapper)
            self._scheduler_capable = True
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
            self.observe_request_output(request_output)
        return original(*args, **kwargs)

    def _request_state_wrapper(self, original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        request_state = args[0] if args else kwargs.get("self")
        if request_state is not None:
            self._register_request_state(request_state)
        return result

    def _decode_wrapper(self, original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        detokenizer = args[0] if args else kwargs.get("self")
        token_id = args[1] if len(args) > 1 else kwargs.get("next_token_id")
        self._record_decode_piece(detokenizer, token_id, result)
        return result

    def _completion_wrapper(self, original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        request_state = args[0] if args else kwargs.get("self")
        if request_state is not None and result is not None:
            self._capture_completion(request_state, result)
        return result

    def _scheduler_wrapper(self, original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        scheduler = args[0] if args else kwargs.get("self")
        scheduler_output = kwargs.get("scheduler_output")
        model_output = kwargs.get("model_runner_output")
        if len(args) > 1:
            scheduler_output = args[1]
        if len(args) > 2:
            model_output = args[2]
        plans = {}
        if scheduler is not None and scheduler_output is not None and model_output is not None:
            # Request._output_token_ids is mutated by the original method, so
            # capture absolute offsets before handing control to vLLM.
            plans = self._scheduler_plans(scheduler, scheduler_output, model_output)
        outputs = original(*args, **kwargs)
        if scheduler is not None and scheduler_output is not None and model_output is not None:
            self.observe_scheduler_output(scheduler, scheduler_output, model_output, outputs, plans)
        return outputs


_PATCHED_CLASSES: dict[tuple[type[Any], str], VllmExactPlugin] = {}
_PATCH_LOCK = threading.RLock()


def _patch_method(owner: Any, name: str, wrapper_factory: Callable[..., Any]) -> None:
    with _PATCH_LOCK:
        current = getattr(owner, name, None)
        if current is None or not callable(current):
            raise VllmPluginError(f"vLLM capability {owner!r}.{name} is not callable")
        if getattr(current, "__parallelhue_vllm_wrapper__", False):
            return

        @wraps(current)
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
        raise VllmPluginError("vLLM exact output hooks are unavailable")
    if not callable(getattr(output_processor, "process_outputs", None)):
        raise VllmPluginError("OutputProcessor.process_outputs is unavailable")
    if not callable(getattr(collector, "put", None)):
        raise VllmPluginError("RequestOutputCollector.put is unavailable")
    return output_processor, collector


def _resolve_scheduler(module: Any) -> Any | None:
    candidates = [module]
    for dotted in ("vllm.v1.core.sched.scheduler", "vllm.core.scheduler"):
        try:
            candidates.append(importlib.import_module(dotted))
        except ImportError:
            pass
    for candidate in candidates:
        scheduler = getattr(candidate, "Scheduler", None)
        if callable(getattr(scheduler, "update_from_output", None)):
            return scheduler
    return None


def _resolve_detokenizer_capabilities() -> tuple[Any | None, tuple[Any, ...]]:
    try:
        module = importlib.import_module("vllm.v1.engine.output_processor")
        request_state = getattr(module, "RequestState", None)
        detok_module = importlib.import_module("vllm.v1.engine.detokenizer")
    except ImportError:
        return None, ()
    detokenizers = tuple(
        cls
        for cls in (
            getattr(detok_module, "FastIncrementalDetokenizer", None),
            getattr(detok_module, "SlowIncrementalDetokenizer", None),
        )
        if cls is not None
    )
    return request_state, detokenizers


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
    if match is None and re.search(r"-[0-9a-f]{8}$", request_id):
        # EngineCore Request keeps only the randomized internal ID; unlike
        # the API RequestState, it has no external_req_id field.
        match = _REQUEST_RE.fullmatch(request_id[:-9])
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


def _run_id_from_request(request_id: str) -> str:
    parsed = _parse_request_id(request_id)
    if parsed is None:
        raise VllmPluginError("request ID is not ParallelHue-scoped")
    return parsed[1]


def _completion_state(value: Any) -> bool:
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
        values = tuple(value)
    except TypeError:
        return ()
    if any(type(item) is not int or item < 0 for item in values):
        return ()
    return values

def _raw_token_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        values = tuple(value)
    except TypeError:
        return None
    if any(type(item) is not int or item < -1 for item in values):
        return None
    return values




def _spec_method(scheduler: Any) -> str:
    config = getattr(scheduler, "vllm_config", None)
    speculative = getattr(config, "speculative_config", None)
    if speculative is None:
        return "none"
    method = getattr(speculative, "method", None)
    method = getattr(method, "value", method)
    return str(method or "").strip().lower()


def validate_socket(path: str | os.PathLike[str], socket_dir: str | os.PathLike[str] | None = None, *, socket_uid: int | None = None) -> bool:
    """Validate a run socket without following symlinks or relaxing ownership."""
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
    owner = os.getuid() if socket_uid is None else socket_uid
    if type(owner) is not int or owner < 0 or os.getuid() not in (0, owner):
        return False
    if directory_stat.st_uid != owner or socket_stat.st_uid != owner:
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
    return isinstance(version, str) and (
        _VERSION_RE.fullmatch(version) is not None or version == _SUPPORTED_FORK_VERSION
    )


def _stream_interval(processor: Any) -> int | None:
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
            raise VllmPluginError("exact mode requires a supported vLLM V1 API") from exc
    if socket_dir is None:
        socket_dir = os.environ.get("PARALLELHUE_SOCKET_DIR")
    if not socket_dir:
        raise VllmPluginError("exact mode requires PARALLELHUE_SOCKET_DIR")
    if not supports_vllm_version(vllm_module):
        raise VllmPluginError("unsupported vLLM version for exact token provenance")
    if stream_interval not in (None, 1):
        raise VllmPluginError("exact mode requires stream_interval=1")
    output_processor, _collector = _resolve_capabilities(vllm_module)
    scheduler = _resolve_scheduler(vllm_module)
    request_state, detokenizers = _resolve_detokenizer_capabilities()
    if scheduler is None or request_state is None or not detokenizers:
        raise VllmPluginError("scheduler and native detokenizer provenance hooks are required")
    required = {
        scheduler.update_from_output: {"scheduler_output", "model_runner_output"},
        output_processor.process_outputs: {"engine_core_outputs", "engine_core_timestamp"},
        request_state._new_completion_output: {"token_ids", "finish_reason"},
    }
    for method, parameters in required.items():
        if not parameters.issubset(inspect.signature(method).parameters):
            raise VllmPluginError("unsupported native provenance hook signature")
    with _PATCH_LOCK:
        existing = _PATCHED_CLASSES.get((output_processor, "process_outputs"))
        if existing is not None:
            return existing
        plugin = VllmExactPlugin(socket_dir, queue_size)
        plugin.patch(vllm_module, scheduler)
        _PATCHED_CLASSES[(output_processor, "process_outputs")] = plugin
        return plugin


def register() -> VllmExactPlugin | None:
    """vLLM general-plugin entry point."""
    return install()



__all__ = [
    "VllmExactPlugin",
    "VllmPluginError",
    "install",
    "register",
    "supports_vllm_version",
    "validate_socket",
]
