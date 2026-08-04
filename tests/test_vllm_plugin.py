from __future__ import annotations

import os
import queue
import socket
import sys
import threading
import tomllib
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from parallelhue.protocol import decode_event
from parallelhue.vllm_plugin import (
    VllmExactPlugin,
    VllmPluginError,
    _MAX_REQUEST_SEQUENCES,
    _Dispatcher,
    _parse_request_id,
    install,
    register,
    validate_socket,
)


RUN_ID = "0123456789abcdef0123456789abcdef"
REQUEST_ID = f"ph1_{RUN_ID}_0"


class FakeOutput:
    def __init__(self, index: int, text: str, token_ids: tuple[int, ...], finished=None):
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.finished = finished


class OfficialCompletionOutput:
    def __init__(
        self,
        index: int,
        text: str,
        token_ids: tuple[int, ...],
        finished: bool,
    ):
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self._finished = finished

    def finished(self) -> bool:
        return self._finished


class FakeRequest:
    def __init__(self, outputs, finished=True, request_id=REQUEST_ID):
        self.request_id = request_id
        self.outputs = outputs
        self.finished = finished


def fake_module(*, root_capabilities=True):
    class OutputProcessor:
        calls = 0

        def __init__(self, stream_interval=1):
            self.stream_interval = stream_interval

        def process_outputs(self, outputs, engine_core_timestamp):
            type(self).calls += 1
            return ("original", outputs, engine_core_timestamp)

    class RequestOutputCollector:
        calls = []

        def put(self, output):
            type(self).calls.append(output)
            return "collector-result"

    attrs = {
        "__version__": "0.26.4",
        "_OutputProcessor": OutputProcessor,
        "_RequestOutputCollector": RequestOutputCollector,
    }
    if root_capabilities:
        attrs.update(
            OutputProcessor=OutputProcessor,
            RequestOutputCollector=RequestOutputCollector,
        )
    return SimpleNamespace(**attrs)


def make_socket(path: Path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(os.fspath(path))
    os.chmod(path, 0o600)
    return sock


def test_disabled_is_noop():
    module = fake_module()
    assert install(module, enabled=False) is None
    assert not hasattr(module.OutputProcessor.process_outputs, "__parallelhue_vllm_wrapper__")


def test_original_call_and_step_grouping(tmp_path):
    os.chmod(tmp_path, 0o700)
    receiver = make_socket(tmp_path / f"{RUN_ID}.sock")
    module = fake_module()
    plugin = install(module, enabled=True, socket_dir=tmp_path)
    try:
        processor = module.OutputProcessor()
        collector = module.RequestOutputCollector()
        assert processor.process_outputs([], 77) == ("original", [], 77)
        request = FakeRequest(
            [FakeOutput(0, "A", (11,), finished=False)],
            finished=False,
        )
        assert collector.put(request) == "collector-result"
        assert processor.process_outputs([], 77) == ("original", [], 77)
        request.finished = True
        request.outputs = [FakeOutput(1, "B", (12,), finished=True)]
        assert collector.put(request) == "collector-result"
        first = decode_event(receiver.recv(4096))
        second = decode_event(receiver.recv(4096))
        assert first.step_id == second.step_id
        assert first.sequence == 0 and second.sequence == 1
        assert first.text == "A" and second.text == "B"
    finally:
        plugin.close()
        receiver.close()


def test_official_finished_method_preserves_sequence_until_terminal():
    plugin = _bare_plugin()
    for index in range(3):
        request = FakeRequest(
            [
                OfficialCompletionOutput(
                    index, f"chunk-{index}", (index,), finished=False
                )
            ],
            finished=False,
        )
        assert plugin.observe_request_output(request) == 1
        assert REQUEST_ID in plugin._sequences
    assert plugin._sequences[REQUEST_ID] == 3

    terminal = FakeRequest(
        [OfficialCompletionOutput(3, "done", (3,), finished=True)],
        finished=False,
    )
    assert plugin.observe_request_output(terminal) == 1
    assert [payload[3] for payload in plugin.dispatcher.payloads] == [0, 1, 2, 3]
    assert [payload[8] for payload in plugin.dispatcher.payloads] == [
        False,
        False,
        False,
        True,
    ]
    assert plugin._sequences == {}


def test_attribute_finished_flags_remain_compatible():
    plugin = _bare_plugin()
    assert plugin.observe_request_output(
        FakeRequest([FakeOutput(0, "chunk", (1,), finished=False)], finished=False)
    ) == 1
    assert REQUEST_ID in plugin._sequences
    assert plugin.observe_request_output(
        FakeRequest([FakeOutput(0, "done", (2,), finished=True)], finished=False)
    ) == 1
    assert plugin._sequences == {}


def test_request_terminal_is_not_overwritten_by_output_false():
    plugin = _bare_plugin()
    request = FakeRequest(
        [FakeOutput(0, "done", (1,), finished=False)],
        finished=True,
    )
    assert plugin.observe_request_output(request) == 1
    assert plugin.dispatcher.payloads[0][8] is True
    assert plugin._sequences == {}


def test_invalid_finished_values_fail_closed():
    class RaisingFinished:
        def __call__(self):
            raise RuntimeError("completion unavailable")

    class InvalidFinished:
        def __call__(self):
            return object()

    plugin = _bare_plugin()
    for finished in (RaisingFinished(), InvalidFinished(), 1, "false", object()):
        request = FakeRequest(
            [FakeOutput(0, "chunk", (1,), finished=finished)],
            finished=False,
        )
        assert (
            plugin._put_wrapper(lambda output: "collector-result", request)
            == "collector-result"
        )

    assert [payload[8] for payload in plugin.dispatcher.payloads] == [
        False,
        False,
        False,
        False,
        False,
    ]
    assert plugin._sequences[REQUEST_ID] == 5


def test_multi_choice_payloads_are_skipped(tmp_path):
    os.chmod(tmp_path, 0o700)
    receiver = make_socket(tmp_path / f"{RUN_ID}.sock")
    receiver.settimeout(0.05)
    module = fake_module()
    plugin = install(module, enabled=True, socket_dir=tmp_path)
    try:
        module.OutputProcessor().process_outputs([], 1)
        assert (
            module.RequestOutputCollector().put(
                FakeRequest([FakeOutput(0, "x", (1,)), FakeOutput(1, "y", (2,))])
            )
            == "collector-result"
        )
        assert plugin.invalid_requests == 1
        with pytest.raises(socket.timeout):
            receiver.recv(4096)
    finally:
        plugin.close()
        receiver.close()


def test_unsupported_version_fails_loudly(tmp_path):
    module = fake_module()
    module.__version__ = "0.27.0"
    with pytest.raises(VllmPluginError, match="0.26"):
        install(module, enabled=True, socket_dir=tmp_path)


def test_bound_stream_interval_must_be_one(tmp_path):
    module = fake_module()
    plugin = install(module, enabled=True, socket_dir=tmp_path)
    try:
        with pytest.raises(VllmPluginError, match="stream_interval=1"):
            module.OutputProcessor(stream_interval=2).process_outputs([], 1)
    finally:
        plugin.close()


def test_zero_argument_entrypoint_uses_official_vllm_module(monkeypatch, tmp_path):
    os.chmod(tmp_path, 0o700)
    receiver = make_socket(tmp_path / f"{RUN_ID}.sock")
    root = ModuleType("vllm")
    root.__version__ = "0.26.4"
    root.__path__ = []
    v1 = ModuleType("vllm.v1")
    v1.__path__ = []
    engine = ModuleType("vllm.v1.engine")
    engine.__path__ = []
    official = ModuleType("vllm.v1.engine.output_processor")
    module = fake_module(root_capabilities=False)
    official.OutputProcessor = module._OutputProcessor
    official.RequestOutputCollector = module._RequestOutputCollector
    monkeypatch.setitem(sys.modules, "vllm", root)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.output_processor", official)
    monkeypatch.setenv("PARALLELHUE_VLLM_EXACT", "1")
    monkeypatch.setenv("PARALLELHUE_SOCKET_DIR", str(tmp_path))
    plugin = register()
    try:
        assert plugin is not None
        assert module._OutputProcessor().process_outputs([], 1) == ("original", [], 1)
    finally:
        plugin.close()
        receiver.close()


def test_package_metadata_registers_plugin_and_dev_extra():
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert metadata["project"]["optional-dependencies"]["dev"] == ["pytest"]
    assert metadata["project"]["entry-points"]["vllm.general_plugins"]["parallelhue"] == (
        "parallelhue.vllm_plugin:register"
    )


def test_queue_overflow_is_nonblocking(tmp_path):
    dispatcher = object.__new__(_Dispatcher)
    dispatcher.queue = queue.Queue(maxsize=1)
    dispatcher.dropped = 0
    payload = (1, RUN_ID, REQUEST_ID, 0, 0, 0, (1,), "x", False)
    assert dispatcher.submit(payload)
    assert not dispatcher.submit(payload)
    assert dispatcher.dropped == 1


def test_socket_validation_rejects_wrong_mode_and_symlink(tmp_path):
    os.chmod(tmp_path, 0o700)
    path = tmp_path / f"{RUN_ID}.sock"
    receiver = make_socket(path)
    try:
        assert validate_socket(path, tmp_path)
        os.chmod(path, 0o666)
        assert not validate_socket(path, tmp_path)
        os.chmod(path, 0o600)
        alias = tmp_path / "alias.sock"
        alias.symlink_to(path)
        assert not validate_socket(alias, tmp_path)
    finally:
        receiver.close()


def test_collector_rejects_non_capability_request_without_changing_original(tmp_path):
    os.chmod(tmp_path, 0o700)
    module = fake_module()
    plugin = install(module, enabled=True, socket_dir=tmp_path)
    try:
        module.OutputProcessor().process_outputs([], 1)
        collector = module.RequestOutputCollector()
        request = FakeRequest([FakeOutput(0, "x", (1,))], request_id="ordinary-id")
        assert collector.put(request) == "collector-result"
        assert plugin.invalid_requests == 1
    finally:
        plugin.close()


@pytest.mark.parametrize(
    ("request_id", "canonical", "stream"),
    [
        (f"ph1_{RUN_ID}_0", f"ph1_{RUN_ID}_0", 0),
        (f"chatcmpl-ph1_{RUN_ID}_1", f"ph1_{RUN_ID}_1", 1),
        (f"cmpl-ph1_{RUN_ID}_4095-17", f"ph1_{RUN_ID}_4095", 4095),
    ],
)
def test_request_id_parser_accepts_official_forms(
    request_id: str, canonical: str, stream: int
):
    assert _parse_request_id(request_id) == (canonical, RUN_ID, stream)


@pytest.mark.parametrize(
    "request_id",
    [
        f"ph1_{RUN_ID}_{'9' * 5000}",
        f"ph1_{RUN_ID}_4096",
        f"cmpl-ph1_{RUN_ID}_0",
        f"cmpl-ph1_{RUN_ID}_0-prompt",
        f"cmpl-ph1_{RUN_ID}_0-1-extra",
        f"chatcmpl-ph1_{RUN_ID}_0-1",
        f"ph1_{RUN_ID}_0-1",
        f"other-ph1_{RUN_ID}_0",
    ],
)
def test_request_id_parser_rejects_nonofficial_suffixes(request_id: str):
    assert _parse_request_id(request_id) is None


def _bare_plugin():
    class RecordingDispatcher:
        def __init__(self):
            self.payloads = []

        def submit(self, payload):
            self.payloads.append(payload)
            return True

    plugin = object.__new__(VllmExactPlugin)
    plugin.dispatcher = RecordingDispatcher()
    plugin._local = threading.local()
    plugin._local.step_id = 7
    plugin._sequence_lock = threading.Lock()
    plugin._sequences = OrderedDict()
    plugin.invalid_requests = 0
    plugin.observed = 0
    return plugin


def test_terminal_requests_release_sequence_state_without_lifetime_exhaustion():
    plugin = _bare_plugin()
    for stream in range(_MAX_REQUEST_SEQUENCES):
        request = FakeRequest(
            [FakeOutput(0, "x", (1,))],
            request_id=f"ph1_{RUN_ID}_{stream}",
        )
        assert plugin.observe_request_output(request) == 1
    assert plugin._sequences == {}


def test_live_request_sequence_state_remains_bounded():
    plugin = _bare_plugin()
    for stream in range(_MAX_REQUEST_SEQUENCES):
        request = FakeRequest(
            [FakeOutput(0, "x", (1,), finished=False)],
            finished=False,
            request_id=f"ph1_{RUN_ID}_{stream}",
        )
        assert plugin.observe_request_output(request) == 1
    assert len(plugin._sequences) == _MAX_REQUEST_SEQUENCES
    overflow = FakeRequest(
        [FakeOutput(0, "x", (1,), finished=False)],
        finished=False,
        request_id=f"ph1_{RUN_ID}_{_MAX_REQUEST_SEQUENCES}",
    )
    assert plugin.observe_request_output(overflow) == 0
    assert plugin.invalid_requests == 1
