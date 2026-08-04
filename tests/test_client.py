import json
import os
import socket
import time

import pytest

from parallelhue.client import ClientConfig, ClientError, ExactTelemetryError, ParallelHueClient, UnixTelemetryReceiver, iter_sse
from parallelhue.protocol import StepEvent, encode_event


class FakeResponse:
    def __init__(self, payloads):
        self.lines = []
        for payload in payloads:
            self.lines.extend([f"data: {json.dumps(payload)}\n".encode(), b"\n"])
        self.lines.extend([b"data: [DONE]\n", b"\n"])
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


def test_iter_sse_handles_done_and_comments():
    response = [b": keepalive\n", b"data: {\"x\": 1}\n", b"\n", b"data: [DONE]\n", b"\n"]
    assert list(iter_sse(response)) == [{"x": 1}]


def test_chunk_mode_posts_openai_compatible_request():
    seen = {}

    def opener(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return FakeResponse([{"choices": [{"delta": {"content": "hello", "token_ids": [17]}}]}])

    config = ClientConfig(endpoint="http://localhost/v1/chat/completions", model="demo", prompt="hi", mode="chunk", api_key="secret")
    items = list(ParallelHueClient(config, opener=opener).stream())
    body = json.loads(seen["request"].data)
    assert body["stream"] is True
    assert body["return_token_ids"] is True
    assert body["model"] == "demo"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["request_id"].startswith("ph1_")
    assert "prompt" not in body
    assert seen["request"].get_header("Authorization") == "Bearer secret"
    assert items[0].token_ids == (17,)
    assert items[0].mode == "SSE CHUNK MODE"


def test_reasoning_and_content_delta_fields_render_in_order():
    def opener(request, timeout):
        return FakeResponse([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning": "think"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {
                "content": "content",
                "reasoning_content": "reasoning-content",
                "reasoning": "reasoning",
            }}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])

    config = ClientConfig(prompt="hi", mode="chunk")
    items = list(ParallelHueClient(config, opener=opener).stream())

    assert [item.raw_text for item in items] == [
        "think",
        "answer",
        "reasoningreasoning-contentcontent",
    ]
    assert all(item.text for item in items)


def test_completion_mode_posts_prompt_and_request_id():
    seen = {}

    def opener(request, timeout):
        seen["request"] = request
        return FakeResponse([{"choices": [{"text": "hello", "token_ids": [17]}]}])

    config = ClientConfig(endpoint="http://localhost/v1/completions", model="demo", prompt="hi", mode="chunk")
    list(ParallelHueClient(config, opener=opener).stream())
    body = json.loads(seen["request"].data)
    assert body["prompt"] == "hi"
    assert "messages" not in body
    assert body["request_id"].startswith("ph1_")

def test_exact_mode_fails_closed_without_telemetry(tmp_path):
    def opener(request, timeout):
        return FakeResponse([{"choices": [{"delta": {"content": "hello", "token_ids": [17]}}]}])

    config = ClientConfig(endpoint="http://localhost", prompt="hi", mode="exact", socket_dir=str(tmp_path))
    with pytest.raises(ExactTelemetryError):
        list(ParallelHueClient(config, opener=opener).stream())
    assert not list(tmp_path.iterdir())


def test_auto_mode_labels_absent_telemetry_as_chunks(tmp_path):
    def opener(request, timeout):
        return FakeResponse([{"choices": [{"delta": {"content": "hello", "token_ids": [17]}}]}])

    config = ClientConfig(prompt="hi", mode="auto", socket_dir=str(tmp_path))
    items = list(ParallelHueClient(config, opener=opener).stream())
    assert items[0].mode == "SSE CHUNK MODE"


def test_run_scoped_socket_accepts_only_own_run(tmp_path):
    receiver = UnixTelemetryReceiver("a" * 32, str(tmp_path)).start()
    try:
        assert os.stat(receiver.path).st_mode & 0o777 == 0o600
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sender.sendto(encode_event(StepEvent(1, "a" * 32, "ph1_" + "a" * 32 + "_0", 0, 9, 0, (3,), "x", False)), receiver.path)
        sender.sendto(encode_event(StepEvent(1, "b" * 32, "ph1_" + "b" * 32 + "_0", 0, 10, 0, (4,), "y", False)), receiver.path)
        sender.close()
        events = []
        for _ in range(30):
            events = receiver.drain()
            if events:
                break
            time.sleep(0.01)
        assert all(event.run_id == "a" * 32 for event in events)
    finally:
        receiver.close()


def test_metadata_frames_do_not_require_telemetry(tmp_path):
    def opener(request, timeout):
        return FakeResponse([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "hello", "token_ids": [17]}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])

    config = ClientConfig(prompt="hi", mode="auto", socket_dir=str(tmp_path))
    items = list(ParallelHueClient(config, opener=opener).stream())
    assert [item.raw_text for item in items] == ["hello"]
    assert items[0].mode == "SSE CHUNK MODE"


def test_exact_coalesced_events_emit_one_color_per_step(tmp_path):
    def opener(request, timeout):
        request_id = json.loads(request.data)["request_id"]
        run_id = request_id.split("_")[1]
        path = str(tmp_path / f"{run_id}.sock")
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sender.sendto(encode_event(StepEvent(1, run_id, request_id, 0, 0, 0, (3,), "a", False)), path)
        sender.sendto(encode_event(StepEvent(1, run_id, request_id, 1, 1, 0, (4,), "b", True)), path)
        sender.close()
        return FakeResponse([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "ab", "token_ids": [3, 4]}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])

    config = ClientConfig(prompt="hi", mode="exact", socket_dir=str(tmp_path))
    items = list(ParallelHueClient(config, opener=opener).stream())
    assert [item.raw_text for item in items] == ["a", "b"]
    assert [item.step_id for item in items] == [0, 1]
    assert all(item.mode == "EXACT SCHEDULER STEP" for item in items)
    assert "\x1b[38;5;46m" in items[0].text
    assert "\x1b[38;5;196m" in items[1].text


def test_http_error_is_terminal_sanitized():
    import io
    from urllib.error import HTTPError

    def opener(request, timeout):
        raise HTTPError(request.full_url, 500, "bad", {}, io.BytesIO(b"\x1b]2;owned\a"))

    config = ClientConfig(prompt="hi", mode="chunk")
    with pytest.raises(Exception) as caught:
        list(ParallelHueClient(config, opener=opener).stream())
    assert "\x1b" not in str(caught.value)

@pytest.mark.parametrize("token_id", [True, 1.5, "17", -1])
def test_chunk_token_ids_reject_non_exact_nonnegative_ints(token_id):
    def opener(request, timeout):
        return FakeResponse([{"choices": [{"delta": {"content": "hello", "token_ids": [token_id]}}]}])

    config = ClientConfig(prompt="hi", mode="chunk")
    items = list(ParallelHueClient(config, opener=opener).stream())
    assert items[0].raw_text == "hello"
    assert items[0].token_ids == ()


def test_stream_many_preserves_final_chunk():
    def opener(request, timeout):
        return FakeResponse([
            {"choices": [{"delta": {"content": "first", "token_ids": [1]}}]},
            {"choices": [{"delta": {"content": "last", "token_ids": [2]}}]},
        ])

    config = ClientConfig(prompt="hi", mode="chunk", concurrency=1)
    items = list(ParallelHueClient(config, opener=opener).stream_many(["one"]))
    assert [item.raw_text for item in items] == ["first", "last"]


def test_stream_many_propagates_worker_exception():
    def opener(request, timeout):
        raise RuntimeError("worker failed")

    config = ClientConfig(prompt="hi", mode="chunk", concurrency=1)
    with pytest.raises(RuntimeError, match="worker failed"):
        list(ParallelHueClient(config, opener=opener).stream_many(["one"]))


@pytest.mark.parametrize("method_name", ["stream", "stream_many"])
def test_auto_receiver_start_failure_downgrades_and_cleans_up(monkeypatch, tmp_path, method_name):
    closed = []

    def fail_start(self):
        raise RuntimeError("receiver unavailable")

    def record_close(self):
        closed.append(self)

    monkeypatch.setattr(UnixTelemetryReceiver, "start", fail_start)
    monkeypatch.setattr(UnixTelemetryReceiver, "close", record_close)

    def opener(request, timeout):
        return FakeResponse([{"choices": [{"delta": {"content": "hello", "token_ids": [1]}}]}])

    config = ClientConfig(prompt="hi", mode="auto", socket_dir=str(tmp_path), concurrency=1)
    client = ParallelHueClient(config, opener=opener)
    items = list(client.stream() if method_name == "stream" else client.stream_many(["one"]))
    assert [item.mode for item in items] == ["SSE CHUNK MODE"]
    assert len(closed) == 1


@pytest.mark.parametrize("method_name", ["stream", "stream_many"])
def test_exact_receiver_start_failure_raises_and_cleans_up(monkeypatch, tmp_path, method_name):
    closed = []

    def fail_start(self):
        raise RuntimeError("receiver unavailable")

    def record_close(self):
        closed.append(self)

    monkeypatch.setattr(UnixTelemetryReceiver, "start", fail_start)
    monkeypatch.setattr(UnixTelemetryReceiver, "close", record_close)
    config = ClientConfig(prompt="hi", mode="exact", socket_dir=str(tmp_path), concurrency=1)
    client = ParallelHueClient(config, opener=lambda request, timeout: FakeResponse([]))
    with pytest.raises(RuntimeError, match="receiver unavailable"):
        list(client.stream() if method_name == "stream" else client.stream_many(["one"]))
    assert len(closed) == 1


def test_socket_directory_owner_is_checked_before_chmod(monkeypatch, tmp_path):
    current_uid = os.lstat(tmp_path).st_uid
    chmod_calls = []
    monkeypatch.setattr(os, "getuid", lambda: current_uid + 1)
    monkeypatch.setattr(os, "chmod", lambda *args: chmod_calls.append(args))
    receiver = UnixTelemetryReceiver("a" * 32, str(tmp_path))
    with pytest.raises(ClientError, match="not owned"):
        receiver.start()
    assert chmod_calls == []
