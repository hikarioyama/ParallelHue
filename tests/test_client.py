import io
import json
import os
import socket
from urllib.error import HTTPError

import pytest

from parallelhue.client import ClientConfig, ClientError, ExactTelemetryError, ParallelHueClient, UnixTelemetryReceiver, iter_sse
from parallelhue.protocol import ProvenanceFrame, TextFrame, TraceSpan, encode_provenance, encode_text
from parallelhue.render import PALETTE, sanitize_terminal


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


def _exact_event_opener(socket_dir, events):
    def opener(request, timeout):
        request_id = json.loads(request.data)["request_id"]
        run = request_id.split("_")[1]
        payloads = []
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            for sequence, (step_id, token_id, value, role) in enumerate(events):
                finished = sequence == len(events) - 1
                provenance = ProvenanceFrame(
                    2, run, request_id, sequence, sequence, (token_id,), (role,), finished,
                )
                text = TextFrame(
                    2, run, request_id, sequence, sequence, (token_id,), value,
                    (TraceSpan(0, len(value.encode("utf-8")), sequence, sequence + 1),),
                    step_id, 0, finished,
                )
                sender.sendto(encode_provenance(provenance), str(socket_dir / f"{run}.sock"))
                sender.sendto(encode_text(text), str(socket_dir / f"{run}.sock"))
                choice = {"delta": {"content": value, "token_ids": [token_id]}}
                if finished:
                    choice["finish_reason"] = "length"
                payloads.append({"choices": [choice]})
        return FakeResponse(payloads)

    return opener


def test_iter_sse_handles_done_and_comments():
    response = [b": keepalive\n", b'data: {"x": 1}\n', b"\n", b"data: [DONE]\n", b"\n"]
    assert list(iter_sse(response)) == [{"x": 1}]


def test_exact_client_uses_authoritative_roles_and_one_step_color(socket_dir, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    def opener(request, timeout):
        request_id = json.loads(request.data)["request_id"]
        run = request_id.split("_")[1]
        p = ProvenanceFrame(2, run, request_id, 0, 0, (10, 11, 12), ("accepted_draft", "target", "bonus"), True)
        t = TextFrame(2, run, request_id, 0, 0, (10, 11, 12), "abc", (TraceSpan(0, 1, 0, 1), TraceSpan(1, 2, 1, 2), TraceSpan(2, 3, 2, 3)), 8, 0, True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.sendto(encode_text(t), str(socket_dir / f"{run}.sock"))
            sender.sendto(encode_provenance(p), str(socket_dir / f"{run}.sock"))
        return FakeResponse([{"choices": [{"delta": {"content": "abc", "token_ids": [10, 11, 12]}, "finish_reason": "length"}]}])

    config = ClientConfig(prompt="hi", mode="exact", backend="dflash", socket_dir=str(socket_dir), timeout=0.2)
    items = list(ParallelHueClient(config, opener=opener).stream())
    assert [role for item in items for role in item.roles] == ["accepted_draft", "target", "bonus"]
    assert sanitize_terminal("".join(item.text for item in items)) == "abc"
    assert all(item.color == PALETTE[0] for item in items)
    assert all(f"\x1b[38;5;{PALETTE[0]}m" in item.text for item in items)
    assert all(
        f"\x1b[38;5;{color}m" not in item.text
        for item in items
        for color in PALETTE[1:]
    )
    assert items[-1].finished
    assert not list(socket_dir.iterdir())


def test_exact_verified_steps_cycle_palette_by_observation_and_wrap(socket_dir, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    events = [
        (17, 100, "a", "target"),
        (103, 101, "b", "target"),
        (2048, 102, "c", "target"),
        (6, 103, "d", "target"),
        (71, 104, "e", "target"),
        (900, 105, "f", "target"),
    ]
    config = ClientConfig(mode="exact", backend="dflash", socket_dir=str(socket_dir), timeout=0.2)
    items = list(
        ParallelHueClient(config, opener=_exact_event_opener(socket_dir, events)).stream()
    )

    assert [item.step_id for item in items] == [event[0] for event in events]
    assert [item.color for item in items] == [
        PALETTE[index % len(PALETTE)] for index in range(len(events))
    ]
    assert [item.raw_text for item in items] == [event[2] for event in events]
    assert all(f"\x1b[38;5;{item.color}m" in item.text for item in items)


def test_exact_fragments_of_one_step_retain_their_assigned_color(socket_dir, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    events = [
        (900, 200, "a", "accepted_draft"),
        (900, 201, "b", "bonus"),
        (11, 202, "c", "target"),
    ]
    config = ClientConfig(mode="exact", backend="dflash", socket_dir=str(socket_dir), timeout=0.2)
    items = list(
        ParallelHueClient(config, opener=_exact_event_opener(socket_dir, events)).stream()
    )

    assert [item.color for item in items] == [PALETTE[0], PALETTE[0], PALETTE[1]]
    assert [item.roles for item in items] == [
        ("accepted_draft",), ("bonus",), ("target",),
    ]


def test_exact_fails_closed_on_missing_provenance_and_releases_socket(socket_dir):
    def opener(request, timeout):
        return FakeResponse([{"choices": [{"delta": {"content": "hello", "token_ids": [17]}}]}])
    config = ClientConfig(prompt="hi", mode="exact", socket_dir=str(socket_dir), timeout=0.02)
    with pytest.raises(ExactTelemetryError):
        list(ParallelHueClient(config, opener=opener).stream())
    assert not list(socket_dir.iterdir())


def test_run_scoped_socket_does_not_accept_other_run(socket_dir):
    with UnixTelemetryReceiver("a" * 32, str(socket_dir)) as receiver:
        own = ProvenanceFrame(2, "a" * 32, "ph1_" + "a" * 32 + "_0", 0, 0, (3,), ("target",), False)
        other = ProvenanceFrame(2, "b" * 32, "ph1_" + "b" * 32 + "_0", 0, 0, (4,), ("target",), False)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.sendto(encode_provenance(other), receiver.path)
            sender.sendto(encode_provenance(own), receiver.path)
        events = receiver.wait_for(own.request_id, 1)
        assert events == [own]
        assert receiver.drain(other.request_id) == []


def test_reasoning_and_content_delta_fields_render_in_order():
    def opener(request, timeout):
        return FakeResponse([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning": "think"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {"content": "content", "reasoning_content": "reasoning-content", "reasoning": "reasoning"}}]},
        ])
    items = list(ParallelHueClient(ClientConfig(prompt="hi", mode="chunk"), opener=opener).stream())
    assert [item.raw_text for item in items] == ["think", "answer", "reasoningreasoning-contentcontent"]


def test_http_error_is_terminal_sanitized():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 500, "bad", {}, io.BytesIO(b"\x1b]2;owned\a"))
    with pytest.raises(ClientError) as caught:
        list(ParallelHueClient(ClientConfig(prompt="hi", mode="chunk"), opener=opener).stream())
    assert "\x1b" not in str(caught.value)


@pytest.mark.parametrize("token_id", [True, 1.5, "17", -1])
def test_chunk_token_ids_reject_non_exact_nonnegative_ints(token_id):
    def opener(request, timeout):
        return FakeResponse([{"choices": [{"delta": {"content": "hello", "token_ids": [token_id]}}]}])
    items = list(ParallelHueClient(ClientConfig(prompt="hi", mode="chunk"), opener=opener).stream())
    assert items[0].raw_text == "hello" and items[0].token_ids == ()


def test_stream_many_preserves_final_chunk_and_treats_concurrency_as_maximum():
    request_prompts = []

    def opener(request, timeout):
        request_prompts.append(json.loads(request.data)["messages"][0]["content"])
        return FakeResponse([
            {"choices": [{"delta": {"content": text, "token_ids": [index]}}]}
            for index, text in enumerate(("first", "last"))
        ])

    client = ParallelHueClient(ClientConfig(mode="chunk", concurrency=3), opener=opener)
    items = list(client.stream_many(["one"]))
    assert request_prompts == ["one"]
    assert [item.raw_text for item in items] == ["first", "last"]


def test_stream_many_keeps_empty_batch_and_default_prompt_behavior():
    request_prompts = []

    def opener(request, timeout):
        request_prompts.append(json.loads(request.data)["messages"][0]["content"])
        return FakeResponse([{"choices": [{"delta": {"content": "ok"}}]}])

    client = ParallelHueClient(
        ClientConfig(prompt="configured", mode="chunk", concurrency=3),
        opener=opener,
    )
    assert list(client.stream_many([])) == []
    assert request_prompts == []
    assert [item.raw_text for item in client.stream_many()] == ["ok"]
    assert request_prompts == ["configured"]


def test_stream_many_rejects_duplicate_prompts_before_request(monkeypatch, socket_dir):
    request_prompts = []
    receiver_starts = []

    def opener(request, timeout):
        request_prompts.append(json.loads(request.data)["messages"][0]["content"])
        return FakeResponse([{"choices": [{"delta": {"content": "unexpected"}}]}])

    def fail_start(self):
        receiver_starts.append(True)
        raise AssertionError("receiver must not start")

    monkeypatch.setattr(UnixTelemetryReceiver, "start", fail_start)
    client = ParallelHueClient(
        ClientConfig(mode="auto", concurrency=3, socket_dir=str(socket_dir)),
        opener=opener,
    )
    with pytest.raises(ValueError):
        list(client.stream_many(["one", "one"]))
    assert request_prompts == []
    assert receiver_starts == []


def test_stream_many_propagates_worker_exception():
    def opener(request, timeout):
        raise RuntimeError("worker failed")
    with pytest.raises(RuntimeError):
        list(ParallelHueClient(ClientConfig(mode="chunk"), opener=opener).stream_many(["one"]))


@pytest.mark.parametrize("method_name", ["stream", "stream_many"])
def test_auto_receiver_failure_downgrades_without_claiming_roles(monkeypatch, socket_dir, method_name):
    def fail_start(self):
        raise RuntimeError("receiver unavailable")
    monkeypatch.setattr(UnixTelemetryReceiver, "start", fail_start)
    def opener(request, timeout):
        return FakeResponse([{"choices": [{"delta": {"content": "hello", "token_ids": [1]}}]}])
    client = ParallelHueClient(ClientConfig(prompt="hi", mode="auto", socket_dir=str(socket_dir)), opener=opener)
    items = list(client.stream() if method_name == "stream" else client.stream_many(["one"]))
    assert [item.raw_text for item in items] == ["hello"]
    assert all(not item.tokens for item in items)


@pytest.mark.parametrize("method_name", ["stream", "stream_many"])
def test_exact_receiver_failure_is_not_downgraded(monkeypatch, socket_dir, method_name):
    def fail_start(self):
        raise RuntimeError("receiver unavailable")
    monkeypatch.setattr(UnixTelemetryReceiver, "start", fail_start)
    client = ParallelHueClient(ClientConfig(prompt="hi", mode="exact", socket_dir=str(socket_dir)))
    with pytest.raises(RuntimeError):
        list(client.stream() if method_name == "stream" else client.stream_many(["one"]))


def test_socket_directory_owner_is_checked_before_chmod(monkeypatch, socket_dir):
    uid = os.lstat(socket_dir).st_uid
    original_mode = os.stat(socket_dir).st_mode
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    with pytest.raises(ClientError):
        UnixTelemetryReceiver("a" * 32, str(socket_dir)).start()
    assert os.stat(socket_dir).st_mode == original_mode


@pytest.mark.parametrize("mode,finish_text", [("exact", True), ("exact", False), ("auto", False)])
def test_parser_delayed_text_is_buffered_but_never_assumed(socket_dir, mode, finish_text):
    def opener(request, timeout):
        request_id = json.loads(request.data)["request_id"]
        run = request_id.split("_")[1]
        provenance = ProvenanceFrame(
            2, run, request_id, 0, 0, (366, 379),
            ("accepted_draft", "target"), True,
        )
        text = TextFrame(
            2, run, request_id, 0, 0, (366, 379), " < y",
            (TraceSpan(0, 2, 0, 1), TraceSpan(2, 4, 1, 2)), 8, 0, True,
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.sendto(encode_provenance(provenance), str(socket_dir / f"{run}.sock"))
            sender.sendto(encode_text(text), str(socket_dir / f"{run}.sock"))
        # GLM's parser can hold '<' while already returning its token ID.
        payloads = [{"choices": [{"delta": {"content": " "}, "token_ids": [366]}]}]
        if finish_text:
            payloads.append({"choices": [{
                "delta": {"content": "< y"}, "token_ids": [379], "finish_reason": "length",
            }]})
        return FakeResponse(payloads)

    config = ClientConfig(mode=mode, backend="dflash", socket_dir=str(socket_dir), timeout=0.2)
    client = ParallelHueClient(config, opener=opener)
    if not finish_text and mode == "exact":
        with pytest.raises(ExactTelemetryError):
            list(client.stream("hi"))
        return
    items = list(client.stream("hi"))
    assert "".join(item.raw_text for item in items) == (" < y" if finish_text else " ")
    assert [role for item in items for role in item.roles] == (
        ["accepted_draft", "target"] if finish_text else []
    )
    assert all(item.mode == ("EXACT TOKEN PROVENANCE" if finish_text else "SSE CHUNK MODE") for item in items)
