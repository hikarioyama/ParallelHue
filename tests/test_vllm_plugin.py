import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from parallelhue.protocol import decode_frame
from parallelhue.vllm_plugin import VllmExactPlugin, VllmPluginError, _TraceState, install, validate_socket

RUN = "a" * 32
REQUEST = f"ph1_{RUN}_0"


@pytest.fixture
def plugin(socket_dir):
    value = VllmExactPlugin(socket_dir)
    yield value
    value.close()


def scheduler(method="dflash"):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method=method)),
        num_sampled_tokens_per_step=1, adaptive_mtp_controller=None,
    )


def test_full_acceptance_and_rejection_have_different_target_roles(plugin):
    s = scheduler()
    assert plugin._classify_roles(s, REQUEST, [10, 11], [10, 11, 99]) == ("accepted_draft", "accepted_draft", "bonus")
    assert plugin._classify_roles(s, REQUEST, [10, 11], [10, 99]) == ("accepted_draft", "target")
    assert plugin._classify_roles(s, REQUEST, [10, 11], [99]) == ("target",)
    assert plugin._classify_roles(s, REQUEST, None, [99]) == ("target",)


def test_post_stop_truncation_cannot_relabel_an_accepted_token_as_bonus(plugin, socket_dir):
    os.chmod(socket_dir, 0o700)
    path = socket_dir / f"{RUN}.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
        receiver.bind(str(path))
        os.chmod(path, 0o600)
        receiver.settimeout(1)
        s = scheduler()
        internal = f"chatcmpl-{REQUEST}-0123abcd"
        s.requests = {internal: SimpleNamespace(request_id=internal, _output_token_ids=[])}
        scheduled = SimpleNamespace(scheduled_spec_decode_tokens={internal: [10, 11]})
        model = SimpleNamespace(req_ids=[internal], sampled_token_ids=[[10, 11, 99]])
        def native_update(s, scheduled, model):
            s.requests[internal]._output_token_ids.extend([10, 11])
            return {0: SimpleNamespace(outputs=[SimpleNamespace(request_id=internal, new_token_ids=[10, 11], finished=True)])}
        result = plugin._scheduler_wrapper(native_update, s, scheduled, model)
        frame = decode_frame(receiver.recv(8192))
        assert frame.token_ids == (10, 11)
        assert frame.roles == ("accepted_draft", "accepted_draft")
        assert frame.finished and frame.token_offset == 0
        assert result[0].outputs[0].new_token_ids == [10, 11]

def test_unknown_or_inconsistent_sampler_output_is_not_presented_as_exact(plugin):
    s = scheduler()
    assert plugin._classify_roles(s, REQUEST, [10, 11], [12, 99]) is None
    assert plugin._classify_roles(s, REQUEST, [10, -2, 11], [10, 99]) is None
    assert plugin._classify_roles(s, REQUEST, [10], [10, 11, 99]) is None
    assert plugin._classify_roles(scheduler("unknown"), REQUEST, [10], [10, 99]) is None
    s.num_sampled_tokens_per_step = 0
    assert plugin._classify_roles(s, REQUEST, [10], [10]) is None


def test_async_placeholders_and_invalid_slots_do_not_invent_bonus_tokens(plugin):
    s = scheduler()
    assert plugin._classify_roles(s, REQUEST, [-1, -1], [10, 11, 99]) == ("accepted_draft", "accepted_draft", "bonus")
    assert plugin._classify_roles(s, REQUEST, [-1, -1], [10, 99]) == ("accepted_draft", "target")
    assert plugin._classify_roles(s, REQUEST, [10, 11, -1], [10, 11, 99]) == ("accepted_draft", "accepted_draft", "target")


class Decoder:
    def __init__(self):
        self.token_ids = []
    def num_output_tokens(self):
        return len(self.token_ids)


def test_native_deferred_unicode_tracks_all_contributing_tokens():
    decoder = Decoder()
    trace = _TraceState(decoder)
    decoder.token_ids.append(1)
    trace.record_piece(1, "")
    assert trace.trace_for(0, (1,), "") == ()
    decoder.token_ids.append(2)
    trace.record_piece(2, "漢")
    assert trace.trace_for(1, (2,), "漢") == ((0, 3, 0, 2),)
    assert not trace.pieces


def test_withheld_text_can_be_emitted_across_output_boundaries():
    decoder = Decoder()
    trace = _TraceState(decoder)
    decoder.token_ids.append(1)
    trace.record_piece(1, "hello")
    assert trace.trace_for(0, (1,), "hel") == ((0, 3, 0, 1),)
    decoder.token_ids.append(2)
    trace.record_piece(2, "!")
    assert trace.trace_for(1, (2,), "lo!") == ((0, 2, 0, 1), (2, 3, 1, 2))
    assert not trace.pieces


def test_native_text_mismatch_disables_exact_trace():
    decoder = Decoder()
    trace = _TraceState(decoder)
    decoder.token_ids.append(1)
    trace.record_piece(1, "source")
    assert trace.trace_for(0, (1,), "transformed") is None
    assert trace.trace_for(0, (1,), "source") is None


def test_disabled_plugin_does_not_import_vllm_or_mutate_environment(monkeypatch):
    monkeypatch.setenv("PARALLELHUE_VLLM_EXACT", "0")
    assert install() is None


def test_unsupported_version_does_not_install_hooks(socket_dir):
    with pytest.raises(VllmPluginError):
        install(SimpleNamespace(__version__="0.25.0"), enabled=True, socket_dir=socket_dir)


def test_private_socket_validation_and_explicit_root_sender(socket_dir, monkeypatch):
    os.chmod(socket_dir, 0o700)
    path = socket_dir / f"{RUN}.sock"
    uid = os.getuid()
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
        receiver.bind(str(path))
        os.chmod(path, 0o600)
        assert validate_socket(path, socket_dir)
        assert not validate_socket(path, socket_dir, socket_uid=uid + 1)
        monkeypatch.setattr(os, "getuid", lambda: 0)
        assert validate_socket(path, socket_dir, socket_uid=uid)
        assert not validate_socket(path, socket_dir)
        os.chmod(path, 0o666)
        assert not validate_socket(path, socket_dir, socket_uid=uid)
