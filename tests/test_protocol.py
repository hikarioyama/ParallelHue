import json

import pytest

from parallelhue.protocol import (
    ProtocolError,
    ProvenanceFrame,
    TextFrame,
    TraceSpan,
    decode_frame,
    encode_provenance,
    encode_text,
    parse_request_id,
)


RUN_ID = "0123456789abcdef0123456789abcdef"
REQUEST_ID = f"ph1_{RUN_ID}_3"


def provenance(**overrides):
    values = dict(
        schema_version=2,
        run_id=RUN_ID,
        request_id=REQUEST_ID,
        source_sequence=0,
        token_offset=0,
        token_ids=(11, 12),
        roles=("accepted_draft", "bonus"),
        finished=False,
    )
    values.update(overrides)
    return ProvenanceFrame(**values)


def text(**overrides):
    values = dict(
        schema_version=2,
        run_id=RUN_ID,
        request_id=REQUEST_ID,
        source_sequence=0,
        token_offset=0,
        token_ids=(11, 12),
        text="ab",
        trace=(TraceSpan(0, 2, 0, 2),),
        step_id=7,
        choice_index=0,
        finished=False,
    )
    values.update(overrides)
    return TextFrame(**values)


def test_two_frame_codecs_are_canonical_and_immutable():
    for frame, encoder in ((provenance(), encode_provenance), (text(), encode_text)):
        encoded = encoder(frame)
        assert encoded == encoder(frame)
        assert decode_frame(encoded) == frame
        assert json.loads(encoded) == frame.to_dict()
        with pytest.raises((AttributeError, TypeError)):
            frame.finished = True


def test_request_id_and_schema_validation_are_strict():
    assert parse_request_id(REQUEST_ID) == (RUN_ID, 3)
    for bad in ("ph1_ABC_3", f"ph1_{RUN_ID}_", f"ph2_{RUN_ID}_3"):
        with pytest.raises(ProtocolError):
            parse_request_id(bad)
    with pytest.raises(ProtocolError):
        provenance(request_id=f"ph1_{RUN_ID}_4", run_id=RUN_ID[:-1] + "e")
    payload = encode_provenance(provenance()).replace(
        b'"finished":false', b'"finished":false,"extra":1'
    )
    with pytest.raises(ProtocolError):
        decode_frame(payload)


def test_trace_must_cover_utf8_without_splitting_codepoints():
    with pytest.raises(ProtocolError):
        text(text="é", trace=(TraceSpan(0, 1, 0, 1),))
    with pytest.raises(ProtocolError):
        decode_frame(b"\xff")
    with pytest.raises(ProtocolError):
        provenance(token_ids=(True,))
    with pytest.raises(ProtocolError):
        text(text="a", trace=())
