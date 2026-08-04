import json

import pytest

from parallelhue.protocol import (
    ProtocolError,
    StepEvent,
    decode_event,
    encode_event,
    parse_request_id,
)


RUN_ID = "0123456789abcdef0123456789abcdef"
REQUEST_ID = f"ph1_{RUN_ID}_3"


def event(**overrides):
    values = dict(
        schema_version=1,
        run_id=RUN_ID,
        request_id=REQUEST_ID,
        sequence=0,
        step_id=7,
        choice_index=0,
        token_ids=(11, 12),
        text="ab",
        finished=False,
    )
    values.update(overrides)
    return StepEvent(**values)


def test_event_is_immutable_and_codec_is_canonical():
    value = event()
    assert value.token_ids == (11, 12)
    with pytest.raises((AttributeError, TypeError)):
        value.text = "changed"
    encoded = encode_event(value)
    assert encoded == encode_event(value)
    assert decode_event(encoded) == value
    assert json.loads(encoded) == value.to_dict()


def test_request_id_and_schema_validation_are_strict():
    assert parse_request_id(REQUEST_ID) == (RUN_ID, 3)
    for bad in ("ph1_ABC_3", f"ph1_{RUN_ID}_", f"ph2_{RUN_ID}_3"):
        with pytest.raises(ProtocolError):
            parse_request_id(bad)
    with pytest.raises(ProtocolError):
        event(request_id=f"ph1_{RUN_ID}_4", run_id=RUN_ID[:-1] + "e")
    with pytest.raises(ProtocolError):
        decode_event(encode_event(event()).replace(b'"finished":false', b'"finished":false,"extra":1'))


def test_codec_rejects_invalid_utf8_and_non_integer_ids():
    with pytest.raises(ProtocolError):
        decode_event(b"\xff")
    with pytest.raises(ProtocolError):
        event(token_ids=(True,))
