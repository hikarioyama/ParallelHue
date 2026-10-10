import pytest

from parallelhue.prompts import PromptSelectionError, validate_prompt_values


def test_validate_prompt_values_copies_supported_sequences_without_text_changes():
    values = (" leading and trailing ", "\u2603")

    validated = validate_prompt_values(values)

    assert validated == list(values)
    assert validated is not values


@pytest.mark.parametrize("values", ["one prompt", b"one prompt", 42, None])
def test_validate_prompt_values_rejects_non_sequence_inputs(values):
    with pytest.raises(PromptSelectionError):
        validate_prompt_values(values, source="test prompts")


@pytest.mark.parametrize("values", [(), ("",), ("valid", 1), ("valid", None), ("valid", b"bytes")])
def test_validate_prompt_values_rejects_non_string_sequence_members(values):
    with pytest.raises(PromptSelectionError):
        validate_prompt_values(values, source="test prompts")
