import pytest

from parallelhue.protocol import StepEvent
from parallelhue.render import (
    PALETTE,
    RenderQueue,
    StepReconciler,
    colorize,
    palette_color,
    sanitize_terminal,
)

RUN_ID = "0123456789abcdef0123456789abcdef"
REQUEST_ID = f"ph1_{RUN_ID}_0"


def event(sequence, token_ids, text, *, step_id=None, finished=False):
    return StepEvent(
        schema_version=1,
        run_id=RUN_ID,
        request_id=REQUEST_ID,
        sequence=sequence,
        step_id=sequence if step_id is None else step_id,
        choice_index=0,
        token_ids=tuple(token_ids),
        text=text,
        finished=finished,
    )


def test_reconciler_coalesces_contiguous_events_and_consumes_once():
    reconciler = StepReconciler()
    assert reconciler.push(event(0, [10], "hel"))
    assert reconciler.push(event(1, [11], "l"))
    result = reconciler.reconcile_chunk(REQUEST_ID, "hell", [10, 11])
    assert result is not None
    assert result.events[0].sequence == 0
    assert result.events[1].sequence == 1
    assert reconciler.reconcile_chunk(REQUEST_ID, "hell", [10, 11]) is None


def test_reconciler_fails_closed_on_gap_and_mismatch():
    gap = StepReconciler()
    assert not gap.push(event(1, [1], "x"))
    assert gap.failed(REQUEST_ID)
    mismatch = StepReconciler()
    assert mismatch.push(event(0, [1], "é"))
    assert mismatch.reconcile_chunk(REQUEST_ID, "e", [1]) is None
    assert mismatch.failed(REQUEST_ID)

    id_mismatch = StepReconciler()
    assert id_mismatch.push(event(0, [2], "same"))
    assert id_mismatch.reconcile_chunk(REQUEST_ID, "same", [3]) is None
    assert id_mismatch.failed(REQUEST_ID)


def test_reconciler_matches_split_utf8_text_by_raw_unicode_values():
    reconciler = StepReconciler()
    assert reconciler.push(event(0, [1], "caf"))
    assert reconciler.push(event(1, [2], "é", finished=True))
    result = reconciler.reconcile_chunk(REQUEST_ID, "café", [1, 2])
    assert result is not None
    assert result.text == "café"


def test_reconciler_completion_stays_false_for_nonterminal_match():
    reconciler = StepReconciler()
    assert reconciler.push(event(0, [1], "x"))
    assert reconciler.reconcile_chunk(REQUEST_ID, "x", [1]) is not None
    assert not reconciler.completed(REQUEST_ID)


def test_reconciler_completion_requires_terminal_match():
    reconciler = StepReconciler()
    assert reconciler.push(event(0, [1], "x", finished=True))
    assert not reconciler.completed(REQUEST_ID)
    assert reconciler.reconcile_chunk(REQUEST_ID, "x", [1]) is not None
    assert reconciler.completed(REQUEST_ID)


def test_reconciler_rejects_events_buffered_after_terminal():
    reconciler = StepReconciler()
    assert reconciler.push(event(0, [1], "x", finished=True))
    assert not reconciler.completed(REQUEST_ID)
    assert not reconciler.push(event(1, [2], "y"))
    assert reconciler.failed(REQUEST_ID)
    assert not reconciler.completed(REQUEST_ID)


def test_reconciler_failed_request_never_completes():
    reconciler = StepReconciler()
    assert reconciler.push(event(0, [1], "x"))
    assert reconciler.reconcile_chunk(REQUEST_ID, "mismatch", [1]) is None
    assert reconciler.failed(REQUEST_ID)
    assert not reconciler.completed(REQUEST_ID)


def test_sanitization_strips_ansi_bidi_and_controls():
    unsafe = "ok\x1b[31mRED\x1b[0m\u202eabc\x00\x7f\x9bJmore\t\n"
    assert sanitize_terminal(unsafe) == "okREDabcJmore\t\n"
    assert "\x1b" not in sanitize_terminal(unsafe)


def test_palette_cycles_and_colorize_sanitizes_model_text(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert [palette_color(i) for i in range(6)] == [*PALETTE, PALETTE[0], PALETTE[1]]
    rendered = colorize("x\x1b[2J", step_id=4)
    assert rendered == "\x1b[38;5;46mx\x1b[0m"


def test_colorize_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert colorize("x\x1b[2J", step_id=4) == "x"
    assert "\x1b" not in colorize("plain", color=196)


def test_colorize_still_colors_when_no_color_empty(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")
    rendered = colorize("hi", step_id=0)
    assert rendered == "\x1b[38;5;46mhi\x1b[0m"

def test_render_queue_overflow_is_nonblocking_and_fail_closed():
    queue = RenderQueue(maxsize=1)
    assert queue.put_nowait("first")
    assert not queue.put_nowait("second")
    assert queue.overflowed and queue.failed
    assert not queue.put_nowait("third")
    assert queue.get_nowait() == "first"
    assert queue.get_nowait() is None
    queue.close()
    assert queue.get_nowait() is None
