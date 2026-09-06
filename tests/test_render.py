import pytest

from parallelhue.protocol import ProvenanceFrame, TextFrame, TraceSpan
from parallelhue.render import RenderQueue, StepReconciler, colorize, sanitize_terminal

RUN = "a" * 32
REQUEST = f"ph1_{RUN}_0"


def provenance(sequence, offset, ids, roles, finished=False):
    return ProvenanceFrame(2, RUN, REQUEST, sequence, offset, tuple(ids), tuple(roles), finished)


def text(sequence, offset, ids, value, spans, finished=False):
    return TextFrame(2, RUN, REQUEST, sequence, offset, tuple(ids), value, tuple(TraceSpan(*s) for s in spans), sequence, 0, finished)


def test_sources_with_different_frame_counts_join_by_token_identity():
    r = StepReconciler()
    r.push(text(0, 0, [10, 11], "ab", [(0, 1, 0, 1), (1, 2, 1, 2)], True))
    r.push(provenance(0, 0, [10], ["accepted_draft"]))
    assert r.reconcile_chunk(REQUEST, "ab", [10, 11]) is None
    r.push(provenance(1, 1, [11], ["target"], True))
    result = r.reconcile_chunk(REQUEST, "ab", [10, 11])
    assert result.roles == ("accepted_draft", "target")
    assert result.text == "ab"
    assert r.completed(REQUEST)
    assert r.reconcile_chunk(REQUEST, "ab", [10, 11]) is None


def test_sse_coalescing_preserves_all_events_and_roles():
    r = StepReconciler()
    r.push(provenance(0, 0, [10, 11], ["accepted_draft", "bonus"], True))
    r.push(text(0, 0, [10], "caf", [(0, 3, 0, 1)]))
    r.push(text(1, 1, [11], "é", [(0, 2, 1, 2)], True))
    result = r.reconcile_chunk(REQUEST, "café", [10, 11])
    assert [e.text for e in result.events] == ["caf", "é"]
    assert result.roles == ("accepted_draft", "bonus")
    assert r.completed(REQUEST)


def test_deferred_unicode_keeps_prior_invisible_token_provenance():
    r = StepReconciler()
    r.push(provenance(0, 0, [1], ["accepted_draft"]))
    r.push(text(0, 0, [1], "", []))
    first = r.reconcile_chunk(REQUEST, "", [1])
    assert first.roles == ("accepted_draft",)
    r.push(provenance(1, 1, [2], ["target"], True))
    r.push(text(1, 1, [2], "漢", [(0, 3, 0, 2)], True))
    result = r.reconcile_chunk(REQUEST, "漢", [2])
    assert result.roles == ("target",)
    segment = result.events[0].segments[0]
    assert segment.text == "漢" and segment.ambiguous and segment.role is None
    assert (segment.token_offset_start, segment.token_offset_end) == (0, 2)


def test_empty_terminal_waits_for_both_sources():
    r = StepReconciler()
    r.push(provenance(0, 0, [1], ["target"]))
    r.push(text(0, 0, [1], "x", [(0, 1, 0, 1)]))
    assert r.reconcile_chunk(REQUEST, "x", [1]) is not None
    r.push(text(1, 1, [], "", [], True))
    assert r.reconcile_chunk(REQUEST, "", []) is None
    assert not r.completed(REQUEST)
    r.push(provenance(1, 1, [], [], True))
    assert r.reconcile_chunk(REQUEST, "", []) is not None
    assert r.completed(REQUEST)


def test_gap_duplicate_and_post_terminal_frames_fail_closed():
    gap = StepReconciler()
    assert not gap.push(provenance(1, 0, [1], ["target"]))
    duplicate = StepReconciler()
    p = provenance(0, 0, [1], ["target"])
    assert duplicate.push(p)
    assert not duplicate.push(p)
    terminal = StepReconciler()
    terminal.push(provenance(0, 0, [1], ["target"], True))
    assert not terminal.push(provenance(1, 1, [2], ["target"]))
    assert all(r.failed(REQUEST) and not r.completed(REQUEST) for r in (gap, duplicate, terminal))


def test_token_and_raw_text_mismatches_cannot_be_reconciled():
    for observed_id, observed_text in ((2, "é"), (1, "e")):
        r = StepReconciler()
        r.push(provenance(0, 0, [1], ["target"], True))
        r.push(text(0, 0, [1], "é", [(0, 2, 0, 1)], True))
        assert r.reconcile_chunk(REQUEST, observed_text, [observed_id]) is None
        assert r.failed(REQUEST)


def test_unresolved_or_future_trace_never_receives_a_role():
    r = StepReconciler()
    r.push(provenance(0, 0, [1], ["target"], True))
    r.push(text(0, 0, [1], "x", [(0, 1, 0, 2)], True))
    assert r.reconcile_chunk(REQUEST, "x", [1]) is None
    assert r.failed(REQUEST)


def test_sanitization_strips_ansi_bidi_and_controls():
    assert sanitize_terminal("ok\x1b[31mRED\x1b[0m\u202eabc\x00\x7f\x9bJmore\t\n") == "okREDabcJmore\t\n"


def test_colorize_respects_no_color_and_sanitizes(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert colorize("x\x1b[2J", step_id=4) == "x"


def test_render_queue_overflow_is_fail_closed():
    q = RenderQueue(maxsize=1)
    assert q.put_nowait("first")
    assert not q.put_nowait("second")
    assert q.overflowed and q.failed
    assert not q.put_nowait("third")
    assert q.get_nowait() == "first"
    q.close()
    assert q.get_nowait() is None
