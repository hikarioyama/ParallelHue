"""Tmux decode-summary pane implementation."""
from __future__ import annotations

import argparse
import os
import select
import shutil
import subprocess
import sys
import termios
import time
import tty

def stream_rate_stats(
    generation: int | None,
    makespan: float,
    concurrency: int,
) -> tuple[float | None, float | None, float | None]:
    """Return (aggregate_tok_s, mean_stream_tok_s, mean_tokens).

    aggregate_tok_s combines all streams over wall makespan; mean_stream_tok_s
    is that aggregate average divided across the concurrent streams (the
    per-stream mean). mean_tokens is the average token count per stream.
    A value is None when it is not computable: aggregate requires generation
    and makespan > 0; mean_stream_tok_s additionally requires concurrency >= 1;
    mean_tokens requires generation and concurrency >= 1.
    """
    concurrency = max(1, int(concurrency))
    rate = generation / makespan if generation is not None and makespan > 0 else None
    mean_rate = rate / concurrency if rate is not None and concurrency >= 1 else None
    mean_tokens = generation / concurrency if generation is not None else None
    return rate, mean_rate, mean_tokens


def _decode_panes_running(tmux: str, session: str) -> bool:
    """remain-on-exit preserves panes, so inspect their processes/dead state."""
    result = subprocess.run(
        [tmux, "list-panes", "-t", f"{session}:decode", "-F", "#{pane_pid}\t#{pane_current_command}\t#{pane_dead}"],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    for line in result.stdout.splitlines() if result.returncode == 0 else ():
        fields = line.split("\t")
        if len(fields) != 3 or fields[2] == "1" or not fields[1]:
            continue
        try:
            os.kill(int(fields[0]), 0)
            return True
        except (ValueError, OSError):
            continue
    return False


def run_summary(args: argparse.Namespace) -> int:
    if not args.session:
        raise SystemExit("parallelhue: --summary-follow requires --session")
    tmux = shutil.which("tmux")
    if not tmux:
        return 1
    before_text = None
    snapshot_path = os.environ.get("PARALLELHUE_SUMMARY_BEFORE_METRICS_FILE")
    baseline_available = False
    if snapshot_path:
        try:
            with open(snapshot_path, encoding="utf-8") as snapshot:
                before_text = snapshot.read()
            baseline_available = True
        except OSError:
            pass
        finally:
            try:
                os.unlink(snapshot_path)
            except OSError:
                pass
    before = _parse_prometheus_counters(before_text)
    endpoint = args.endpoint
    endpoint_path = os.environ.get("PARALLELHUE_SUMMARY_ENDPOINT_FILE")
    if endpoint_path:
        try:
            with open(endpoint_path, encoding="utf-8") as endpoint_file:
                endpoint = endpoint_file.read().strip()
        except OSError:
            endpoint = args.endpoint
        finally:
            try:
                os.unlink(endpoint_path)
            except OSError:
                pass
    started = args.summary_start_epoch if args.summary_start_epoch is not None else time.time()
    deadline = time.monotonic() + max(1.0, args.summary_timeout)
    timed_out = False
    while _decode_panes_running(tmux, args.session):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.25)
    after = _parse_prometheus_counters(_fetch_metrics(endpoint, args.timeout))
    makespan = max(0.0, time.time() - started)
    profile = get_backend(args.backend, args.model)
    generation = _counter_delta(before, after, *profile.generation_counters) if baseline_available else None
    rate, mean_rate, mean_tokens = stream_rate_stats(generation, makespan, args.concurrency)
    concurrency = max(1, int(args.concurrency))
    mean_text = "unavailable" if mean_tokens is None else f"{mean_tokens:.1f}"
    mean_rate_text = "unavailable" if mean_rate is None else f"{mean_rate:.2f}"
    print("\nParallelHue summary", flush=True)
    if timed_out:
        print("decode wait: timed out; reporting current metrics", flush=True)
    print(f"aggregate generation tok/s: {rate:.2f}" if rate is not None else "aggregate generation tok/s: unavailable", flush=True)
    print(f"mean generation tok/s: {mean_rate_text}", flush=True)
    print(f"mean generation/completion tokens: {mean_text}", flush=True)
    print(f"concurrency: {concurrency}", flush=True)
    print(f"wall makespan: {makespan:.2f}s", flush=True)
    print(f"backend profile: {profile.name}", flush=True)
    for line in profile.extra_summary_lines(before, after, makespan):
        print(line, flush=True)
    accepted = _counter_delta(before, after, *profile.accepted_counters) if baseline_available else None
    draft = _counter_delta(before, after, *profile.draft_counters) if baseline_available else None
    drafts = _counter_delta(before, after, *profile.drafts_counters) if baseline_available else None
    if accepted is not None or draft is not None or drafts is not None:
        accept_text = "unavailable" if accepted is None else f"{accepted:,.0f}"
        draft_text = "unavailable" if draft is None else f"{draft:,.0f}"
        drafts_text = "unavailable" if drafts is None else f"{drafts:,.0f}"
        accept_rate = accepted / draft * 100 if accepted is not None and draft is not None and draft > 0 else None
        print(f"spec accepted tokens: {accept_text}", flush=True)
        print(f"spec draft tokens: {draft_text}", flush=True)
        print(f"spec drafts: {drafts_text}", flush=True)
        print(f"spec accept rate: {accept_rate:.2f}%" if accept_rate is not None else "spec accept rate: unavailable", flush=True)
    print("Press Esc in this summary pane to close the tmux session.", flush=True)
    subprocess.run([tmux, "select-window", "-t", f"{args.session}:summary"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stdin_fd = sys.stdin.fileno()
    terminal_state = None
    try:
        if sys.stdin.isatty():
            terminal_state = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)
        while True:
            key = os.read(stdin_fd, 1)
            if not key:
                break
            if key != b"\x1b":
                continue
            if select.select([stdin_fd], [], [], 0.03)[0]:
                select.select([stdin_fd], [], [], 0)
                os.read(stdin_fd, 1)
                while select.select([stdin_fd], [], [], 0.01)[0]:
                    os.read(stdin_fd, 1)
                continue
            subprocess.run([tmux, "kill-session", "-t", args.session], check=False)
            break
    except (KeyboardInterrupt, OSError, termios.error):
        pass
    finally:
        if terminal_state is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, terminal_state)
    return 0
