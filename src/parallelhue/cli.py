"""The ``parallelhue`` command-line interface and legacy patch points."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .backends import get_backend
from .client import ClientConfig, ClientError, ExactTelemetryError, ParallelHueClient
from . import metrics as _metrics
from .metrics import _counter_delta, _metrics_url as _metrics_url_impl, _parse_prometheus_counters
from .render import sanitize_terminal
from . import summary as _summary
from . import tmux_launch as _tmux_launch
from .prompts import PromptFileError, load_prompt_file

_summary_decode_panes_running = _summary._decode_panes_running


def _metrics_url(endpoint: str) -> str:
    """Compatibility wrapper for callers that patch CLI URL parsing symbols."""
    original_split, original_unsplit = _metrics.urlsplit, _metrics.urlunsplit
    _metrics.urlsplit, _metrics.urlunsplit = urlsplit, urlunsplit
    try:
        return _metrics_url_impl(endpoint)
    finally:
        _metrics.urlsplit, _metrics.urlunsplit = original_split, original_unsplit


def _fetch_metrics(endpoint: str, timeout: float) -> str | None:
    """Compatibility wrapper for callers that patch CLI metrics symbols."""
    original_url, original_request, original_open = _metrics._metrics_url, _metrics.Request, _metrics.urlopen
    _metrics._metrics_url, _metrics.Request, _metrics.urlopen = _metrics_url, Request, urlopen
    try:
        return _metrics._fetch_metrics(endpoint, timeout)
    finally:
        _metrics._metrics_url, _metrics.Request, _metrics.urlopen = original_url, original_request, original_open


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parallelhue", description="Color local OpenAI-compatible streams truthfully.")
    parser.add_argument("prompt_arg", nargs="?", help="prompt (or use --prompt)")
    parser.add_argument("--endpoint", default=os.environ.get("PARALLELHUE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("PARALLELHUE_MODEL", ""))
    parser.add_argument("--backend", choices=("auto", "generic", "mtp", "dspark"), default=os.environ.get("PARALLELHUE_BACKEND", "auto"), help="metrics backend profile (default: PARALLELHUE_BACKEND or auto)")
    parser.add_argument("--prompt", dest="prompt", default=os.environ.get("PARALLELHUE_PROMPT", ""))
    parser.add_argument("--prompt-file", default=os.environ.get("PARALLELHUE_PROMPT_FILE"), help="JSON array of prompts, one per stream (overrides --prompt)")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("PARALLELHUE_MAX_TOKENS", "128")))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("PARALLELHUE_CONCURRENCY", "1")))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("PARALLELHUE_API_KEY"))
    parser.add_argument("--mode", choices=("exact", "auto", "chunk"), default=os.environ.get("PARALLELHUE_MODE", "auto"))
    parser.add_argument("--socket-dir", default=os.environ.get("PARALLELHUE_SOCKET_DIR"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--tmux", action="store_true", help="launch one worker per tmux pane when tmux is available")
    parser.add_argument("--no-attach", action="store_true", help="with --tmux, create session but do not attach (scripts/CI)")
    parser.add_argument("--no-tmux", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--summary-follow", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--session", help=argparse.SUPPRESS)
    parser.add_argument("--summary-start-epoch", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--summary-timeout", type=float, default=3600.0, help=argparse.SUPPRESS)
    return parser


def run_summary(args: argparse.Namespace) -> int:
    """Compatibility wrapper: honour helpers monkeypatched on ``cli``."""
    _summary._fetch_metrics = _fetch_metrics
    _summary._parse_prometheus_counters = _parse_prometheus_counters
    _summary._counter_delta = _counter_delta
    decode_helper = _decode_panes_running
    _summary._decode_panes_running = (
        _summary_decode_panes_running if decode_helper is _DEFAULT_DECODE_PANES_RUNNING else decode_helper
    )
    _summary.get_backend = get_backend
    return _summary.run_summary(args)


def _decode_panes_running(tmux: str, session: str) -> bool:
    """Compatibility re-export for the extracted summary helper."""
    return _summary_decode_panes_running(tmux, session)


_DEFAULT_DECODE_PANES_RUNNING = _decode_panes_running



def launch_tmux(args: argparse.Namespace, argv0: str | None = None) -> int:
    """Compatibility wrapper: honour metrics helpers monkeypatched on ``cli``."""
    _tmux_launch._fetch_metrics = _fetch_metrics
    return _tmux_launch.launch_tmux(args, argv0)


def run_worker(args: argparse.Namespace) -> int:
    worker_index = 0 if args.worker_index is None else int(args.worker_index)
    stream_prompts: list[str] | None = None
    if args.prompt_file:
        try:
            file_prompts = load_prompt_file(args.prompt_file)
        except PromptFileError as exc:
            print(f"parallelhue: {exc}", file=sys.stderr)
            raise SystemExit(64) from exc
        if args.concurrency > 1:
            stream_prompts = [file_prompts[i % len(file_prompts)] for i in range(args.concurrency)]
            prompt = stream_prompts[0]
        else:
            # One pane per stream (tmux path): pick by worker index so each
            # pane runs a distinct prompt, cycling when there are more panes
            # than prompts.
            prompt = file_prompts[worker_index % len(file_prompts)]
    else:
        prompt = args.prompt or args.prompt_arg
        if not prompt:
            raise SystemExit("parallelhue: a prompt is required (use --prompt or a positional prompt)")
        if args.concurrency > 1:
            stream_prompts = [prompt] * args.concurrency
    config = ClientConfig(endpoint=args.endpoint, model=args.model, prompt=prompt, max_tokens=args.max_tokens,
                          concurrency=args.concurrency, api_key=args.api_key, mode=args.mode,
                          backend=args.backend, socket_dir=args.socket_dir, timeout=args.timeout)
    client = ParallelHueClient(config)
    printed_label: str | None = None
    if args.concurrency == 1 and args.worker_index is not None:
        total = os.environ.get("PARALLELHUE_TOTAL")
        print(f"[{worker_index + 1}/{total}]" if total and total.isdigit() else f"[{worker_index + 1}]", flush=True)
    delay_raw = os.environ.get("PARALLELHUE_START_DELAY", "").strip()
    if delay_raw:
        try: delay = max(0.0, float(delay_raw))
        except ValueError: delay = 0.0
        if delay > 0:
            print(f"[waiting {delay:.0f}s — attach now]", flush=True)
            print("generation starts from 0 after countdown; stops at max_tokens", flush=True)
            end = time.time() + delay
            while (left := end - time.time()) > 0:
                print(f"\rstarting in {left:4.1f}s   ", end="", flush=True)
                time.sleep(0.1)
            print("\n[START]", flush=True)
    streams = client.stream_many(stream_prompts) if stream_prompts is not None else client.stream(prompt, stream_index=worker_index)
    for item in streams:
        if item.mode != printed_label:
            print(f"[{item.mode}]", file=sys.stderr)
            printed_label = item.mode
        print(item.text, end="", flush=True)
    print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summary_follow:
        return run_summary(args)
    if args.tmux and not args.no_tmux:
        if shutil.which("tmux"):
            return launch_tmux(args, sys.argv[0])
        print("parallelhue: tmux unavailable; using single-terminal worker path", file=sys.stderr)
    try:
        return run_worker(args)
    except ClientError as exc:
        print(f"parallelhue: {sanitize_terminal(str(exc))}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"parallelhue: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
