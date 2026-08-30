"""Tmux worker-pane launcher."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.parse import urlsplit

from .metrics import _fetch_metrics


def launch_tmux(args: argparse.Namespace, argv0: str | None = None) -> int:
    """Launch one tmux pane per stream and a summary pane."""
    tmux = shutil.which("tmux")
    if not tmux:
        print("parallelhue: tmux unavailable", file=sys.stderr)
        return 1
    n = max(1, int(args.concurrency))
    session = "parallelhue-" + uuid.uuid4().hex[:8]
    summary_timeout = getattr(args, "summary_timeout", 3600.0)
    before_metrics = _fetch_metrics(args.endpoint, args.timeout)
    base = [sys.executable, "-m", "parallelhue", "--no-tmux", "--mode", args.mode,
            "--backend", args.backend, "--endpoint", args.endpoint, "--model", args.model,
            "--max-tokens", str(args.max_tokens), "--concurrency", "1", "--timeout", str(args.timeout)]
    if getattr(args, "prompt_file", None):
        # Every pane selects its own prompt from the file via --worker-index.
        base += ["--prompt-file", args.prompt_file]
    else:
        prompt = args.prompt or args.prompt_arg
        if prompt:
            base += ["--prompt", prompt]

    def command(worker_index: int) -> str:
        env_pairs = [f"PARALLELHUE_TOTAL={n}", f"PARALLELHUE_INDEX={worker_index}"]
        for name in ("NO_COLOR", "PARALLELHUE_START_DELAY", "PARALLELHUE_TEMPERATURE",
                     "PARALLELHUE_FREQUENCY_PENALTY", "PARALLELHUE_PRESENCE_PENALTY",
                     "PARALLELHUE_REPEAT_PENALTY", "PARALLELHUE_FORCE_FULL_LENGTH"):
            value = os.environ.get(name)
            if value:
                env_pairs.append(f"{name}={value}")
        return shlex.join(["env", *env_pairs, *base, "--worker-index", str(worker_index)])

    runtime_dir: str | None = None
    wrapper_paths: list[str] = []
    before_metrics_path: str | None = None
    endpoint_path: str | None = None
    try:
        parsed_endpoint = urlsplit(args.endpoint)
        endpoint_has_credentials = parsed_endpoint.username is not None or parsed_endpoint.password is not None
    except ValueError:
        endpoint_has_credentials = False
    if args.api_key or before_metrics is not None or endpoint_has_credentials:
        runtime_dir = tempfile.mkdtemp(prefix="parallelhue-")
        try:
            os.chmod(runtime_dir, 0o700)
            if hasattr(os, "getuid") and os.stat(runtime_dir).st_uid != os.getuid():
                raise PermissionError("parallelhue: private runtime directory is not user-owned")

            def write_private(prefix: str, content: str, suffix: str = "") -> str:
                fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=runtime_dir)
                try:
                    os.fchmod(fd, 0o700 if suffix else 0o600)
                    with os.fdopen(fd, "w", encoding="utf-8") as output:
                        output.write(content)
                    fd = -1
                    return path
                except BaseException:
                    if fd >= 0:
                        os.close(fd)
                    try: os.unlink(path)
                    except FileNotFoundError: pass
                    raise

            if before_metrics is not None:
                before_metrics_path = write_private("metrics-before-", before_metrics)
            if endpoint_has_credentials:
                endpoint_path = write_private("endpoint-", args.endpoint)
            for worker_index in range(n) if args.api_key else ():
                script = "\n".join(["#!/bin/sh", "set -eu", "wrapper=$0",
                    "trap 'rm -f -- \"$wrapper\"' EXIT HUP INT TERM", 'rm -f -- "$wrapper"',
                    f"export PARALLELHUE_API_KEY={shlex.quote(args.api_key)}",
                    f"export OPENAI_API_KEY={shlex.quote(args.api_key)}",
                    f"exec {command(worker_index)}", ""])
                wrapper_paths.append(write_private(f"worker-{worker_index}-", script, ".sh"))
        except BaseException:
            shutil.rmtree(runtime_dir, ignore_errors=True)
            raise

    def run_tmux(command_args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(command_args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command_args, result.stdout, result.stderr)
        return result

    def pane_command(worker_index: int) -> str:
        return wrapper_paths[worker_index] if wrapper_paths else command(worker_index)

    cols, rows = max(200, 40 * min(n, 8)), max(60, 12 * ((n + 3) // 4))
    decode, session_created = f"{session}:decode", False
    try:
        run_tmux([tmux, "new-session", "-d", "-s", session, "-x", str(cols), "-y", str(rows), "-n", "decode", "sleep 2147483647"])
        session_created = True
        run_tmux([tmux, "set-window-option", "-t", decode, "remain-on-exit"])
        summary_started = time.time()
        run_tmux([tmux, "respawn-pane", "-k", "-t", decode, pane_command(0)])
        summary_command = shlex.join([sys.executable, "-m", "parallelhue", "--summary-follow", "--session", session,
            "--summary-start-epoch", str(summary_started), "--summary-timeout", str(summary_timeout),
            "--endpoint", "" if endpoint_path else args.endpoint, "--timeout", str(args.timeout),
            "--concurrency", str(n), "--backend", args.backend, "--model", args.model])
        if before_metrics_path:
            summary_command = f"PARALLELHUE_SUMMARY_BEFORE_METRICS_FILE={shlex.quote(before_metrics_path)} {summary_command}"
        if endpoint_path:
            summary_command = f"PARALLELHUE_SUMMARY_ENDPOINT_FILE={shlex.quote(endpoint_path)} {summary_command}"
        for worker_index in range(1, n):
            run_tmux([tmux, "split-window", "-t", decode, "-h", pane_command(worker_index)])
            run_tmux([tmux, "select-layout", "-t", decode, "tiled"], check=False)
        run_tmux([tmux, "select-layout", "-t", decode, "tiled"], check=False)
        run_tmux([tmux, "new-window", "-t", session, "-n", "summary", summary_command])
        run_tmux([tmux, "set-window-option", "-t", f"{session}:summary", "remain-on-exit", "on"], check=False)
        run_tmux([tmux, "select-window", "-t", decode], check=False)
        run_tmux([tmux, "set-option", "-t", session, "mouse", "on"], check=False)
    except BaseException as exc:
        print(f"parallelhue: launch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if session_created:
            try: subprocess.run([tmux, "kill-session", "-t", session], check=False)
            except BaseException: pass
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        return 1
    print(f"ParallelHue tmux session: {session} panes={n}", file=sys.stderr)
    print(f"attach with: tmux attach -t {shlex.quote(session)}", file=sys.stderr)
    if sys.stdout.isatty() and not getattr(args, "no_attach", False):
        subprocess.call([tmux, "attach-session", "-t", session])
        while subprocess.run([tmux, "has-session", "-t", session], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            time.sleep(0.2)
    return 0
