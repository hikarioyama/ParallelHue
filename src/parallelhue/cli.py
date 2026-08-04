"""The ``parallelhue`` command-line interface."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Sequence

from .client import ClientConfig, ClientError, ExactTelemetryError, ParallelHueClient
from .render import sanitize_terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parallelhue", description="Color local OpenAI-compatible streams truthfully.")
    parser.add_argument("prompt_arg", nargs="?", help="prompt (or use --prompt)")
    parser.add_argument("--endpoint", default=os.environ.get("PARALLELHUE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("PARALLELHUE_MODEL", ""))
    parser.add_argument("--prompt", dest="prompt", default=os.environ.get("PARALLELHUE_PROMPT", ""))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("PARALLELHUE_MAX_TOKENS", "128")))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("PARALLELHUE_CONCURRENCY", "1")))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("PARALLELHUE_API_KEY"))
    parser.add_argument("--mode", choices=("exact", "auto", "chunk"), default=os.environ.get("PARALLELHUE_MODE", "auto"))
    parser.add_argument("--socket-dir", default=os.environ.get("PARALLELHUE_SOCKET_DIR"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--tmux", action="store_true", help="launch one worker per tmux pane when tmux is available")
    parser.add_argument("--no-tmux", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def launch_tmux(args: argparse.Namespace, argv0: str | None = None) -> int:
    """Launch bounded workers in a practical tmux session, without GPU assumptions."""
    tmux = shutil.which("tmux")
    if not tmux:
        return 1
    session = "parallelhue-" + uuid.uuid4().hex[:8]
    base = [
        sys.executable,
        "-m",
        "parallelhue",
        "--no-tmux",
        "--mode",
        args.mode,
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
        "--concurrency",
        "1",
        "--timeout",
        str(args.timeout),
    ]
    if args.socket_dir:
        base += ["--socket-dir", args.socket_dir]
    prompt = args.prompt or args.prompt_arg
    if prompt:
        base += ["--prompt", prompt]

    def command(worker_index: int) -> str:
        return shlex.join([*base, "--worker-index", str(worker_index)])

    runtime_dir: str | None = None
    wrapper_paths: list[str] = []
    if args.api_key:
        runtime_dir = tempfile.mkdtemp(prefix="parallelhue-")
        try:
            os.chmod(runtime_dir, 0o700)
            if hasattr(os, "getuid") and os.stat(runtime_dir).st_uid != os.getuid():
                raise PermissionError("parallelhue: private runtime directory is not user-owned")

            for worker_index in range(max(1, args.concurrency)):
                fd, wrapper_path = tempfile.mkstemp(
                    prefix=f"worker-{worker_index}-",
                    suffix=".sh",
                    dir=runtime_dir,
                )
                try:
                    os.fchmod(fd, 0o700)
                    script = "\n".join(
                        [
                            "#!/bin/sh",
                            "set -eu",
                            "wrapper=$0",
                            """trap 'rm -f -- "$wrapper"' EXIT HUP INT TERM""",
                            'rm -f -- "$wrapper"',
                            f"export PARALLELHUE_API_KEY={shlex.quote(args.api_key)}",
                            f"export OPENAI_API_KEY={shlex.quote(args.api_key)}",
                            f"exec {command(worker_index)}",
                            "",
                        ]
                    )
                    with os.fdopen(fd, "w", encoding="utf-8") as wrapper:
                        fd = -1
                        wrapper.write(script)
                    wrapper_paths.append(wrapper_path)
                except BaseException:
                    if fd >= 0:
                        os.close(fd)
                    try:
                        os.unlink(wrapper_path)
                    except FileNotFoundError:
                        pass
                    raise
        except BaseException:
            shutil.rmtree(runtime_dir, ignore_errors=True)
            raise

    def run_tmux(command_args: list[str], check: bool) -> None:
        result = subprocess.run(command_args, check=check)
        if getattr(result, "returncode", 0) and check:
            raise subprocess.CalledProcessError(result.returncode, command_args)

    def pane_command(worker_index: int) -> str:
        if wrapper_paths:
            return wrapper_paths[worker_index]
        return command(worker_index)

    session_created = False
    try:
        run_tmux([tmux, "new-session", "-d", "-s", session, pane_command(0)], True)
        session_created = True
        for worker_index in range(1, args.concurrency):
            run_tmux([tmux, "split-window", "-t", session, pane_command(worker_index)], True)
        run_tmux([tmux, "select-layout", "-t", session, "tiled"], True)
    except BaseException:
        if session_created:
            try:
                subprocess.run([tmux, "kill-session", "-t", session], check=False)
            except BaseException:
                pass
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        raise
    print(f"ParallelHue tmux session: {session}", file=sys.stderr)
    return 0


def run_worker(args: argparse.Namespace) -> int:
    prompt = args.prompt or args.prompt_arg
    if not prompt:
        raise SystemExit("parallelhue: a prompt is required (use --prompt or a positional prompt)")
    config = ClientConfig(
        endpoint=args.endpoint,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        api_key=args.api_key,
        mode=args.mode,
        socket_dir=args.socket_dir,
        timeout=args.timeout,
    )
    client = ParallelHueClient(config)
    printed_label: str | None = None
    streams = client.stream_many([prompt] * args.concurrency) if args.concurrency > 1 else client.stream(prompt, stream_index=args.worker_index or 0)
    for item in streams:
        if item.mode != printed_label:
            print(f"[{item.mode}]", file=sys.stderr)
            printed_label = item.mode
        print(item.text, end="", flush=True)
    print()
    return 0

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
