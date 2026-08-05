"""The ``parallelhue`` command-line interface."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import select
import tempfile
import time
import termios
import tty
import uuid
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .backends import get_backend
from .client import ClientConfig, ClientError, ExactTelemetryError, ParallelHueClient
from .render import sanitize_terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parallelhue", description="Color local OpenAI-compatible streams truthfully.")
    parser.add_argument("prompt_arg", nargs="?", help="prompt (or use --prompt)")
    parser.add_argument("--endpoint", default=os.environ.get("PARALLELHUE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("PARALLELHUE_MODEL", ""))
    parser.add_argument(
        "--backend",
        choices=("auto", "generic", "mtp", "dspark"),
        default=os.environ.get("PARALLELHUE_BACKEND", "auto"),
        help="metrics backend profile (default: PARALLELHUE_BACKEND or auto)",
    )
    parser.add_argument("--prompt", dest="prompt", default=os.environ.get("PARALLELHUE_PROMPT", ""))
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


def _metrics_url(endpoint: str) -> str:
    """Convert an OpenAI endpoint, normally /v1/chat/completions, to /metrics."""
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc or parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    parts = [part for part in parsed.path.split("/") if part]
    prefix = parts[:parts.index("v1")] if "v1" in parts else parts[:-1]
    return urlunsplit((parsed.scheme, parsed.netloc, "/" + "/".join([*prefix, "metrics"]), "", ""))


def _fetch_metrics(endpoint: str, timeout: float) -> str | None:
    try:
        request = Request(_metrics_url(endpoint), headers={"Accept": "text/plain"})
        with urlopen(request, timeout=max(1.0, min(timeout, 15.0))) as response:  # noqa: S310
            return response.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        # Don't print a user-supplied URL: it can contain credentials.
        print("parallelhue: metrics unavailable", file=sys.stderr)
        return None


def _parse_prometheus_counters(text: str | None) -> dict[str, float]:
    counters: dict[str, float] = {}
    for line in (text or "").splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0].startswith("#"):
            continue
        name = fields[0].split("{", 1)[0]
        if not (name.endswith("_total") or name.endswith("_sum")):
            continue
        try:
            counters[name] = counters.get(name, 0.0) + float(fields[1])
        except ValueError:
            pass
    return counters


def _counter_delta(before: dict[str, float], after: dict[str, float], *candidates: str) -> float | None:
    """Return a non-reset delta for the first exact counter name available."""
    for name in candidates:
        if name in before and name in after and after[name] >= before[name]:
            return after[name] - before[name]
    return None


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
    # Profiles prefer exact Prometheus names; never sum per_pos series.
    generation = (
        _counter_delta(before, after, *profile.generation_counters)
        if baseline_available
        else None
    )
    rate = generation / makespan if generation is not None and makespan > 0 else None
    count = "unavailable" if generation is None else f"{generation:,.0f}"
    print("\nParallelHue summary", flush=True)
    if timed_out:
        print("decode wait: timed out; reporting current metrics", flush=True)
    print(f"aggregate generation tok/s: {rate:.2f}" if rate is not None else "aggregate generation tok/s: unavailable", flush=True)
    print(f"total generation/completion tokens: {count}", flush=True)
    print(f"concurrency: {max(1, int(args.concurrency))}", flush=True)
    print(f"wall makespan: {makespan:.2f}s", flush=True)
    print(f"backend profile: {profile.name}", flush=True)
    for line in profile.extra_summary_lines(before, after, makespan):
        print(line, flush=True)
    accepted = (
        _counter_delta(before, after, *profile.accepted_counters)
        if baseline_available
        else None
    )
    draft = (
        _counter_delta(before, after, *profile.draft_counters)
        if baseline_available
        else None
    )
    drafts = (
        _counter_delta(before, after, *profile.drafts_counters)
        if baseline_available
        else None
    )
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
    # dspark8 parity: flip the attached client onto the summary window
    # once decode finishes, so the result screen is visible without manual switch.
    subprocess.run(
        [tmux, "select-window", "-t", f"{args.session}:summary"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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
            # Arrow/function keys start with ESC followed by CSI/SS3 bytes.
            # Consume that sequence; only an unadorned ESC closes tmux.
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


def launch_tmux(args: argparse.Namespace, argv0: str | None = None) -> int:
    """Launch one tmux pane per stream, then attach like dspark8 launch().

    Flow matches the tweet demo:
    1) create a named session with a large canvas
    2) tile N worker panes (each runs concurrency=1)
    3) if stdout is a TTY, attach so the user immediately sees the split
    4) wait until the session ends
    """
    tmux = shutil.which("tmux")
    if not tmux:
        print("parallelhue: tmux unavailable", file=sys.stderr)
        return 1

    n = max(1, int(args.concurrency))
    session = "parallelhue-" + uuid.uuid4().hex[:8]
    summary_timeout = getattr(args, "summary_timeout", 3600.0)
    # This must precede worker creation, otherwise deltas include an unknown
    # portion of the run.  The snapshot never contains the API key.
    before_metrics = _fetch_metrics(args.endpoint, args.timeout)
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
        # tmux split-window does not use a shell; wrap so env exports work.
        inner = shlex.join([*base, "--worker-index", str(worker_index)])
        return shlex.join(
            [
                "env",
                f"PARALLELHUE_TOTAL={n}",
                f"PARALLELHUE_INDEX={worker_index}",
                *base,
                "--worker-index",
                str(worker_index),
            ]
        )

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

            if before_metrics is not None:
                fd, before_metrics_path = tempfile.mkstemp(prefix="metrics-before-", dir=runtime_dir)
                try:
                    os.fchmod(fd, 0o600)
                    with os.fdopen(fd, "w", encoding="utf-8") as snapshot:
                        fd = -1
                        snapshot.write(before_metrics)
                finally:
                    if fd >= 0:
                        os.close(fd)
            if endpoint_has_credentials:
                fd, endpoint_path = tempfile.mkstemp(prefix="endpoint-", dir=runtime_dir)
                try:
                    os.fchmod(fd, 0o600)
                    with os.fdopen(fd, "w", encoding="utf-8") as endpoint_file:
                        fd = -1
                        endpoint_file.write(args.endpoint)
                finally:
                    if fd >= 0:
                        os.close(fd)

            for worker_index in range(n) if args.api_key else ():
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

    def run_tmux(command_args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command_args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, command_args, result.stdout, result.stderr
            )
        return result

    def pane_command(worker_index: int) -> str:
        if wrapper_paths:
            return wrapper_paths[worker_index]
        return command(worker_index)


    # Large canvas so concurrency=16 tiled splits fit (detached default is 80x24).
    cols = max(200, 40 * min(n, 8))
    rows = max(60, 12 * ((n + 3) // 4))
    decode = f"{session}:decode"
    session_created = False
    try:
        # Start an inert process so remain-on-exit is configured before any
        # worker can finish and remove the decode window.
        run_tmux(
            [
                tmux,
                "new-session",
                "-d",
                "-s",
                session,
                "-x",
                str(cols),
                "-y",
                str(rows),
                "-n",
                "decode",
                "sleep 2147483647",
            ]
        )
        session_created = True
        run_tmux([tmux, "set-window-option", "-t", decode, "remain-on-exit"])
        summary_started = time.time()
        run_tmux([tmux, "respawn-pane", "-k", "-t", decode, pane_command(0)])
        summary_command = shlex.join(
            [
                sys.executable,
                "-m", "parallelhue", "--summary-follow",
                "--session", session,
                "--summary-start-epoch", str(summary_started),
                "--summary-timeout", str(summary_timeout),
                "--endpoint", "" if endpoint_path else args.endpoint,
                "--timeout", str(args.timeout),
                "--concurrency", str(n),
                "--backend", args.backend,
                "--model", args.model,
            ]
        )
        if before_metrics_path:
            summary_command = (
                f"PARALLELHUE_SUMMARY_BEFORE_METRICS_FILE={shlex.quote(before_metrics_path)} "
                + summary_command
            )
        if endpoint_path:
            summary_command = (
                f"PARALLELHUE_SUMMARY_ENDPOINT_FILE={shlex.quote(endpoint_path)} "
                + summary_command
            )
        for worker_index in range(1, n):
            run_tmux(
                [
                    tmux,
                    "split-window",
                    "-t",
                    decode,
                    "-h",
                    pane_command(worker_index),
                ]
            )
            run_tmux([tmux, "select-layout", "-t", decode, "tiled"], check=False)
        run_tmux([tmux, "select-layout", "-t", decode, "tiled"], check=False)
        run_tmux([tmux, "new-window", "-t", session, "-n", "summary", summary_command])
        run_tmux([tmux, "set-window-option", "-t", f"{session}:summary", "remain-on-exit", "on"], check=False)
        run_tmux([tmux, "select-window", "-t", decode], check=False)
        run_tmux([tmux, "set-option", "-t", session, "mouse", "on"], check=False)
    except BaseException as exc:
        print(f"parallelhue: launch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if session_created:
            try:
                subprocess.run([tmux, "kill-session", "-t", session], check=False)
            except BaseException:
                pass
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        return 1

    print(f"ParallelHue tmux session: {session} panes={n}", file=sys.stderr)
    print(f"attach with: tmux attach -t {shlex.quote(session)}", file=sys.stderr)

    # dspark8 parity: if interactive TTY, attach immediately so the user sees
    # the 16-way split without a second command.
    if sys.stdout.isatty() and not getattr(args, "no_attach", False):
        subprocess.call([tmux, "attach-session", "-t", session])
        while (
            subprocess.run(
                [tmux, "has-session", "-t", session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        ):

            time.sleep(0.2)
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
    worker_index = 0 if args.worker_index is None else int(args.worker_index)
    # dspark8 prints [1/16]..[16/16] so each pane is identifiable at a glance.
    if args.concurrency == 1 and args.worker_index is not None:
        total = os.environ.get("PARALLELHUE_TOTAL")
        if total and total.isdigit():
            print(f"[{worker_index + 1}/{total}]", flush=True)
        else:
            print(f"[{worker_index + 1}]", flush=True)
    streams = (
        client.stream_many([prompt] * args.concurrency)
        if args.concurrency > 1
        else client.stream(prompt, stream_index=worker_index)
    )
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
