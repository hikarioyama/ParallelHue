import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from parallelhue import cli
from parallelhue.client import StreamChunk


def test_parser_reads_prompt_and_mode():
    args = cli.build_parser().parse_args(["--model", "m", "--mode", "chunk", "hello"])
    assert args.model == "m"
    assert args.prompt_arg == "hello"
    assert args.mode == "chunk"


def test_worker_prints_truthful_chunk_label(monkeypatch, capsys):
    class FakeClient:
        def __init__(self, config):
            pass

        def stream(self, prompt, stream_index=0):
            yield StreamChunk("ph1_" + "a" * 32 + "_0", 0, "safe", mode="SSE CHUNK MODE", color=46)

    monkeypatch.setattr(cli, "ParallelHueClient", FakeClient)
    args = cli.build_parser().parse_args(["--mode", "chunk", "hello"])
    assert cli.run_worker(args) == 0
    captured = capsys.readouterr()
    assert "[SSE CHUNK MODE]" in captured.err
    assert captured.out == "safe\n"


def test_tmux_falls_back_to_worker_when_missing(monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli, "run_worker", lambda args: 7)
    assert cli.main(["--tmux", "hello"]) == 7
    assert "single-terminal worker path" in capsys.readouterr().err


def test_tmux_is_honored_at_concurrency_one(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(cli, "launch_tmux", lambda args, argv0: 11)
    assert cli.main(["--tmux", "hello"]) == 11


def test_tmux_subprocess_arguments_use_worker_index_zero(monkeypatch):
    calls = []

    class FixedUUID:
        hex = "0123456789abcdef"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PARALLELHUE_API_KEY", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: FixedUUID())
    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: calls.append((command, check)))
    args = cli.build_parser().parse_args(
        [
            "--tmux",
            "--concurrency",
            "1",
            "--mode",
            "chunk",
            "--endpoint",
            "http://example.test/v1/chat/completions",
            "--model",
            "demo",
            "--max-tokens",
            "7",
            "--timeout",
            "12.5",
            "hello world",
        ]
    )
    assert cli.launch_tmux(args, "parallelhue") == 0

    expected = shlex.join(
        [
            sys.executable,
            "-m",
            "parallelhue",
            "--no-tmux",
            "--mode",
            "chunk",
            "--endpoint",
            "http://example.test/v1/chat/completions",
            "--model",
            "demo",
            "--max-tokens",
            "7",
            "--concurrency",
            "1",
            "--timeout",
            "12.5",
            "--prompt",
            "hello world",
            "--worker-index",
            "0",
        ]
    )
    assert calls == [
        (["/usr/bin/tmux", "new-session", "-d", "-s", "parallelhue-01234567", expected], True),
        (["/usr/bin/tmux", "select-layout", "-t", "parallelhue-01234567", "tiled"], True),
    ]


def test_tmux_subprocess_arguments_assign_unique_worker_indices(monkeypatch):
    calls = []

    class FixedUUID:
        hex = "fedcba9876543210"

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: FixedUUID())
    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: calls.append((command, check)))
    args = cli.build_parser().parse_args(["--tmux", "--concurrency", "3", "hello"])

    assert cli.launch_tmux(args, "parallelhue") == 0
    worker_commands = [
        command for command, check in calls if check and command[1] in ("new-session", "split-window")
    ]
    worker_indices = [shlex.split(command[-1])[-1] for command in worker_commands]
    assert worker_indices == ["0", "1", "2"]
    assert len(set(worker_indices)) == 3
    assert all(
        shlex.split(command[-1])[-2:] == ["--worker-index", index]
        for command, index in zip(worker_commands, worker_indices)
    )


def test_non_tmux_worker_uses_configured_concurrency(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self, config):
            calls["concurrency"] = config.concurrency

        def stream_many(self, prompts):
            calls["prompts"] = list(prompts)
            yield StreamChunk("ph1_" + "a" * 32 + "_0", 0, "safe", mode="SSE CHUNK MODE")

    monkeypatch.setattr(cli, "ParallelHueClient", FakeClient)
    args = cli.build_parser().parse_args(["--mode", "chunk", "--concurrency", "2", "hello"])
    assert cli.run_worker(args) == 0
    assert calls == {"concurrency": 2, "prompts": ["hello", "hello"]}


def test_tmux_authenticated_workers_use_private_wrappers(monkeypatch):
    calls = []
    inspected = []
    secret = "safe key 'with' $shell\nnewline"

    class FixedUUID:
        hex = "0123456789abcdef"

    def fake_run(command, check):
        calls.append((command, check))
        if command[1] in ("new-session", "split-window"):
            wrapper = Path(command[-1])
            inspected.append(
                (
                    wrapper,
                    wrapper.read_text(),
                    wrapper.stat().st_mode & 0o777,
                    wrapper.parent.stat().st_mode & 0o777,
                    wrapper.stat().st_uid,
                )
            )

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: FixedUUID())
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    args = cli.build_parser().parse_args(["--tmux", "--concurrency", "2", "--api-key", secret, "hello"])

    try:
        assert cli.launch_tmux(args) == 0
        worker_calls = [
            command for command, check in calls if check and command[1] in ("new-session", "split-window")
        ]
        assert len(worker_calls) == 2
        assert len({command[-1] for command in worker_calls}) == 2
        assert all(command[-1].endswith(".sh") for command in worker_calls)
        assert secret not in repr(calls)
        assert all(
            f"export {name}={shlex.quote(secret)}" in script
            for _, script, _, _, _ in inspected
            for name in ("PARALLELHUE_API_KEY", "OPENAI_API_KEY")
        )
        assert all(
            "exec " + shlex.join([sys.executable, "-m", "parallelhue"]) in script
            for _, script, _, _, _ in inspected
        )
        assert all(mode == 0o700 for _, _, mode, _, _ in inspected)
        assert all(mode == 0o700 for _, _, _, mode, _ in inspected)
        assert all(owner == os.getuid() for _, _, _, _, owner in inspected)
    finally:
        if inspected:
            shutil.rmtree(inspected[0][0].parent, ignore_errors=True)


def test_tmux_partial_launch_failure_kills_session_and_cleans_private_runtime(monkeypatch):
    calls = []
    runtime_dir = None

    class FixedUUID:
        hex = "0123456789abcdef"

    def fail_split(command, check):
        nonlocal runtime_dir
        calls.append((command, check))
        if command[1] == "new-session":
            runtime_dir = Path(command[-1]).parent
            return None
        if command[1] == "split-window":
            raise RuntimeError("split failed")
        raise AssertionError(f"unexpected tmux command: {command!r}")

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: FixedUUID())
    monkeypatch.setattr(cli.subprocess, "run", fail_split)
    args = cli.build_parser().parse_args(["--tmux", "--concurrency", "2", "--api-key", "secret", "hello"])

    with pytest.raises(RuntimeError, match="split failed"):
        cli.launch_tmux(args)

    assert runtime_dir is not None
    assert not runtime_dir.exists()
    assert [command[0][1] for command in calls] == ["new-session", "split-window", "kill-session"]
    assert calls[-1] == (["/usr/bin/tmux", "kill-session", "-t", "parallelhue-01234567"], False)


def test_tmux_authenticated_launch_failure_cleans_private_runtime(monkeypatch):
    calls = []
    runtime_dir = None

    def fail_run(command, check):
        nonlocal runtime_dir
        calls.append((command, check))
        runtime_dir = Path(command[-1]).parent
        assert "secret" not in repr(command)
        raise RuntimeError("tmux failed")

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(cli.subprocess, "run", fail_run)
    args = cli.build_parser().parse_args(["--tmux", "--api-key", "secret", "hello"])

    with pytest.raises(RuntimeError, match="tmux failed"):
        cli.launch_tmux(args)
    assert runtime_dir is not None
    assert not runtime_dir.exists()
    assert [command[0][1] for command in calls] == ["new-session"]


def test_main_sanitizes_client_error(monkeypatch, capsys):
    def fail(args):
        raise cli.ClientError("\x1b]2;owned\a")

    monkeypatch.setattr(cli, "run_worker", fail)
    assert cli.main(["hello"]) == 2
    assert "\x1b" not in capsys.readouterr().err
