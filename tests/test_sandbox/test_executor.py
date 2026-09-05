"""Sandbox executor -- subprocess isolation mode (always available, no
Docker dependency needed for these tests) plus a Docker-mode regression
test that is skipped when Docker isn't available."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from src.sandbox.executor import SandboxExecutor, _static_check


def _executor(**kw) -> SandboxExecutor:
    kw.setdefault("timeout_seconds", 5.0)
    return SandboxExecutor(use_docker=False, **kw)


async def test_prints_output_correctly():
    result = await _executor().run("print(2 + 2)")
    assert result.ok is True
    assert result.stdout.strip() == "4"
    assert result.used_docker is False
    assert result.timed_out is False


async def test_runtime_error_is_reported_not_raised():
    result = await _executor().run("raise ValueError('boom')")
    assert result.ok is False
    assert result.return_code != 0
    assert "ValueError" in result.stderr
    assert result.stage == "executed"  # ran, just failed -- not rejected


async def test_infinite_loop_is_killed_by_timeout():
    result = await _executor(timeout_seconds=1.0).run("while True:\n    pass\n")
    assert result.timed_out is True
    assert result.ok is False


async def test_output_is_truncated():
    result = await _executor(max_output_chars=100).run("print('x' * 10000)")
    assert len(result.stdout) < 200
    assert "truncated" in result.stdout


# ---------------------------------------------------------------------
# Static (AST) pre-check: rejected before any process is ever spawned.


async def test_denied_import_is_rejected_before_execution():
    result = await _executor().run("import os\nprint(os.getcwd())")
    assert result.ok is False
    assert result.stage == "rejected"
    assert "os" in result.error


async def test_denied_import_from_is_rejected():
    result = await _executor().run("from subprocess import run\nrun(['ls'])")
    assert result.stage == "rejected"


async def test_denied_call_eval_is_rejected():
    result = await _executor().run("eval('1+1')")
    assert result.stage == "rejected"
    assert "eval" in result.error


async def test_denied_call_open_is_rejected():
    result = await _executor().run("open('/etc/passwd').read()")
    assert result.stage == "rejected"


async def test_denied_dunder_attribute_is_rejected():
    result = await _executor().run("().__class__.__bases__[0].__subclasses__()")
    assert result.stage == "rejected"


def test_safe_numeric_code_passes_static_check():
    assert _static_check("x = 1 + 2\nprint(x)\nimport math\nprint(math.sqrt(x))") is None


async def test_allowed_stdlib_import_runs_fine():
    result = await _executor().run("import math\nprint(math.sqrt(16))")
    assert result.ok is True
    assert result.stdout.strip() == "4.0"


# ---------------------------------------------------------------------
# Docker mode -- skipped when Docker isn't installed/reachable, and when
# the base image hasn't been pulled (this suite must not require network
# access to pass).

_docker_available = bool(shutil.which("docker"))
if _docker_available:
    try:
        _docker_available = (
            subprocess.run(
                ["docker", "image", "inspect", "python:3.11-slim"],
                capture_output=True, timeout=5,
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001
        _docker_available = False

requires_docker = pytest.mark.skipif(
    not _docker_available, reason="docker (with python:3.11-slim pulled) not available"
)


@requires_docker
async def test_docker_mode_timeout_does_not_leak_a_running_container():
    """Regression test for a real bug found by hand: killing the `docker
    run` CLI client on timeout does NOT stop the container itself --
    dockerd keeps it running detached unless it is killed by name/id. An
    unfixed version of this leaves an infinite loop consuming CPU forever
    in an orphaned container.
    """
    executor = SandboxExecutor(use_docker=True, timeout_seconds=2.0)
    result = await executor.run("while True:\n    pass\n")
    assert result.timed_out is True
    assert result.used_docker is True

    leaked = subprocess.run(
        ["docker", "ps", "--filter", "name=aionsite-sandbox", "--format", "{{.ID}}"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    assert leaked == "", f"sandbox container(s) leaked after timeout: {leaked!r}"


@requires_docker
async def test_docker_mode_runs_in_a_real_separate_container():
    executor = SandboxExecutor(use_docker=True, timeout_seconds=8.0)
    result = await executor.run("print(2 + 2)")
    assert result.ok is True
    assert result.used_docker is True
    assert result.stdout.strip() == "4"


@requires_docker
async def test_docker_mode_filesystem_is_read_only():
    executor = SandboxExecutor(use_docker=True, timeout_seconds=8.0)
    result = await executor.run(
        "from pathlib import Path\nPath('/sandbox/pwned.txt').write_text('x')"
    )
    assert result.ok is False
    assert "Read-only file system" in result.stderr
