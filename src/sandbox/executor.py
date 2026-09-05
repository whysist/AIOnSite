"""Isolated Python code execution.

Two isolation modes, chosen automatically:

* **Docker mode** (``settings.sandbox_use_docker`` and the ``docker`` binary
  is actually on ``PATH``): the code runs in a throwaway container with
  ``--network none``, a memory/CPU/pids cap, and a read-only root -- a real
  OS-level boundary (separate kernel namespaces, no host filesystem access
  beyond the one read-only mount of the script itself).
* **Subprocess mode** (fallback, always available): the code runs as a
  plain child process (``python -I -S``) in a scratch temp directory with a
  stripped-down environment (no inherited secrets) and a hard wall-clock
  timeout. This is weaker than Docker -- it is the same OS user and kernel,
  so it is not a security boundary against a determined attacker -- but it
  still contains runaway/buggy code (infinite loops, huge output, accidental
  side effects) which is the common case for LLM-generated snippets.

Both modes are preceded by a static AST check (:func:`_static_check`) that
rejects imports/calls with no legitimate reason to appear in a short
calculation snippet (``os``, ``subprocess``, ``socket``, ``eval``, ``open``,
...). This is defense in depth, not the security boundary itself -- a
determined attacker can defeat AST denylisting (e.g. via
``().__class__.__bases__``), which is exactly why process/container
isolation is still the real boundary and Docker mode should be preferred
whenever code from an untrusted source (an LLM) needs real isolation
guarantees.
"""

from __future__ import annotations

import ast
import asyncio
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from pydantic import BaseModel

_DENIED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "multiprocessing",
    "importlib", "pickle", "marshal", "pty", "signal", "resource", "asyncio",
    "threading", "http", "urllib", "ftplib", "smtplib", "telnetlib", "webbrowser",
}
_DENIED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "vars", "globals", "locals"}
_DENIED_ATTRS = {"__subclasses__", "__globals__", "__builtins__", "__base__", "__bases__"}


class SandboxResult(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    timed_out: bool = False
    used_docker: bool = False
    duration_ms: float = 0.0
    # Set only when rejected before any process ran (static check failure) --
    # distinguishes "this code was refused" from "this code ran and failed".
    stage: str = "executed"
    error: str | None = None


def _static_check(code: str) -> str | None:
    """Return an error string if *code* is rejected, else ``None``."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"could not parse code: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DENIED_IMPORTS:
                    return f"import of {alias.name!r} is not allowed in the sandbox"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _DENIED_IMPORTS:
                return f"import from {node.module!r} is not allowed in the sandbox"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DENIED_CALLS:
                return f"call to {node.func.id!r} is not allowed in the sandbox"
        elif isinstance(node, ast.Attribute) and node.attr in _DENIED_ATTRS:
            return f"access to {node.attr!r} is not allowed in the sandbox"
    return None


class SandboxExecutor:
    def __init__(
        self,
        *,
        use_docker: bool = False,
        docker_image: str = "python:3.11-slim",
        timeout_seconds: float = 10.0,
        max_output_chars: int = 8000,
        memory_limit_mb: int = 256,
        cpu_limit: float = 1.0,
    ) -> None:
        self._use_docker = use_docker
        self._docker_image = docker_image
        self._timeout = timeout_seconds
        self._max_output = max_output_chars
        self._memory_limit_mb = memory_limit_mb
        self._cpu_limit = cpu_limit

    async def run(self, code: str) -> SandboxResult:
        rejection = _static_check(code)
        if rejection:
            return SandboxResult(ok=False, stage="rejected", error=rejection)

        docker_bin = shutil.which("docker") if self._use_docker else None
        tmpdir = Path(tempfile.mkdtemp(prefix="aionsite_sandbox_"))
        script = tmpdir / "snippet.py"
        script.write_text(code, encoding="utf-8")
        try:
            if docker_bin:
                return await self._run_docker(docker_bin, tmpdir)
            return await self._run_subprocess(script, tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    async def _run_subprocess(self, script: Path, cwd: Path) -> SandboxResult:
        # Isolated mode (-I: ignore PYTHONPATH/user site/env-derived config,
        # -S: skip the `site` module) plus a stripped environment -- not a
        # real security boundary (same OS user/kernel), but removes the
        # obvious footguns (inherited secrets in env vars, user site-packages).
        env = {"PATH": os.environ.get("PATH", "")}
        if os.name == "nt":
            # python.exe on Windows needs these to even start reliably.
            for key in ("SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP"):
                if key in os.environ:
                    env[key] = os.environ[key]
        started = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            "python", "-I", "-S", str(script),
            cwd=str(cwd), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            timed_out = False
            return_code = proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_b, stderr_b, timed_out, return_code = b"", b"execution timed out", True, None
        duration_ms = (time.perf_counter() - started) * 1000
        return SandboxResult(
            ok=(not timed_out and return_code == 0),
            stdout=_truncate(stdout_b.decode("utf-8", errors="replace"), self._max_output),
            stderr=_truncate(stderr_b.decode("utf-8", errors="replace"), self._max_output),
            return_code=return_code, timed_out=timed_out, used_docker=False,
            duration_ms=duration_ms,
        )

    async def _run_docker(self, docker_bin: str, cwd: Path) -> SandboxResult:
        started = time.perf_counter()
        # A container is given an explicit name specifically so a timeout
        # can kill *it* -- killing the `docker run` CLI client process (via
        # proc.kill()) only kills the client; dockerd keeps the container
        # itself running detached, unbounded, unless it is stopped by name/
        # id. Verified live: without this, a timed-out infinite loop kept
        # consuming CPU in an orphaned container indefinitely.
        container_name = f"aionsite-sandbox-{uuid.uuid4().hex[:12]}"
        cmd = [
            docker_bin, "run", "--rm", "--name", container_name,
            "--network", "none",
            "--memory", f"{self._memory_limit_mb}m",
            "--cpus", str(self._cpu_limit),
            "--pids-limit", "64",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=16m",
            "-v", f"{cwd}:/sandbox:ro",
            "-w", "/sandbox",
            self._docker_image, "python", "-I", "-S", "/sandbox/snippet.py",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            # A little slack over the in-container timeout for docker's own
            # container start/stop overhead.
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout + 5
            )
            timed_out = False
            return_code = proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            await self._force_kill_container(docker_bin, container_name)
            stdout_b, stderr_b, timed_out, return_code = b"", b"execution timed out", True, None
        except asyncio.CancelledError:
            # This coroutine was cancelled from outside (e.g. the pipeline
            # node's own timeout firing first) -- the container must still
            # be cleaned up, then cancellation must propagate as normal.
            proc.kill()
            await self._force_kill_container(docker_bin, container_name)
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        return SandboxResult(
            ok=(not timed_out and return_code == 0),
            stdout=_truncate(stdout_b.decode("utf-8", errors="replace"), self._max_output),
            stderr=_truncate(stderr_b.decode("utf-8", errors="replace"), self._max_output),
            return_code=return_code, timed_out=timed_out, used_docker=True,
            duration_ms=duration_ms,
        )

    @staticmethod
    async def _force_kill_container(docker_bin: str, name: str) -> None:
        """Best-effort ``docker kill`` by name -- harmless if the container
        already exited on its own (``--rm`` will have removed it, and
        ``docker kill`` on an unknown name just fails silently here)."""
        try:
            kill_proc = await asyncio.create_subprocess_exec(
                docker_bin, "kill", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(kill_proc.wait(), timeout=5)
        except Exception:  # noqa: BLE001 -- cleanup must never raise into the caller
            pass


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(truncated, {len(text) - limit} more chars)"
