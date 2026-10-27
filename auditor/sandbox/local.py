"""LocalSandbox: a subprocess in a temporary copy, with a scrubbed env and a timeout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
import venv
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from auditor.sandbox.base import (
    OUTPUT_DIR_ENV,
    Sandbox,
    SandboxResult,
    check_env,
    check_relative_path,
)

IS_WINDOWS = os.name == "nt"


class LocalSandbox(Sandbox):
    def __init__(self, python: Path, runs_dir: Path) -> None:
        self.python = Path(python)
        self.runs_dir = Path(runs_dir)

    def run(
        self,
        command: Sequence[str],
        files: Mapping[str, Path],
        env: Mapping[str, str] | None = None,
        timeout: float = 600,
    ) -> SandboxResult:
        extra_env = dict(env or {})
        check_env(extra_env)
        rel_paths = {name: check_relative_path(name) for name in files}

        log_dir = self._new_log_dir()
        with tempfile.TemporaryDirectory(prefix="audit-sbx-", ignore_cleanup_errors=True) as tmp:
            work = Path(tmp) / "work"
            out_dir = Path(tmp) / "output"
            tmp_dir = Path(tmp) / "tmp"
            for d in (work, out_dir, tmp_dir):
                d.mkdir()
            for name, source in files.items():
                dest = work.joinpath(*rel_paths[name].parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)

            argv = [str(self.python), *command[1:]] if command and command[0] == "python" else list(command)
            run_env = {**self._base_env(tmp_dir), **extra_env, OUTPUT_DIR_ENV: str(out_dir)}

            exit_code, stdout, stderr, duration, timed_out = _run_process(argv, work, run_env, timeout)
            artifacts = _save_artifacts(out_dir, log_dir / "outputs")

        (log_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (log_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        result = SandboxResult(
            command=list(command),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_s=round(duration, 3),
            timed_out=timed_out,
            artifacts=artifacts,
            log_dir=log_dir,
        )
        (log_dir / "run.json").write_text(json.dumps(_run_record(result, timeout), indent=2), encoding="utf-8")
        return result

    def _new_log_dir(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir = self.runs_dir / f"{stamp}-{uuid.uuid4().hex[:6]}"
        log_dir.mkdir(parents=True)
        return log_dir

    def _base_env(self, tmp_dir: Path) -> dict[str, str]:
        """Only what Python needs to start. Nothing from the caller's environment leaks in."""
        env = {
            "PATH": str(self.python.parent),
            "TEMP": str(tmp_dir),
            "TMP": str(tmp_dir),
            "TMPDIR": str(tmp_dir),
            "HOME": str(tmp_dir),
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if IS_WINDOWS:
            for name in ("SYSTEMROOT", "WINDIR", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"):
                if name in os.environ:
                    env[name] = os.environ[name]
            env["USERPROFILE"] = str(tmp_dir)
            env["PATH"] += os.pathsep + os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")
        return env


def _run_process(
    argv: list[str], cwd: Path, env: dict[str, str], timeout: float
) -> tuple[int | None, str, str, float, bool]:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    start = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=flags,
        start_new_session=not IS_WINDOWS,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        stdout, stderr = proc.communicate()
        timed_out = True
    duration = time.perf_counter() - start
    return (None if timed_out else proc.returncode), stdout, stderr, duration, timed_out


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill the process and anything it started."""
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.kill()


def _save_artifacts(out_dir: Path, dest: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(out_dir)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            artifacts[rel.as_posix()] = target
    return artifacts


def _run_record(result: SandboxResult, timeout: float) -> dict[str, object]:
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "timeout_s": timeout,
        "duration_s": result.duration_s,
        "artifacts": sorted(result.artifacts),
    }


def ensure_sandbox_python(venv_dir: Path, requirements: Path) -> Path:
    """Create (or reuse) the sandbox's own venv. Rebuilt when the requirements change."""
    venv_dir = Path(venv_dir)
    python = venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    marker = venv_dir / ".requirements.sha256"
    digest = hashlib.sha256(Path(requirements).read_bytes()).hexdigest()
    if python.exists() and marker.exists() and marker.read_text().strip() == digest:
        return python

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", str(requirements)],
        check=True,
    )
    marker.write_text(digest)
    return python


def default_sandbox(repo_root: Path) -> LocalSandbox:
    python = ensure_sandbox_python(repo_root / ".sandbox-venv", repo_root / "sandbox-requirements.txt")
    return LocalSandbox(python=python, runs_dir=repo_root / "runs")
