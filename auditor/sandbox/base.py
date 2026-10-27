"""The Sandbox interface: the only way audited code is ever executed."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Audited scripts write their outputs (metrics.json, ...) into this directory.
OUTPUT_DIR_ENV = "AUDIT_OUTPUT_DIR"

# Env var names that must never reach audited code.
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)

SKIP_DIRS = {".git", ".venv", ".sandbox-venv", "__pycache__", ".pytest_cache", "runs"}


@dataclass(frozen=True)
class SandboxResult:
    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    artifacts: dict[str, Path]
    log_dir: Path

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox(ABC):
    @abstractmethod
    def run(
        self,
        command: Sequence[str],
        files: Mapping[str, Path],
        env: Mapping[str, str] | None = None,
        timeout: float = 600,
    ) -> SandboxResult:
        """Run `command` in a fresh, isolated copy of `files`.

        `files` maps a relative path inside the sandbox to a local source file.
        A leading "python" in `command` means the sandbox's own interpreter.
        Anything the command writes to $AUDIT_OUTPUT_DIR comes back as `artifacts`.
        """


def collect_files(folder: Path) -> dict[str, Path]:
    """Map every file under `folder` to its relative path, skipping tooling dirs."""
    folder = folder.resolve()
    files: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder)
        if path.is_file() and not SKIP_DIRS.intersection(rel.parts):
            files[rel.as_posix()] = path
    return files


def check_relative_path(name: str) -> PurePosixPath:
    """Reject sandbox paths that are absolute or climb out of the sandbox root."""
    rel = PurePosixPath(name.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or ":" in rel.parts[0]:
        raise ValueError(f"sandbox file path must be relative and inside the sandbox: {name!r}")
    return rel


def check_env(env: Mapping[str, str]) -> None:
    for name in env:
        if SECRET_NAME.search(name):
            raise ValueError(f"refusing to pass {name!r} into the sandbox: looks like a secret")
