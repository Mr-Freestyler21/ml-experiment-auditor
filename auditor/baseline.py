"""Rerun an audit target as-is and check that its reported metric reproduces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from auditor.config import AuditConfig
from auditor.sandbox import Sandbox, SandboxResult, collect_files

# A reported value counts as reproduced if the rerun lands within this distance of it.
# Reported values are often rounded (0.95 for 0.952), so exact equality is too strict.
TOLERANCE = 0.01

VERSIONS_SNIPPET = """
import json, platform
from importlib import metadata
packages = {}
for name in ("numpy", "pandas", "scikit-learn"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({"python": platform.python_version(), "packages": packages}))
"""


@dataclass(frozen=True)
class BaselineResult:
    status: str  # "reproduced", "not_reproduced" or "failed"
    reported_metric: str
    reported_value: float
    measured_value: float | None
    difference: float | None
    metrics: dict[str, Any]
    random_seed: int | None
    environment: dict[str, Any]
    duration_s: float
    log_dir: Path
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["log_dir"] = str(self.log_dir)
        return data


def run_baseline(config: AuditConfig, sandbox: Sandbox) -> BaselineResult:
    run = sandbox.run(
        ["python", config.entry_script],
        files=collect_files(config.project_dir),
        timeout=config.timeout_seconds,
    )
    environment = _environment(sandbox)
    metrics, error = _read_metrics(run)
    measured = metrics.get(config.reported_metric) if metrics else None

    if error is None and not isinstance(measured, (int, float)):
        error = f"metrics.json has no numeric {config.reported_metric!r}"
    if error is not None:
        status, measured, difference = "failed", None, None
    else:
        difference = round(float(measured) - config.reported_value, 6)
        status = "reproduced" if abs(difference) <= TOLERANCE else "not_reproduced"

    result = BaselineResult(
        status=status,
        reported_metric=config.reported_metric,
        reported_value=config.reported_value,
        measured_value=float(measured) if measured is not None else None,
        difference=difference,
        metrics=metrics or {},
        random_seed=config.random_seed,
        environment=environment,
        duration_s=run.duration_s,
        log_dir=run.log_dir,
        error=error,
    )
    (run.log_dir / "baseline.json").write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return result


def _read_metrics(run: SandboxResult) -> tuple[dict[str, Any] | None, str | None]:
    if run.timed_out:
        return None, f"run timed out after {run.duration_s:.0f}s"
    if run.exit_code != 0:
        last_line = run.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
        return None, f"script exited with code {run.exit_code}: {last_line[0]}"
    path = run.artifacts.get("metrics.json")
    if path is None:
        return None, "script did not write metrics.json to AUDIT_OUTPUT_DIR"
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"metrics.json is not valid JSON: {exc}"
    if not isinstance(metrics, dict):
        return None, "metrics.json must contain a JSON object"
    return metrics, None


def _environment(sandbox: Sandbox) -> dict[str, Any]:
    """Record the interpreter and library versions the run used."""
    run = sandbox.run(["python", "-c", VERSIONS_SNIPPET], files={}, timeout=60)
    try:
        return json.loads(run.stdout)
    except json.JSONDecodeError:
        return {"error": "could not read library versions", "stderr": run.stderr[-500:]}
