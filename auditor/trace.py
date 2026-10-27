"""Run an audit target under the trace launcher and turn the trace into train/test splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auditor.config import AuditConfig
from auditor.sandbox import Sandbox, SandboxResult, collect_files

LAUNCHER_SOURCE = Path(__file__).resolve().parent / "sandbox" / "trace_launcher.py"
LAUNCHER_NAME = "_audit_trace.py"
TRACE_NAME = "trace.jsonl"
PREDICT_METHODS = {"predict", "predict_proba", "decision_function", "score"}


@dataclass(frozen=True)
class EvaluationSplit:
    """One model fitted on some rows and then evaluated on others."""

    model: str
    train_rows: list[Any] | None
    test_rows: list[Any] | None
    fit_line: int | None
    predict_line: int | None
    predict_methods: tuple[str, ...]
    n_train: int | None
    n_test: int | None

    @property
    def has_row_identity(self) -> bool:
        return self.train_rows is not None and self.test_rows is not None


@dataclass(frozen=True)
class TracedRun:
    run: SandboxResult
    events: list[dict[str, Any]]
    splits: list[EvaluationSplit]
    metrics: dict[str, Any] | None
    error: str | None

    @property
    def incomplete_reason(self) -> str | None:
        """Why the trace may be missing calls, or None if it saw everything."""
        escapes = [e for e in self.events if e.get("event") == "parallel_escape"]
        if not escapes:
            return None
        first = escapes[0]
        return (
            f"{len(escapes)} parallel job(s) ran outside the traced process "
            f"(backend {first['backend']} with {first['n_jobs']} workers, line {first.get('line')}), "
            "so some fit/predict calls were not seen"
        )


def compare_metrics(plain: dict[str, Any], traced: dict[str, Any] | None) -> dict[str, Any]:
    """Check that running under the tracer did not change the experiment's results."""
    if traced is None:
        return {"match": False, "differences": {}, "note": "the traced run wrote no metrics.json"}
    differences = {
        key: [plain.get(key), traced.get(key)]
        for key in sorted(set(plain) | set(traced))
        if plain.get(key) != traced.get(key)
    }
    return {"match": not differences, "differences": differences}


def run_traced(config: AuditConfig, sandbox: Sandbox) -> TracedRun:
    files = collect_files(config.project_dir)
    if LAUNCHER_NAME in files:
        raise ValueError(f"project already contains a file named {LAUNCHER_NAME}")
    files[LAUNCHER_NAME] = LAUNCHER_SOURCE
    run = sandbox.run(["python", LAUNCHER_NAME, config.entry_script], files=files, timeout=config.timeout_seconds)

    trace_path = run.artifacts.get(TRACE_NAME)
    if not run.ok:
        error = "run timed out" if run.timed_out else f"script exited with code {run.exit_code}"
    elif trace_path is None:
        error = "the run produced no trace"
    else:
        error = None
    events = read_events(trace_path) if trace_path else []
    metrics_path = run.artifacts.get("metrics.json")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path else None
    except json.JSONDecodeError:
        metrics = None
    return TracedRun(run=run, events=events, splits=extract_splits(events), metrics=metrics, error=error)


def read_events(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_splits(events: list[dict[str, Any]]) -> list[EvaluationSplit]:
    """Pair each supervised fit with the predictions the same model object made afterwards.

    Works for a single train_test_split and for cross-validation alike: every fold's
    cloned model is its own object with its own fit and predict calls.
    """
    calls = sorted((e for e in events if e.get("event") == "call" and e.get("depth") == 0), key=lambda e: e["seq"])
    splits: list[EvaluationSplit] = []
    open_fits: dict[int, dict[str, Any]] = {}
    predictions: dict[int, list[dict[str, Any]]] = {}

    def close(obj: int) -> None:
        fit = open_fits.pop(obj, None)
        preds = predictions.pop(obj, [])
        if fit is None:
            return
        # Group predictions by the rows they were made on (predict and predict_proba
        # on the same test set count as one evaluation).
        by_rows: dict[str, list[dict[str, Any]]] = {}
        for p in preds:
            key = json.dumps(p.get("rows")) if p.get("rows") is not None else f"n={p.get('n_rows')}"
            by_rows.setdefault(key, []).append(p)
        for group in by_rows.values():
            first = group[0]
            splits.append(EvaluationSplit(
                model=fit["class"],
                train_rows=fit.get("rows"),
                test_rows=first.get("rows"),
                fit_line=fit.get("line"),
                predict_line=first.get("line"),
                predict_methods=tuple(dict.fromkeys(p["method"] for p in group)),
                n_train=fit.get("n_rows"),
                n_test=first.get("n_rows"),
            ))

    for call in calls:
        obj = call["obj"]
        if call["method"] == "fit" and call.get("has_y"):
            close(obj)
            open_fits[obj] = call
            predictions[obj] = []
        elif call["method"] in PREDICT_METHODS and obj in open_fits:
            predictions[obj].append(call)
    for obj in list(open_fits):
        close(obj)
    return splits
