"""The tools the model may call, and the code that runs them.

Every tool returns plain JSON. Each result is stored as an evidence item with an id
(E1, E2, ...) that the model must cite in its final verdict. Arguments from the model
are validated here; bad arguments come back as an error result, never an exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from auditor.baseline import BaselineResult, run_baseline
from auditor.checks.base import CheckResult
from auditor.checks.future_info import Suspect, check_future_info, feature_power, rules_suspects, run_shuffled
from auditor.checks.group_overlap import check_group_overlap
from auditor.checks.preprocessing import check_preprocessing
from auditor.config import AuditConfig
from auditor.sandbox import Sandbox, collect_files
from auditor.trace import TracedRun, compare_metrics, run_traced

MAX_READ_LINES = 200
MAX_LIST_ITEMS = 5
MAX_ABLATION_COLUMNS = 3
SPLIT_STRATEGIES = ("random", "group", "temporal")


@dataclass
class EvidenceItem:
    id: str
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AuditSession:
    """Everything one audit has measured so far. Expensive runs are cached and reused."""

    config: AuditConfig
    sandbox: Sandbox
    data: pd.DataFrame
    evidence: list[EvidenceItem] = field(default_factory=list)
    final: dict[str, Any] | None = None
    _plain: BaselineResult | None = None
    _traced: TracedRun | None = None

    # --- cached runs ---

    def plain(self) -> BaselineResult:
        if self._plain is None:
            self._plain = run_baseline(self.config, self.sandbox)
        return self._plain

    def traced(self) -> TracedRun:
        if self._traced is None:
            self._traced = run_traced(self.config, self.sandbox)
        return self._traced

    def incomplete_reason(self) -> str | None:
        traced = self.traced()
        if traced.error:
            return f"traced run failed: {traced.error}"
        if not compare_metrics(self.plain().metrics, traced.metrics)["match"]:
            return "the traced run's metrics differ from the plain run"
        return traced.incomplete_reason

    # --- dispatch ---

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Run one tool call and record it as evidence. Always returns a JSON-able dict."""
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            result: dict[str, Any] = {"error": f"unknown tool {name!r}"}
        elif arguments is None:
            result = {"error": "arguments were not valid JSON"}
        else:
            try:
                result = handler(self, arguments)
            except ToolArgumentError as exc:
                result = {"error": str(exc)}
            except Exception as exc:  # a tool failure is reported to the model, not raised
                result = {"error": f"{type(exc).__name__}: {exc}"}
        if name == "finalize":
            return result  # verdicts are not evidence
        item = EvidenceItem(id=f"E{len(self.evidence) + 1}", tool=name, arguments=arguments or {}, result=result)
        self.evidence.append(item)
        return {"evidence_id": item.id, **result}

    def evidence_by_id(self) -> dict[str, EvidenceItem]:
        return {e.id: e for e in self.evidence}


class ToolArgumentError(ValueError):
    pass


# --- tool implementations ---------------------------------------------------------------


def _list_files(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    files = collect_files(s.config.project_dir)
    return {"files": [{"path": p, "bytes": f.stat().st_size} for p, f in files.items()]}


def _read_file(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    path = _require_str(args, "path")
    files = collect_files(s.config.project_dir)
    if path not in files:
        raise ToolArgumentError(f"no such file in the project: {path!r}. Use list_files.")
    text = files[path].read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if path.lower().endswith(".csv"):
        return {"path": path, "total_lines": len(lines), "first_lines": lines[:6]}
    start = int(args.get("start_line") or 1)
    chosen = lines[start - 1 : start - 1 + MAX_READ_LINES]
    numbered = "\n".join(f"{start + i:4d}  {line}" for i, line in enumerate(chosen))
    return {"path": path, "total_lines": len(lines), "content": numbered}


def _run_baseline(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    b = s.plain()
    return {
        "status": b.status,
        "reported": {b.reported_metric: b.reported_value},
        "measured": b.measured_value,
        "difference": b.difference,
        "metrics": b.metrics,
        "random_seed": b.random_seed,
        "environment": b.environment,
        "duration_s": b.duration_s,
        "error": b.error,
    }


def _run_check(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    check = _require_choice(args, "check", ("A", "B", "C"))
    traced = s.traced()
    if traced.error:
        return {"error": f"the traced run failed: {traced.error}"}
    incomplete = s.incomplete_reason()
    if check == "A":
        result = check_preprocessing(traced.events, traced.splits, incomplete)
    elif check == "C":
        result = check_group_overlap(
            s.data, traced.splits, s.config.group_columns, s.config.time_column, incomplete
        )
    else:
        result = _check_b(s, traced, args.get("suspects") or [])
    return compact(result.to_json())


def _check_b(s: AuditSession, traced: TracedRun, proposed: list[Any]) -> CheckResult:
    features = next((list(sp.features) for sp in traced.splits if sp.features), None)
    suspects: list[Suspect] = []
    if features:
        power = feature_power(s.data, [f for f in features if f in s.data.columns], s.config.target)
        suspects = rules_suspects(features, s.config.target, power)
        for item in proposed:
            if isinstance(item, dict) and item.get("column") in features:
                suspects.append(Suspect(item["column"], "llm", str(item.get("reason", ""))[:300]))
    return check_future_info(s.config, s.sandbox, s.data, features, s.plain().metrics, suspects)


def _run_ablation(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    columns = args.get("columns")
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        raise ToolArgumentError("columns must be a non-empty list of column names")
    unknown = [c for c in columns if c not in s.data.columns]
    if unknown:
        raise ToolArgumentError(f"not columns of the data: {unknown}")
    metric = s.config.reported_metric
    baseline = s.plain().metrics.get(metric)
    results = []
    for column in columns[:MAX_ABLATION_COLUMNS]:
        metrics, error, log_dir = run_shuffled(s.config, s.sandbox, column, seed=s.config.random_seed or 0)
        value = metrics.get(metric) if metrics else None
        results.append({
            "column": column,
            metric: value,
            "baseline": baseline,
            "change": round(value - baseline, 4) if isinstance(value, (int, float)) and baseline is not None else None,
            "error": error,
        })
    out: dict[str, Any] = {"method": "column shuffled in a copy of the data, script rerun unchanged", "results": results}
    if len(columns) > MAX_ABLATION_COLUMNS:
        out["not_run"] = columns[MAX_ABLATION_COLUMNS:]
    return out


def _run_with_split(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    strategy = _require_choice(args, "strategy", SPLIT_STRATEGIES)
    spec: dict[str, Any] = {"strategy": strategy, "seed": s.config.random_seed or 0}
    if strategy == "group":
        cols = args.get("group_columns") or (list(s.config.group_columns[0]) if s.config.group_columns else None)
        if not cols or not all(c in s.data.columns for c in cols):
            raise ToolArgumentError(f"group_columns must be columns of the data, got {cols!r}")
        spec["group_columns"] = cols
    if strategy == "temporal":
        col = args.get("time_column") or s.config.time_column
        if col not in s.data.columns:
            raise ToolArgumentError(f"time_column must be a column of the data, got {col!r}")
        spec["time_column"] = col
    if args.get("test_size") is not None:
        size = float(args["test_size"])
        if not 0 < size < 1:
            raise ToolArgumentError("test_size must be between 0 and 1")
        spec["test_size"] = size
    restrict = args.get("restrict_to")
    if restrict:
        if not isinstance(restrict, dict) or restrict.get("column") not in s.data.columns or "value" not in restrict:
            raise ToolArgumentError("restrict_to must be {column, value} with a column of the data")
        spec["restrict"] = {"column": restrict["column"], "value": restrict["value"]}

    traced = run_traced(s.config, s.sandbox, split_override=spec)
    if traced.error:
        return {"error": f"rerun failed: {traced.error}", "stderr_tail": traced.run.stderr[-600:]}
    if not any(e.get("event") == "split" and e.get("override") for e in traced.events):
        return {"error": "the script does not call sklearn's train_test_split, so its split cannot be swapped"}

    metric = s.config.reported_metric
    baseline = s.plain().metrics.get(metric)
    value = (traced.metrics or {}).get(metric)
    return {
        "split": spec,
        "metrics": traced.metrics,
        "baseline_" + metric: baseline,
        "change_vs_baseline": round(value - baseline, 4) if isinstance(value, (int, float)) and baseline else None,
        "evaluations": [_describe_split(s, sp) for sp in traced.splits[:MAX_LIST_ITEMS]],
    }


def _describe_split(s: AuditSession, split: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"model": split.model, "n_train": split.n_train, "n_test": split.n_test}
    if not split.has_row_identity:
        return out
    y = s.data[s.config.target]
    if y.nunique() == 2:
        positive = sorted(y.dropna().unique())[1]
        out["target_rate_train"] = round(float((y.loc[split.train_rows] == positive).mean()), 4)
        out["target_rate_test"] = round(float((y.loc[split.test_rows] == positive).mean()), 4)
        out["majority_class_accuracy_test"] = round(max(out["target_rate_test"], 1 - out["target_rate_test"]), 4)
    overlap = check_group_overlap(s.data, [split], s.config.group_columns, s.config.time_column)
    if overlap.evidence.get("splits"):
        detail = overlap.evidence["splits"][0]
        out["groups"] = [
            {k: g[k] for k in ("columns", "train_groups", "test_groups", "shared_groups")} for g in detail["groups"]
        ]
        if detail.get("time"):
            out["time"] = {k: detail["time"][k] for k in ("train_range", "test_range", "test_rows_before_train_end")}
    return out


def _finalize(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    from auditor.agent.verdict import validate_verdict

    problems = validate_verdict(args, s.evidence)
    if problems:
        return {"accepted": False, "problems": problems, "hint": "Fix these and call finalize again."}
    s.final = args
    return {"accepted": True}


TOOL_HANDLERS: dict[str, Callable[[AuditSession, dict[str, Any]], dict[str, Any]]] = {
    "list_files": _list_files,
    "read_file": _read_file,
    "run_baseline": _run_baseline,
    "run_check": _run_check,
    "run_ablation": _run_ablation,
    "run_with_split": _run_with_split,
    "finalize": _finalize,
}


# --- schemas shown to the model ---------------------------------------------------------


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
        "strict": False,
    }


EVIDENCE_IDS = {"type": "array", "items": {"type": "string"}, "description": "Evidence ids such as E3"}

TOOL_SCHEMAS = [
    _tool("list_files", "List the project's files.", {}, []),
    _tool("read_file", "Read a project file with line numbers (CSV: header and first rows only).",
          {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"]),
    _tool("run_baseline", "Rerun the script unchanged in the sandbox and return its metrics.", {}, []),
    _tool(
        "run_check",
        "Run one deterministic check. A: preprocessing fitted on test rows. B: features not known at "
        "prediction time (pass the columns you suspect, with reasons; code measures them with shuffle reruns). "
        "C: train/test overlap of groups (group_columns) and time (time_column).",
        {
            "check": {"type": "string", "enum": ["A", "B", "C"]},
            "suspects": {
                "type": "array",
                "items": {"type": "object", "properties": {"column": {"type": "string"}, "reason": {"type": "string"}},
                          "required": ["column", "reason"]},
                "description": "Check B only: features you think would not be known at prediction time.",
            },
        },
        ["check"],
    ),
    _tool("run_ablation", "Shuffle each given column (max 3) in a copy of the data, rerun, report the metric change.",
          {"columns": {"type": "array", "items": {"type": "string"}}}, ["columns"]),
    _tool(
        "run_with_split",
        "Rerun the unchanged script with its train_test_split replaced. strategy random: random rows; group: no "
        "group (e.g. race) in both train and test; temporal: test on the latest period. restrict_to keeps only "
        "rows where column == value (e.g. one season) before splitting. Returns metrics, the change versus the "
        "baseline, train/test target rates, the majority-class baseline and group/time overlap.",
        {
            "strategy": {"type": "string", "enum": list(SPLIT_STRATEGIES)},
            "group_columns": {"type": "array", "items": {"type": "string"}},
            "time_column": {"type": "string"},
            "test_size": {"type": "number"},
            "restrict_to": {"type": "object", "properties": {"column": {"type": "string"}, "value": {}},
                            "required": ["column", "value"]},
        },
        ["strategy"],
    ),
    _tool(
        "finalize",
        "Submit the audit verdict. Every finding and caveat must cite evidence ids. Every number you write must "
        "come from a tool result. confirmed needs a flagged check; not_found needs checks A, B and C all run.",
        {
            "verdict": {"type": "string", "enum": ["confirmed", "not_found", "inconclusive"]},
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "family": {"type": "string", "enum": ["A", "B", "C"]},
                        "title": {"type": "string"},
                        "explanation": {"type": "string"},
                        "evidence_ids": EVIDENCE_IDS,
                    },
                    "required": ["family", "title", "explanation", "evidence_ids"],
                },
            },
            "caveats": {
                "type": "array",
                "items": {"type": "object", "properties": {"text": {"type": "string"}, "evidence_ids": EVIDENCE_IDS},
                          "required": ["text", "evidence_ids"]},
            },
        },
        ["verdict", "summary", "findings", "caveats"],
    ),
]


# --- helpers ----------------------------------------------------------------------------


def compact(obj: Any, max_list: int = MAX_LIST_ITEMS) -> Any:
    """Shorten long lists so tool results stay small enough to send to the model."""
    if isinstance(obj, dict):
        return {k: compact(v, max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [compact(v, max_list) for v in obj[:max_list]]
        if len(obj) > max_list:
            items.append(f"... {len(obj) - max_list} more")
        return items
    return obj


def to_json_text(result: dict[str, Any]) -> str:
    return json.dumps(result, default=str)


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ToolArgumentError(f"{key} must be a non-empty string")
    return value


def _require_choice(args: dict[str, Any], key: str, choices: tuple[str, ...]) -> str:
    value = args.get(key)
    if value not in choices:
        raise ToolArgumentError(f"{key} must be one of {list(choices)}, got {value!r}")
    return value


def load_session(config: AuditConfig, sandbox: Sandbox) -> AuditSession:
    return AuditSession(config=config, sandbox=sandbox, data=pd.read_csv(Path(config.project_dir) / config.data_path))
