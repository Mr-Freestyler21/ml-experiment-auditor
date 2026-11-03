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
from auditor.checks.code_review import code_evidence_by_column, review_project
from auditor.checks.duplicates import check_shared_rows
from auditor.checks.group_overlap import check_group_overlap
from auditor.checks.preprocessing import check_preprocessing
from auditor.checks.selection import check_selection
from auditor.config import AuditConfig
from auditor.patching import Patch, PatchError, make_patch
from auditor.sandbox import Sandbox, collect_files
from auditor.trace import TracedRun, compare_metrics, run_traced

MAX_READ_LINES = 200
MAX_LIST_ITEMS = 5
MAX_ABLATION_COLUMNS = 3
SPLIT_STRATEGIES = ("random", "group", "temporal")
CHECKS = ("A", "B", "C", "D", "E", "R")
FAMILIES = ("A", "B", "C", "D", "E")


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
    work_dir: Path  # where patched copies and diffs are written (never the project folder)
    evidence: list[EvidenceItem] = field(default_factory=list)
    patches: dict[str, Patch] = field(default_factory=dict)
    final: dict[str, Any] | None = None
    _plain: BaselineResult | None = None
    _traced: TracedRun | None = None
    _review: CheckResult | None = None
    _honest: tuple[dict[str, Any], dict[str, Any]] | None | bool = False  # False: not computed yet

    # --- cached runs ---

    def code_review(self) -> CheckResult:
        if self._review is None:
            self._review = review_project(self.config.project_dir, self.config.entry_script, self.config.time_column)
        return self._review

    def honest(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """An honest split for measuring features (time-ordered, or by group) and the script's
        metrics under it; None when the project has no time or group column, or the script's
        split cannot be swapped. Duplicate rows and neighbouring days can hide a leaky
        feature under a random split; under an honest split they cannot."""
        if self._honest is False:
            self._honest = None
            spec: dict[str, Any] | None = None
            if self.config.time_column and self.config.time_column in self.data.columns:
                spec = {"strategy": "temporal", "time_column": self.config.time_column}
            elif self.config.group_columns:
                spec = {"strategy": "group", "group_columns": list(self.config.group_columns[0])}
            if spec is not None and any(e.get("event") == "split" for e in self.traced().events):
                spec["seed"] = self.config.random_seed or 0
                run = run_traced(self.config, self.sandbox, split_override=spec, prepare_scripts=[])
                if not run.error and run.metrics:
                    self._honest = (spec, run.metrics)
        return self._honest or None

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
        "log_dir": str(b.log_dir),
    }


def _run_check(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    check = _require_choice(args, "check", CHECKS)
    if check == "R":
        return compact(s.code_review().to_json(), max_list=12)
    traced = s.traced()
    if traced.error:
        return {"error": f"the traced run failed: {traced.error}"}
    incomplete = s.incomplete_reason()
    if check == "A":
        result = check_preprocessing(traced.events, traced.splits, incomplete, _entry_script(s))
    elif check == "C":
        result = check_group_overlap(
            s.data, traced.splits, s.config.group_columns, s.config.time_column, incomplete
        )
    elif check == "D":
        result = check_shared_rows(s.data, traced.splits, s.config.time_column, incomplete)
    elif check == "E":
        result = check_selection(traced.splits, s.code_review().evidence.get("items", []), incomplete)
    else:
        result = _check_b(s, traced, args.get("suspects") or [], args.get("on_patches") or None)
    return compact(result.to_json())


def _entry_script(s: AuditSession) -> str:
    return (s.config.project_dir / s.config.entry_script).read_text(encoding="utf-8")


def _check_b(s: AuditSession, traced: TracedRun, proposed: list[Any], on_patches: list[str] | None = None
             ) -> CheckResult:
    features = next((list(sp.features) for sp in traced.splits if sp.features), None)
    # A column asked about is measured even when the model got it as a plain array (no names) or
    # through a transformer: shuffling it in the data file still shows whether the result depends on it.
    asked = [str(i.get("column")) for i in proposed if isinstance(i, dict) and i.get("column") in s.data.columns
             and i.get("column") != s.config.target]
    extra = [c for c in dict.fromkeys(asked) if c not in (features or [])]
    if extra:
        features = (features or []) + extra
    suspects: list[Suspect] = []
    if features:
        power = feature_power(s.data, [f for f in features if f in s.data.columns], s.config.target)
        suspects = rules_suspects(features, s.config.target, power)
        for item in proposed:
            if isinstance(item, dict) and item.get("column") in features:
                suspects.append(Suspect(item["column"], "llm", str(item.get("reason", ""))[:300]))
    code = code_evidence_by_column(s.code_review().evidence.get("items", []))
    if on_patches:
        # Measure each suspect on top of the accepted fixes: when several leaks cover for each other,
        # removing one in the flawed pipeline barely moves the score.
        metrics, rerun, label = _patched_rerun(s, on_patches)
        return check_future_info(s.config, s.sandbox, s.data, features, metrics, suspects, rerun=rerun,
                                 code_evidence=code, measured_under=f"the pipeline with fixes {label} applied")
    honest = s.honest()
    if honest is not None:
        spec, metrics = honest
        return check_future_info(s.config, s.sandbox, s.data, features, metrics, suspects, split_override=spec,
                                 code_evidence=code)
    return check_future_info(s.config, s.sandbox, s.data, features, s.plain().metrics, suspects,
                             code_evidence=code)


def _patch_files(s: AuditSession, ids: list[Any]) -> tuple[dict[str, Path], list[str], str]:
    """Patched copies to swap in, the preparation scripts to rerun, and a label like 'P1+P2'."""
    if not isinstance(ids, list) or not ids:
        raise ToolArgumentError("give patch_ids: a list of accepted patch ids (e.g. ['P1', 'P2'])")
    patches = []
    for pid in ids:
        patch = s.patches.get(str(pid))
        if patch is None:
            raise ToolArgumentError(f"unknown patch_id {pid!r}; accepted patches: {sorted(s.patches)}")
        patches.append(patch)
    # A patch built on top of another already contains it: keep only the latest of a chain.
    included = {pid for p in patches for pid in _chain(s, p)[1:]}
    patches = [p for p in patches if p.id not in included]
    ids = [p.id for p in patches]
    paths = [p.path for p in patches]
    if len(set(paths)) != len(paths):
        raise ToolArgumentError("two separate patches change the same file; propose one that builds on the other "
                                "(a new patch of a file is applied on top of its latest patch)")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    overrides = {}
    for patch in patches:
        patched_file = s.work_dir / f"patched_{patch.id}_{Path(patch.path).name}"
        patched_file.write_text(patch.patched, encoding="utf-8")
        overrides[patch.path] = patched_file
    # Patched files other than the entry script are preparation steps: rerun them first, so a fix
    # to the script that builds the data file reaches the data the model is trained on.
    prepare = list(s.config.prepare_scripts) + sorted(
        p for p in paths if p != s.config.entry_script and p not in s.config.prepare_scripts)
    return overrides, prepare, "+".join(str(i) for i in ids)


def _chain(s: AuditSession, patch: Any) -> list[str]:
    """The patch id followed by the ids of the earlier patches it was built on."""
    chain = [patch.id]
    while patch.builds_on and patch.builds_on in s.patches:
        patch = s.patches[patch.builds_on]
        chain.append(patch.id)
    return chain


def _patched_rerun(s: AuditSession, ids: list[Any]):
    """Metrics of the patched pipeline, and a function that reruns it with one column shuffled."""
    overrides, prepare, label = _patch_files(s, ids)
    base = run_traced(s.config, s.sandbox, file_overrides=overrides, prepare_scripts=prepare)
    if base.error or not base.metrics:
        raise ToolArgumentError(f"the patched pipeline {label} failed: {base.error or 'no metrics.json'}")

    def rerun(column: str):
        run = run_traced(s.config, s.sandbox, file_overrides=overrides, prepare_scripts=prepare,
                         shuffle_column=column)
        if run.error or not run.metrics:
            return None, f"rerun failed ({run.error or 'no metrics.json'})", run.run.log_dir
        return run.metrics, None, run.run.log_dir

    return base.metrics, rerun, label


def _run_ablation(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    columns = args.get("columns")
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        raise ToolArgumentError("columns must be a non-empty list of column names")
    unknown = [c for c in columns if c not in s.data.columns]
    if unknown:
        raise ToolArgumentError(f"not columns of the data: {unknown}")
    metric = s.config.reported_metric
    spec, rerun, label = None, None, None
    if args.get("on_patches"):
        base_metrics, rerun, label = _patched_rerun(s, args["on_patches"])
        baseline = base_metrics.get(metric)
    else:
        honest = s.honest() if args.get("honest_split", True) else None
        spec = honest[0] if honest else None
        baseline = (honest[1] if honest else s.plain().metrics).get(metric)
    results = []
    for column in columns[:MAX_ABLATION_COLUMNS]:
        if rerun is not None:
            metrics, error, log_dir = rerun(column)
        else:
            metrics, error, log_dir = run_shuffled(s.config, s.sandbox, column, seed=s.config.random_seed or 0,
                                                   split_override=spec)
        value = metrics.get(metric) if metrics else None
        results.append({
            "column": column,
            metric: value,
            "baseline": baseline,
            "change": round(value - baseline, 4) if isinstance(value, (int, float)) and baseline is not None else None,
            "error": error,
        })
    method = "column shuffled in a copy of the data, script rerun unchanged"
    if label:
        method = f"column shuffled, on top of the pipeline with fixes {label} applied"
    elif spec:
        method += f", under an honest {spec['strategy']} split (so copies and neighbouring rows cannot hide a leak)"
    out: dict[str, Any] = {"method": method, "results": results}
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
        "log_dir": str(traced.run.log_dir),
    }


def _describe_split(s: AuditSession, split: Any, data: pd.DataFrame | None = None) -> dict[str, Any]:
    """Sizes, target rates, majority baseline and overlap of one evaluation. `data` is the data
    file the run actually used (rebuilt by a patched preparation script), default the original."""
    data = s.data if data is None else data
    out: dict[str, Any] = {"model": split.model, "n_train": split.n_train, "n_test": split.n_test}
    if not split.has_row_identity or s.config.target not in data.columns:
        return out
    if not set(split.train_rows) | set(split.test_rows) <= set(data.index):
        out["note"] = "row labels do not match the data file, so rates and overlap were not computed"
        return out
    y = data[s.config.target]
    if y.nunique() == 2:
        positive = sorted(y.dropna().unique())[1]
        out["target_rate_train"] = round(float((y.loc[split.train_rows] == positive).mean()), 4)
        out["target_rate_test"] = round(float((y.loc[split.test_rows] == positive).mean()), 4)
        out["majority_class_accuracy_test"] = round(max(out["target_rate_test"], 1 - out["target_rate_test"]), 4)
    overlap = check_group_overlap(data, [split], s.config.group_columns, s.config.time_column)
    if overlap.evidence.get("splits"):
        detail = overlap.evidence["splits"][0]
        out["groups"] = [
            {k: g[k] for k in ("columns", "train_groups", "test_groups", "shared_groups")} for g in detail["groups"]
        ]
        if detail.get("time"):
            out["time"] = {k: detail["time"][k] for k in ("train_range", "test_range", "test_rows_before_train_end")}
    return out


def _propose_patch(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    path = args.get("path") or s.config.entry_script
    files = collect_files(s.config.project_dir)
    if path not in files:
        raise ToolArgumentError(f"no such file in the project: {path!r}")
    models = sorted({sp.model for sp in s.traced().splits})
    patch_id = f"P{len(s.patches) + 1}"
    base = None if args.get("fresh") else next(
        (p for p in reversed(list(s.patches.values())) if p.path == path), None)
    try:
        patch = make_patch(patch_id, path, files[path], args.get("edits") or [], str(args.get("rationale", "")),
                           models, base=base)
    except PatchError as exc:
        problems = exc.problems
        if base is not None:
            problems = problems + [f"edits are applied on top of {base.id} (the latest accepted patch of {path}); "
                                   "copy 'old' from that patched text, or pass fresh: true to start from the original"]
        return {"accepted": False, "problems": problems}
    s.patches[patch_id] = patch
    s.work_dir.mkdir(parents=True, exist_ok=True)
    (s.work_dir / f"patch_{patch_id}.diff").write_text(patch.diff, encoding="utf-8")
    return {
        "accepted": True,
        "patch_id": patch_id,
        "includes": _chain(s, patch),
        "lines_changed": patch.lines_changed,
        "protected_model_calls": models,
        "diff": patch.diff,
    }


def _run_patched(s: AuditSession, args: dict[str, Any]) -> dict[str, Any]:
    ids = args.get("patch_ids") or ([args["patch_id"]] if args.get("patch_id") else [])
    overrides, prepare, label = _patch_files(s, ids)
    ids = label.split("+")
    patches = [s.patches[i] for i in ids]
    traced = run_traced(s.config, s.sandbox, file_overrides=overrides, prepare_scripts=prepare)
    if traced.error:
        return {"patch_ids": ids, "error": f"patched run failed: {traced.error}",
                "stderr_tail": traced.run.stderr[-800:]}
    data = s.data
    rebuilt = traced.run.artifacts.get("prepared_data.csv")
    if rebuilt is not None:
        data = pd.read_csv(rebuilt)
    metric = s.config.reported_metric
    baseline = s.plain().metrics.get(metric)
    value = (traced.metrics or {}).get(metric)
    incomplete = traced.incomplete_reason
    entry = next((p for p in patches if p.path == s.config.entry_script), None)
    script = entry.patched if entry else _entry_script(s)
    return {
        "patch_id": label,
        "patch_ids": ids,
        "pipeline_rerun": prepare,
        "data_rebuilt": rebuilt is not None,
        "metrics": traced.metrics,
        "baseline_" + metric: baseline,
        "change_vs_baseline": round(value - baseline, 4) if isinstance(value, (int, float)) and baseline else None,
        "evaluations": [_describe_split(s, sp, data) for sp in traced.splits[:MAX_LIST_ITEMS]],
        "checks_after_patch": {
            "A": check_preprocessing(traced.events, traced.splits, incomplete, script).status,
            **_check_c_parts(check_group_overlap(
                data, traced.splits, s.config.group_columns, s.config.time_column, incomplete
            )),
            "D": check_shared_rows(data, traced.splits, s.config.time_column, incomplete).status,
            "E": check_selection(traced.splits, (), incomplete).status,
        },
        "log_dir": str(traced.run.log_dir),
    }


def _check_c_parts(result: CheckResult) -> dict[str, str]:
    """Check C's two questions answered separately: shared groups, and training on later data."""
    splits = [sp for sp in result.evidence.get("splits", []) if isinstance(sp, dict)]
    if result.status not in ("flagged", "clear") or not splits:
        return {"C": result.status}
    parts = {}
    if any(sp.get("groups") for sp in splits):
        shared = any(g["shared_groups"] > 0 for sp in splits for g in sp.get("groups", []))
        parts["C_groups"] = "flagged" if shared else "clear"
    if any(sp.get("time") for sp in splits):
        later = any(sp["time"]["test_rows_before_train_end"] > 0 for sp in splits if sp.get("time"))
        parts["C_time"] = "flagged" if later else "clear"
    return parts


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
    "propose_patch": _propose_patch,
    "run_patched": _run_patched,
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
        "Run one deterministic check. R: code review of EVERY .py file (data preparation too): suspicious lines "
        "with file and line, to verify. A: preprocessing fitted on test rows. B: features not known at prediction "
        "time (pass the columns you suspect, with reasons; code measures them with shuffle reruns under an honest "
        "split, or on top of your accepted fixes with on_patches). C: train/test overlap of groups and time. "
        "D: the same rows on both sides (copies from resampling, duplicated joins). E: models or thresholds "
        "chosen on the test rows.",
        {
            "check": {"type": "string", "enum": list(CHECKS)},
            "suspects": {
                "type": "array",
                "items": {"type": "object", "properties": {"column": {"type": "string"}, "reason": {"type": "string"}},
                          "required": ["column", "reason"]},
                "description": "Check B only: features you think would not be known at prediction time.",
            },
            "on_patches": {"type": "array", "items": {"type": "string"},
                           "description": "Check B only: measure on top of these accepted patches (an honest "
                                          "pipeline), so other leaks cannot hide the suspects."},
        },
        ["check"],
    ),
    _tool("run_ablation",
          "Shuffle each given column (max 3) and rerun; report the metric change. By default under an honest split; "
          "with on_patches, on top of those accepted fixes.",
          {"columns": {"type": "array", "items": {"type": "string"}},
           "on_patches": {"type": "array", "items": {"type": "string"}},
           "honest_split": {"type": "boolean"}}, ["columns"]),
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
        "propose_patch",
        "Propose a minimal fix to the evaluation or the data preparation (a split, a feature built from later "
        "rows, resampling before the split, statistics over all rows); the model and its settings must not change. "
        "Any .py file of the project can be patched: a patched preparation script (e.g. prepare.py) is rerun "
        "before training, so the fix reaches the data. One patch per file. Give exact text replacements: each "
        "'old' must be copied exactly from read_file output WITHOUT the line-number prefix and must occur once. "
        "Code applies them to a copy, checks the result compiles and leaves the model untouched, and returns the "
        "unified diff and a patch_id. A new patch of a file is applied ON TOP of that file's latest accepted "
        "patch (it includes it), so you can fix a file step by step; pass fresh: true to start from the original.",
        {
            "path": {"type": "string", "description": "File to patch; default is the entry script."},
            "edits": {
                "type": "array",
                "items": {"type": "object", "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
                          "required": ["old", "new"]},
            },
            "rationale": {"type": "string"},
            "fresh": {"type": "boolean", "description": "Start from the original file instead of its latest patch."},
        },
        ["edits", "rationale"],
    ),
    _tool(
        "run_patched",
        "Run the project with one or more accepted patches applied together (at most one per file), under tracing. "
        "Patched preparation scripts are rerun first, so the data is rebuilt. Returns its metrics, the change versus "
        "the baseline, target rates and majority baseline, and checks A, C, D and E on the patched run.",
        {"patch_ids": {"type": "array", "items": {"type": "string"}}},
        ["patch_ids"],
    ),
    _tool(
        "finalize",
        "Submit the audit. Every finding, suspicion and caveat cites evidence ids; every number must come from a "
        "tool result. confirmed needs a finding backed by a flagged check of its family; not_found needs checks "
        "A to E all run.",
        {
            "verdict": {"type": "string", "enum": ["confirmed", "not_found", "inconclusive"]},
            "summary": {"type": "string", "description": "Three or four sentences: what is wrong and what it means."},
            "findings": {
                "type": "array",
                "description": "Confirmed flaws, most severe first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "family": {"type": "string", "enum": list(FAMILIES)},
                        "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                        "title": {"type": "string"},
                        "explanation": {"type": "string", "description": "What happens, where (file and line), "
                                                                         "and why it breaks the evaluation."},
                        "impact": {"type": "string", "description": "The measured effect on the score, or 'not "
                                                                    "measured'."},
                        "evidence_ids": EVIDENCE_IDS,
                    },
                    "required": ["family", "severity", "title", "explanation", "impact", "evidence_ids"],
                },
            },
            "suspected": {
                "type": "array",
                "description": "Likely problems the evidence does not prove (e.g. from the code review).",
                "items": {"type": "object", "properties": {"title": {"type": "string"}, "explanation": {"type": "string"},
                                                           "evidence_ids": EVIDENCE_IDS},
                          "required": ["title", "explanation", "evidence_ids"]},
            },
            "score_assessment": {"type": "string", "description": "How far to trust the corrected score: what is "
                                                                  "still in it, and whether it is an upper bound."},
            "recommendations": {"type": "array", "items": {"type": "string"},
                                "description": "Concrete fixes for the author, most important first."},
            "caveats": {
                "type": "array",
                "items": {"type": "object", "properties": {"text": {"type": "string"}, "evidence_ids": EVIDENCE_IDS},
                          "required": ["text", "evidence_ids"]},
            },
        },
        ["verdict", "summary", "findings", "score_assessment", "caveats"],
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


def load_session(config: AuditConfig, sandbox: Sandbox, work_dir: Path) -> AuditSession:
    data = pd.read_csv(Path(config.project_dir) / config.data_path)
    return AuditSession(config=config, sandbox=sandbox, data=data, work_dir=Path(work_dir))
