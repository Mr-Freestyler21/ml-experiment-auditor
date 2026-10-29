"""Write report.md and report.json from an audit's evidence.

Code fills in every table and number from evidence items. The only text from the model
is its verdict (summary, findings, caveats), which passed validation before it was
accepted: each claim cites evidence ids and every number in it appears in a tool result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from auditor.agent.loop import AgentResult
from auditor.agent.tools import AuditSession, EvidenceItem
from auditor.report_html import render_html

FAMILY_NAMES = {
    "A": "Preprocessing fitted on test rows",
    "B": "Information not available at prediction time",
    "C": "Same entities or later periods in train and test",
}


def write_report(session: AuditSession, result: AgentResult, model: str) -> Path:
    data = build_report(session, result, model)
    (result.audit_dir / "report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    path = result.audit_dir / "report.md"
    path.write_text(render_markdown(data, result.audit_dir), encoding="utf-8")
    (result.audit_dir / "report.html").write_text(render_html(json.loads(json.dumps(data, default=str))),
                                                  encoding="utf-8")
    return path


def build_report(session: AuditSession, result: AgentResult, model: str) -> dict[str, Any]:
    config = session.config
    verdict = result.verdict
    by_id = session.evidence_by_id()
    script_lines = (config.project_dir / config.entry_script).read_text(encoding="utf-8").splitlines()
    patch_runs = [e for e in session.evidence if e.tool == "run_patched" and not e.result.get("error")]
    patch = session.patches.get(patch_runs[-1].result["patch_id"]) if patch_runs else None

    return {
        "project": config.project_dir.name,
        "verdict": verdict["verdict"],
        "summary": verdict["summary"],
        "stopped_because": result.stopped_because,
        "baseline": _baseline(session),
        "context_metrics": _context_metrics(session),
        "findings": [_finding(f, by_id, session, script_lines) for f in verdict.get("findings", [])],
        "patch": {
            "id": patch.id, "path": patch.path, "rationale": patch.rationale,
            "lines_changed": patch.lines_changed, "diff": patch.diff,
        } if patch else None,
        "evaluations": _evaluations(session),
        "caveats": verdict.get("caveats", []),
        "limits": _limits(session, result),
        "cost": {"model": model, "steps": result.steps, "duration_s": result.duration_s, **result.usage},
        "evidence_index": [{"id": e.id, "tool": e.tool, "arguments": e.arguments, "brief": _brief(e)}
                           for e in session.evidence],
    }


# --- sections ---------------------------------------------------------------------------


def _baseline(session: AuditSession) -> dict[str, Any] | None:
    b = session._plain
    if b is None:
        return None
    return {
        "metric": b.reported_metric,
        "reported": b.reported_value,
        "measured": b.measured_value,
        "difference": b.difference,
        "status": b.status,
        "random_seed": b.random_seed,
        "environment": b.environment,
        "log_dir": str(b.log_dir),
    }


def _context_metrics(session: AuditSession) -> dict[str, Any]:
    """Imbalance-aware metrics the script reported, plus the majority baseline on its test rows."""
    metrics = dict(session._plain.metrics) if session._plain else {}
    out = {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "majority_class_accuracy")
           if isinstance(metrics.get(k), (int, float))}
    traced = session._traced
    y = session.data[session.config.target]
    if traced and traced.splits and traced.splits[0].test_rows is not None and y.nunique() == 2:
        positive = sorted(y.dropna().unique())[1]
        rate = float((y.loc[traced.splits[0].test_rows] == positive).mean())
        out["test_positive_rate (auditor)"] = round(rate, 4)
        out["majority_class_accuracy (auditor)"] = round(max(rate, 1 - rate), 4)
    return out


def _finding(f: dict[str, Any], by_id: dict[str, EvidenceItem], session: AuditSession, lines: list[str]) -> dict[str, Any]:
    cited = [by_id[i] for i in f.get("evidence_ids", []) if i in by_id]
    code = {}
    for e in cited:
        if e.tool != "run_check":
            continue
        for split in e.result.get("evidence", {}).get("splits", []) or []:
            if isinstance(split, dict):
                code.setdefault(split.get("fit_line"), "model fitted on the training rows")
                code.setdefault(split.get("predict_line"), "model evaluated on the test rows")
        for leak in e.result.get("evidence", {}).get("leaks", []) or []:
            if isinstance(leak, dict):
                code.setdefault(leak.get("line"), f"{leak.get('class')}.{leak.get('method')} fitted before the split")
    if f["family"] == "C" and session._traced:
        for event in session._traced.events:
            if event.get("event") == "split":
                code.setdefault(event.get("line"), "rows assigned to train and test")
    quoted = [
        {"line": n, "role": role, "code": lines[n - 1].strip()}
        for n, role in sorted((k, v) for k, v in code.items() if isinstance(k, int) and 0 < k <= len(lines))
    ]
    return {
        **f,
        "family_name": FAMILY_NAMES.get(f["family"], f["family"]),
        "code": quoted,
        "evidence": [{"id": e.id, "tool": e.tool, "brief": _brief(e)} for e in cited],
    }


def _evaluations(session: AuditSession) -> list[dict[str, Any]]:
    metric = session.config.reported_metric
    rows = []
    for e in session.evidence:
        r = e.result
        if r.get("error"):
            continue
        if e.tool == "run_baseline":
            label, value, change = "Original evaluation (rerun)", r.get("measured"), 0.0
            ev = {}
            majority = (r.get("metrics") or {}).get("majority_class_accuracy")
        elif e.tool in ("run_with_split", "run_patched"):
            value, change = (r.get("metrics") or {}).get(metric), r.get("change_vs_baseline")
            ev = (r.get("evaluations") or [{}])[0] if r.get("evaluations") else {}
            majority = ev.get("majority_class_accuracy_test")
            label = f"Patched script ({r['patch_id']})" if e.tool == "run_patched" else _split_label(e.arguments)
        else:
            continue
        groups = ev.get("groups") or []
        rows.append({
            "label": label,
            "evidence_id": e.id,
            metric: value,
            "change_vs_baseline": change,
            "majority_class_accuracy_test": majority,
            "target_rate_train": ev.get("target_rate_train"),
            "target_rate_test": ev.get("target_rate_test"),
            "shared_groups": groups[0].get("shared_groups") if groups else None,
            "checks_after_patch": r.get("checks_after_patch"),
            "log_dir": r.get("log_dir"),
        })
    return rows


def _split_label(args: dict[str, Any]) -> str:
    """'Group split by Year + Race, only Year = 2024' from run_with_split arguments."""
    strategy = args.get("strategy", "?")
    label = {"random": "Random split", "group": "Group split", "temporal": "Temporal split"}.get(strategy, strategy)
    if strategy == "group" and args.get("group_columns"):
        label += " by " + " + ".join(args["group_columns"])
    if strategy == "temporal" and args.get("time_column"):
        label += " on " + args["time_column"]
    restrict = args.get("restrict_to")
    if isinstance(restrict, dict):
        label += f", only {restrict.get('column')} = {restrict.get('value')}"
    return label


def _limits(session: AuditSession, result: AgentResult) -> list[str]:
    limits = []
    if result.stopped_because != "finalized":
        limits.append(f"The audit stopped before the model finalized ({result.stopped_because}); the verdict is "
                      "therefore inconclusive.")
    ran = {e.arguments.get("check"): e for e in session.evidence if e.tool == "run_check"}
    for family in ("A", "B", "C"):
        e = ran.get(family)
        if e is None:
            limits.append(f"Check {family} ({FAMILY_NAMES[family]}) was not run.")
        elif e.result.get("status") in ("not_applicable", "error") or e.result.get("error"):
            limits.append(f"Check {family}: {e.result.get('status', 'error')}: "
                          f"{e.result.get('summary') or e.result.get('error')} ({e.id})")
    if not session.patches and result.verdict.get("verdict") == "confirmed":
        limits.append("No patch was accepted, so the corrected evaluation comes from split reruns only.")
    return limits


# --- markdown ---------------------------------------------------------------------------


def render_markdown(r: dict[str, Any], audit_dir: Path) -> str:
    out: list[str] = []
    add = out.append
    metric = r["baseline"]["metric"] if r["baseline"] else "metric"
    add(f"# Audit report: {r['project']}\n")
    add(f"## 1. Verdict: **{r['verdict'].upper()}**\n")
    add(r["summary"] + "\n")

    add("## 2. Reproduced baseline\n")
    b = r["baseline"]
    if b:
        env = b["environment"] or {}
        packages = ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())
        add(f"| | |\n|---|---|\n| Reported {metric} | {_fmt(b['reported'])} |\n| Measured {metric} | {_fmt(b['measured'])} "
            f"({b['status']}, difference {_fmt(b['difference'], signed=True)}) |\n| Seed | {b['random_seed']} |\n"
            f"| Environment | Python {env.get('python')}; {packages} |\n| Run log | {_link(b['log_dir'], audit_dir)} |\n")
    else:
        add("The baseline was not run.\n")

    add("## 3. Context metrics\n")
    if r["context_metrics"]:
        add("| Metric | Value |\n|---|---|")
        for k, v in r["context_metrics"].items():
            add(f"| {k} | {_fmt(v)} |")
        add("\nAccuracy should be read against the majority-class baseline: always predicting the most common class "
            "scores that much without learning anything.\n")

    add("## 4. Findings\n")
    if not r["findings"]:
        add("No evaluation flaw was confirmed.\n")
    for i, f in enumerate(r["findings"], 1):
        add(f"### 4.{i} [{f['family']}] {f['title']}\n")
        add(f"*{f['family_name']}*\n")
        add(f["explanation"] + "\n")
        if f["code"]:
            add("Offending code (`" + "`, `".join(sorted({str(c['line']) for c in f['code']})) + "`):\n")
            add("```python")
            for c in f["code"]:
                add(f"{c['line']:>4}  {c['code']}    # {c['role']}")
            add("```\n")
        add("Evidence: " + "; ".join(f"**{e['id']}** {e['tool']}: {e['brief']}" for e in f["evidence"]) + "\n")

    add("## 5. Patch\n")
    p = r["patch"]
    if p:
        add(f"**{p['id']}** on `{p['path']}` ({p['lines_changed']} lines). {p['rationale']}\n")
        add("```diff\n" + p["diff"].rstrip() + "\n```\n")
        add(f"The original file is unchanged; the patch was applied to a copy (`evidence/patch_{p['id']}.diff`).\n")
    else:
        add("No patch was applied.\n")

    add("## 6. Corrected evaluation\n")
    if r["evaluations"]:
        add(f"| Evaluation | {metric} | Change vs baseline | Majority baseline (test) | Target rate train / test "
            "| Shared groups | Checks after patch | Evidence |")
        add("|---|---|---|---|---|---|---|---|")
        for e in r["evaluations"]:
            checks = ", ".join(f"{k}: {v}" for k, v in (e["checks_after_patch"] or {}).items()) or ""
            rates = (f"{_fmt(e['target_rate_train'])} / {_fmt(e['target_rate_test'])}"
                     if e["target_rate_test"] is not None else "")
            add(f"| {e['label']} | {_fmt(e.get(metric))} | {_fmt(e['change_vs_baseline'], signed=True)} | "
                f"{_fmt(e['majority_class_accuracy_test'])} | {rates} | "
                f"{'' if e['shared_groups'] is None else e['shared_groups']} | {checks} | {e['evidence_id']} |")
        add("")
    else:
        add("No re-evaluation was run.\n")

    add("## 7. Limits and caveats\n")
    for c in r["caveats"]:
        add(f"- {c['text']} ({', '.join(c['evidence_ids'])})")
    for line in r["limits"]:
        add(f"- {line}")
    if not r["caveats"] and not r["limits"]:
        add("- None recorded.")
    add("")

    add("## 8. Cost and time\n")
    c = r["cost"]
    add(f"Model `{c['model']}`: {c['calls']} calls, {c['tokens']} tokens (cap {c['cap']}), estimated "
        f"${c['est_cost_usd']:.4f}. {c['steps']} agent steps in {c['duration_s']} s "
        f"(stopped: {r['stopped_because']}).\n")

    add("## Evidence index\n")
    add("| Id | Tool | Arguments | Result |\n|---|---|---|---|")
    for e in r["evidence_index"]:
        args = json.dumps(e["arguments"], default=str)
        add(f"| {e['id']} | {e['tool']} | `{_cell(args, 80)}` | {_cell(e['brief'], 160)} |")
    add("\nFull evidence, the transcript and the verdict are in `evidence/`.")
    return "\n".join(out) + "\n"


def _brief(e: EvidenceItem) -> str:
    r = e.result
    if r.get("error"):
        return f"error: {r['error']}"
    if e.tool == "run_check":
        return f"{r.get('status')}: {r.get('summary', '')}"
    if e.tool == "run_baseline":
        return f"{r.get('status')}, measured {_fmt(r.get('measured'))}"
    if e.tool in ("run_with_split", "run_patched"):
        metrics = r.get("metrics") or {}
        first = next(iter(metrics.items()), (None, None))
        extra = f", checks after patch {r['checks_after_patch']}" if "checks_after_patch" in r else ""
        return f"{first[0]} {_fmt(first[1])} (change {_fmt(r.get('change_vs_baseline'), signed=True)}){extra}"
    if e.tool == "propose_patch":
        return f"accepted as {r['patch_id']}" if r.get("accepted") else f"rejected: {r.get('problems')}"
    if e.tool == "run_ablation":
        return "; ".join(f"{x['column']}: change {_fmt(x.get('change'), signed=True)}" for x in r.get("results", []))
    if e.tool == "read_file":
        return f"read {r.get('path')}"
    if e.tool == "list_files":
        return f"{len(r.get('files', []))} files"
    return ""


def _fmt(value: Any, signed: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.4f}" if signed else f"{value:.4f}"
    return str(value)


def _cell(text: str, limit: int) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _link(path: str | None, base: Path) -> str:
    if not path:
        return ""
    rel = os.path.relpath(path, base).replace("\\", "/")
    return f"[{rel}]({rel})"
