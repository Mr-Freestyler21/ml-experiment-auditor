"""The three contestants, each auditing one project folder.

  rules     The deterministic checks with fixed thresholds. No model.
  agent     Muse investigates with the checks as tools (the full auditor).
  llm_only  Muse reads the code, config and a data sample and answers in one go,
            with no tools and no measurements.

Each returns the families it flags (A, B, C) plus details. A system "flags" a family
only when it commits to it: a flagged check (rules), a confirmed finding (agent), or a
listed flaw (llm_only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from auditor.agent.llm_client import LLMClient
from auditor.agent.loop import run_agent
from auditor.agent.tools import load_session
from auditor.baseline import run_baseline
from auditor.checks.future_info import check_future_info, feature_power, rules_suspects
from auditor.checks.group_overlap import check_group_overlap
from auditor.checks.preprocessing import check_preprocessing
from auditor.config import AuditConfig
from auditor.report import write_report
from auditor.sandbox import Sandbox
from auditor.trace import compare_metrics, run_traced


def run_rules(config: AuditConfig, sandbox: Sandbox) -> dict[str, Any]:
    start = time.perf_counter()
    plain = run_baseline(config, sandbox)
    traced = run_traced(config, sandbox)
    if traced.error or plain.status == "failed":
        return {"error": traced.error or plain.error, "families": [], "duration_s": time.perf_counter() - start}
    incomplete = traced.incomplete_reason
    if not compare_metrics(plain.metrics, traced.metrics)["match"]:
        incomplete = "traced metrics differ from the plain run"
    data = pd.read_csv(config.project_dir / config.data_path)
    features = next((list(s.features) for s in traced.splits if s.features), None)
    suspects = []
    if features:
        power = feature_power(data, [f for f in features if f in data.columns], config.target)
        suspects = rules_suspects(features, config.target, power)
    checks = {
        "A": check_preprocessing(traced.events, traced.splits, incomplete,
                                 (config.project_dir / config.entry_script).read_text(encoding="utf-8")),
        "B": check_future_info(config, sandbox, data, features, plain.metrics, suspects),
        "C": check_group_overlap(data, traced.splits, config.group_columns, config.time_column, incomplete),
    }
    return {
        "families": sorted(f for f, r in checks.items() if r.status == "flagged"),
        "checks": {f: {"status": r.status, "summary": r.summary} for f, r in checks.items()},
        "duration_s": round(time.perf_counter() - start, 1),
        "cost_usd": 0.0,
    }


def run_agent_system(config: AuditConfig, sandbox: Sandbox, client: LLMClient, audit_dir: Path) -> dict[str, Any]:
    session = load_session(config, sandbox, work_dir=audit_dir / "evidence")
    result = run_agent(session, client, audit_dir)
    write_report(session, result, client.model)
    verdict = result.verdict
    patched = [e for e in session.evidence if e.tool == "run_patched" and not e.result.get("error")]
    patched_metric = (patched[-1].result.get("metrics") or {}).get(config.reported_metric) if patched else None
    return {
        "verdict": verdict["verdict"],
        "families": sorted({f["family"] for f in verdict.get("findings", [])}) if verdict["verdict"] == "confirmed" else [],
        "stopped_because": result.stopped_because,
        "steps": result.steps,
        "patched_metric": patched_metric,
        "patches_proposed": sum(1 for e in session.evidence if e.tool == "propose_patch"),
        "summary": verdict["summary"],
        "duration_s": result.duration_s,
        "tokens": result.usage["tokens"],
        "cost_usd": result.usage["est_cost_usd"],
        "audit_dir": str(audit_dir),
    }


LLM_ONLY_TOOL = "report_audit"
LLM_ONLY_INSTRUCTIONS = """You audit how a machine learning experiment was evaluated. You cannot run code or
measure anything: decide from the files shown. Flaw families:
A) preprocessing (scaling, imputation, feature selection, ...) fitted on rows that include the test set;
B) features whose values would not be known at prediction time, or that are derived from the target;
C) the same entities (e.g. patients) or later time periods in both train and test, when real use is on new
   entities or future periods.
Verdict: confirmed (at least one flaw), not_found, or inconclusive. Also estimate the accuracy a correct
evaluation would give. Answer only by calling the report_audit tool."""


def run_llm_only(config: AuditConfig, client: LLMClient) -> dict[str, Any]:
    start = time.perf_counter()
    script = (config.project_dir / config.entry_script).read_text(encoding="utf-8")
    yaml_text = (config.project_dir / "audit.yaml").read_text(encoding="utf-8")
    head = "\n".join((config.project_dir / config.data_path).read_text(encoding="utf-8").splitlines()[:6])
    tool = {
        "type": "function",
        "name": LLM_ONLY_TOOL,
        "description": "Report the audit result.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["confirmed", "not_found", "inconclusive"]},
                "flaws": {"type": "array", "items": {"type": "object", "properties": {
                    "family": {"type": "string", "enum": ["A", "B", "C"]}, "explanation": {"type": "string"}},
                    "required": ["family", "explanation"]}},
                "estimated_correct_accuracy": {"type": "number"},
            },
            "required": ["verdict", "flaws", "estimated_correct_accuracy"],
        },
    }
    prompt = f"audit.yaml:\n{yaml_text}\n\n{config.entry_script}:\n{script}\n\nFirst rows of {config.data_path}:\n{head}"
    response = client.respond(instructions=LLM_ONLY_INSTRUCTIONS, input=prompt, tools=[tool],
                              reasoning_effort="medium", purpose="llm_only")
    call = next((c for c in response.tool_calls if c.name == LLM_ONLY_TOOL and c.arguments), None)
    if call is None:
        return {"error": "no report_audit call", "families": [], "text": response.text[:500],
                "duration_s": round(time.perf_counter() - start, 1), "cost_usd": client.budget.cost_usd()}
    args = call.arguments
    flaws = [f for f in args.get("flaws", []) if isinstance(f, dict) and f.get("family") in ("A", "B", "C")]
    verdict = args.get("verdict")
    return {
        "verdict": verdict,
        "families": sorted({f["family"] for f in flaws}) if verdict == "confirmed" else [],
        "flaws": flaws,
        "estimated_correct_metric": args.get("estimated_correct_accuracy"),
        "duration_s": round(time.perf_counter() - start, 1),
        "tokens": client.budget.used,
        "cost_usd": client.budget.cost_usd(),
    }


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
