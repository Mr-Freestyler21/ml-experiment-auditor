"""Command line entry point.

  python -m auditor.cli baseline demo/f1
  python -m auditor.cli check demo/f1            # rules + Meta model suspects (if a key is set)
  python -m auditor.cli check demo/f1 --no-llm   # rules only
  python -m auditor.cli audit demo/f1            # the Meta model investigates with the checks as tools
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from auditor.agent.llm_client import meta_client_from_env
from auditor.agent.loop import DEFAULT_MAX_STEPS, AgentResult, run_agent
from auditor.agent.tools import load_session
from auditor.baseline import TOLERANCE, BaselineResult, run_baseline
from auditor.checks.base import CheckResult
from auditor.checks.future_info import check_future_info, feature_power, llm_suspects, rules_suspects
from auditor.checks.group_overlap import check_group_overlap
from auditor.checks.preprocessing import check_preprocessing
from auditor.config import AuditConfig, ConfigError, load_config
from auditor.report import write_report
from auditor.sandbox import Sandbox, default_sandbox
from auditor.trace import TracedRun, compare_metrics, run_traced

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auditor.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", help="rerun a project and check its reported metric reproduces")
    baseline.add_argument("project", type=Path, help="folder containing audit.yaml")
    check = commands.add_parser("check", help="run the deterministic leakage checks on a project")
    check.add_argument("project", type=Path, help="folder containing audit.yaml")
    check.add_argument("--no-llm", action="store_true", help="rules only: do not ask the Meta model for suspects")
    audit = commands.add_parser("audit", help="full audit: the Meta model investigates using the checks as tools")
    audit.add_argument("project", type=Path, help="folder containing audit.yaml")
    audit.add_argument("--max-tokens", type=int, help="token cap for this audit (default: AUDITOR_MAX_TOKENS)")
    audit.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="maximum model turns")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.project)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    print("Preparing sandbox environment (first run installs packages)...")
    sandbox = default_sandbox(REPO_ROOT)
    if args.command == "baseline":
        return _baseline(config, sandbox)
    if args.command == "audit":
        return _audit(config, sandbox, args.max_tokens, args.max_steps)
    return _check(config, sandbox, use_llm=not args.no_llm)


def _audit(config: AuditConfig, sandbox: Sandbox, max_tokens: int | None, max_steps: int) -> int:
    client = meta_client_from_env(REPO_ROOT)
    if client is None:
        print("META_MODEL_API_KEY is not set (in .env or the environment).", file=sys.stderr)
        return 2
    if max_tokens:
        client.budget.cap = max_tokens
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = REPO_ROOT / "runs" / f"audit-{stamp}"
    print(f"Auditing {config.project_dir.name} with {client.model} (max {max_steps} steps, "
          f"{client.budget.cap} tokens). Transcript: {audit_dir}")
    session = load_session(config, sandbox, work_dir=audit_dir / "evidence")
    result = run_agent(session, client, audit_dir, max_steps=max_steps, on_event=_show_event)
    _print_verdict(result)
    report = write_report(session, result, client.model)
    print(f"Report: {report}")
    print(f"Open in a browser: {report.with_name('report.html')}")
    return 0 if result.stopped_because == "finalized" else 1


def _show_event(kind: str, data: dict) -> None:
    if kind == "model":
        if data["text"].strip():
            print(f"\n[{data['step']}] Muse: {data['text'].strip()[:400]}")
        for call in data["tool_calls"]:
            args = json.dumps(call["arguments"], default=str)
            print(f"[{data['step']}] -> {call['name']}({args[:160]})")
    elif kind == "tool":
        r = data["result"]
        brief = r.get("error") or r.get("summary") or r.get("status") or (
            f"accepted={r['accepted']} {r.get('problems', '')}" if "accepted" in r else "")
        if data["name"] in ("run_with_split", "run_patched") and "metrics" in r:
            brief = f"metrics={json.dumps(r['metrics'])[:200]} change={r.get('change_vs_baseline')}"
            if "checks_after_patch" in r:
                brief += f" checks after patch={r['checks_after_patch']}"
        if data["name"] == "propose_patch":
            brief = (f"accepted as {r['patch_id']} ({r['lines_changed']} lines)" if r.get("accepted")
                     else f"rejected: {r.get('problems')}")
        if data["name"] == "run_baseline":
            brief = f"{r.get('status')}: measured {r.get('measured')}"
        print(f"     {r.get('evidence_id', '')} {str(brief)[:300]}")
    elif kind == "stopped":
        print(f"\nStopped: {data['reason']}")


def _print_verdict(result: AgentResult) -> None:
    v = result.verdict
    print("\n" + "=" * 72)
    print(f"VERDICT: {v['verdict'].upper()}   ({result.stopped_because}, {result.steps} steps, {result.duration_s}s)")
    print(v["summary"])
    for f in v.get("findings", []):
        print(f"\n  Finding [{f['family']}] {f['title']}  {f['evidence_ids']}\n    {f['explanation']}")
    for c in v.get("caveats", []):
        print(f"\n  Caveat {c['evidence_ids']}: {c['text']}")
    u = result.usage
    print(f"\nModel usage: {u['calls']} calls, {u['tokens']} tokens, est. ${u['est_cost_usd']:.4f}")
    print(f"Evidence and transcript: {result.audit_dir}")


def _baseline(config: AuditConfig, sandbox: Sandbox) -> int:
    print(f"Running {config.entry_script} in the sandbox (timeout {config.timeout_seconds}s)...")
    result = run_baseline(config, sandbox)
    _print_baseline(result)
    return 0 if result.status == "reproduced" else 1


def _check(config: AuditConfig, sandbox: Sandbox, use_llm: bool) -> int:
    print(f"Running {config.entry_script} in the sandbox as-is...")
    plain = run_baseline(config, sandbox)
    if plain.status == "failed":
        print(f"Run failed: {plain.error}. See {plain.log_dir}", file=sys.stderr)
        return 1
    print(f"Running {config.entry_script} again with call tracing...")
    traced = run_traced(config, sandbox)
    if traced.error:
        print(f"Run failed: {traced.error}. See {traced.run.log_dir}", file=sys.stderr)
        return 1

    same = compare_metrics(plain.metrics, traced.metrics)
    print(f"Tracing changed the results: {'no' if same['match'] else 'YES ' + json.dumps(same['differences'])}")
    print(f"Traced {len(traced.events)} calls; found {len(traced.splits)} train/test evaluation(s).")
    for split in traced.splits:
        print(
            f"  {split.model}: fit at line {split.fit_line} on {split.n_train} rows, "
            f"{'/'.join(split.predict_methods)} at line {split.predict_line} on {split.n_test} rows"
        )
    incomplete = traced.incomplete_reason
    if not same["match"]:
        incomplete = "the traced run's metrics differ from the plain run, so the trace may not describe the same experiment"

    data = pd.read_csv(config.project_dir / config.data_path)
    results = [
        check_preprocessing(traced.events, traced.splits, incomplete, _script(config)),
        _check_b(config, sandbox, data, traced, plain.metrics, use_llm),
        check_group_overlap(data, traced.splits, config.group_columns, config.time_column, incomplete),
    ]
    (traced.run.log_dir / "metrics_comparison.json").write_text(json.dumps(same, indent=2), encoding="utf-8")
    for result in results:
        (traced.run.log_dir / f"check_{result.check_id}.json").write_text(
            json.dumps(result.to_json(), indent=2, default=str), encoding="utf-8"
        )
        _print_check(result)
    print()
    print(f"Evidence:  {traced.run.log_dir}")
    return 0


def _check_b(
    config: AuditConfig, sandbox: Sandbox, data: pd.DataFrame, traced: TracedRun, baseline: dict, use_llm: bool
) -> CheckResult:
    features = next((list(s.features) for s in traced.splits if s.features), None)
    if not features:
        return check_future_info(config, sandbox, data, None, baseline, [])
    power = feature_power(data, [f for f in features if f in data.columns], config.target)
    suspects = rules_suspects(features, config.target, power)
    notes: dict = {"rules_suspects": [s.column for s in suspects]}

    client = meta_client_from_env(REPO_ROOT) if use_llm else None
    if use_llm and client is None:
        print("No META_MODEL_API_KEY set: Check B uses rules only.")
    if client is not None:
        print(f"Asking {client.model} which features would not be known at prediction time...")
        llm, llm_notes = llm_suspects(client, config, features, data)
        suspects += llm
        notes["llm"] = {**llm_notes, "suspects": [s.column for s in llm], "budget": client.budget.summary()}
        print(f"  {client.model} suspects: {[s.column for s in llm] or 'none'}  ({client.budget.summary()})")
    if suspects:
        print(f"Shuffle-rerunning suspected columns: {sorted({s.column for s in suspects})}")
    return check_future_info(config, sandbox, data, features, baseline, suspects, notes)


def _script(config: AuditConfig) -> str:
    return (config.project_dir / config.entry_script).read_text(encoding="utf-8")


def _print_baseline(result: BaselineResult) -> None:
    print()
    print(f"Status:    {result.status}")
    if result.error:
        print(f"Error:     {result.error}")
    else:
        print(
            f"{result.reported_metric}: reported {result.reported_value}, measured {result.measured_value:.4f} "
            f"(difference {result.difference:+.4f}, tolerance {TOLERANCE})"
        )
        for name, value in result.metrics.items():
            if name != result.reported_metric:
                print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    env = result.environment
    packages = ", ".join(f"{k} {v}" for k, v in env.get("packages", {}).items())
    print(f"Seed:      {result.random_seed}")
    print(f"Python:    {env.get('python')} ({packages})")
    print(f"Duration:  {result.duration_s:.1f}s")
    print(f"Evidence:  {result.log_dir}")


def _print_check(result: CheckResult) -> None:
    print()
    print(f"[{result.check_id}] {result.status.upper()}")
    print(f"  {result.summary}")
    for split in result.evidence.get("splits", [])[:1]:
        for g in split.get("groups", []):
            for ex in g["examples"][:3]:
                print(f"    e.g. {ex['group']}: {ex['train_rows']} train rows, {ex['test_rows']} test rows")
    for leak in result.evidence.get("leaks", [])[1:4]:
        print(f"    also: {leak['class']}.{leak['method']} at line {leak['line']} ({leak['kind']})")
    for rerun in result.evidence.get("shuffle_reruns", []):
        print(
            f"    shuffled '{rerun['column']}' ({'/'.join(rerun['sources'])}): "
            f"{rerun['baseline']} -> {rerun['shuffled']} (drop {rerun['metric_drop']}) {rerun['error'] or ''}"
        )


if __name__ == "__main__":
    sys.exit(main())
