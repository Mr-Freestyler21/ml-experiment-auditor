"""The demo in one command, with narration that reads well on screen.

    python -m auditor.cli demo            # live: Muse audits demo/f1, then demo/clean (~4 min, ~$0.07)
    python -m auditor.cli demo --replay   # replays the recorded audits in examples/ (no API key needed)

Sequence: reproduce the F1 project's 95%, expose the leakage, isolate it from the label
shift, patch the evaluation, rerun it, then audit an honest project and show it is not
falsely accused. Both HTML reports open in the browser at the end.
"""

from __future__ import annotations

import json
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auditor.agent.llm_client import meta_client_from_env
from auditor.agent.loop import run_agent
from auditor.agent.tools import load_session
from auditor.config import load_config
from auditor.report import write_report
from auditor.sandbox import default_sandbox

WIDTH = 78


@dataclass(frozen=True)
class DemoProject:
    title: str
    folder: str  # project folder, relative to the repo
    example: str  # recorded audit, relative to the repo
    story: str


PROJECTS = [
    DemoProject(
        "F1 pit-stop prediction",
        "demo/f1",
        "examples/f1",
        "A Random Forest predicts whether a driver pits on the next lap. Reported accuracy: 95%. "
        "Evaluated with a random 80/20 split of 52,455 laps.",
    ),
    DemoProject(
        "Hospital readmission (honest)",
        "demo/clean",
        "examples/clean",
        "A model predicts 30-day readmission. Patients are kept apart with a grouped split, and one "
        "feature, prior readmissions, is strongly predictive and known at discharge.",
    ),
]


def run_demo(repo: Path, replay: bool, open_reports: bool, delay: float) -> int:
    # Model text can hold characters a Windows console code page cannot print; never crash on them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if not replay and meta_client_from_env(repo) is None:
        print("No META_MODEL_API_KEY found, so the live demo cannot run. Replaying the recorded audits instead.\n")
        replay = True
    _banner("ML Experiment Auditor", "Audits how an ML experiment was evaluated. The model reasons, code measures.")
    if replay:
        print("REPLAY of recorded audits from examples/ (same output as a live run, no API calls).\n")

    summaries = []
    for n, project in enumerate(PROJECTS, 1):
        _section(f"{n}. {project.title}", project.story)
        if replay:
            report_dir = repo / project.example
            _replay(report_dir / "evidence" / "transcript.jsonl", delay)
        else:
            report_dir = _live(repo, project)
            if report_dir is None:
                return 1
        data = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
        _verdict(data)
        summaries.append((project, data, report_dir))

    _summary(summaries)
    if open_reports:
        for _, _, report_dir in summaries:
            webbrowser.open((report_dir / "report.html").resolve().as_uri())
    return 0


# --- live and replay --------------------------------------------------------------------


def _live(repo: Path, project: DemoProject) -> Path | None:
    client = meta_client_from_env(repo)
    config = load_config(repo / project.folder)
    sandbox = default_sandbox(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = repo / "runs" / f"audit-{stamp}"
    session = load_session(config, sandbox, work_dir=audit_dir / "evidence")
    result = run_agent(session, client, audit_dir, on_event=narrate)
    write_report(session, result, client.model)
    if result.stopped_because != "finalized":
        print(f"\nThe audit stopped early ({result.stopped_because}). Try again, or use --replay.")
        return None
    return audit_dir


def _replay(transcript: Path, delay: float) -> None:
    for line in transcript.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        narrate(event["kind"], event)
        if event["kind"] in ("model", "tool"):
            time.sleep(delay)


# --- narration --------------------------------------------------------------------------


def narrate(kind: str, data: dict[str, Any]) -> None:
    """One readable line per model message, tool request and tool result."""
    if kind == "model":
        if data.get("text", "").strip():
            print(f"\n  Muse: {_clip(data['text'].strip(), 200)}")
        for call in data.get("tool_calls", []):
            print(f"    -> {call['name']}{_args(call['name'], call.get('arguments'))}")
    elif kind == "tool":
        line = describe_result(data["name"], data.get("result") or {})
        if line:
            print(f"       {line}")
    elif kind == "stopped":
        print(f"\n  Stopped: {data.get('reason')}")
    sys.stdout.flush()


def _args(name: str, args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    if name == "read_file":
        return f" {args.get('path')}"
    if name == "run_check":
        suspects = [s.get("column") for s in args.get("suspects") or [] if isinstance(s, dict)]
        return f" {args.get('check')}" + (f" (Muse suspects: {', '.join(suspects)})" if suspects else "")
    if name == "run_with_split":
        parts = [args.get("strategy", "?")]
        if args.get("group_columns"):
            parts.append("by " + " + ".join(args["group_columns"]))
        if isinstance(args.get("restrict_to"), dict):
            parts.append(f"only {args['restrict_to'].get('column')} = {args['restrict_to'].get('value')}")
        return " (" + ", ".join(parts) + ")"
    if name in ("propose_patch", "finalize"):
        return ""
    return f" {_clip(json.dumps(args), 80)}"


def describe_result(name: str, r: dict[str, Any]) -> str:
    eid = r.get("evidence_id", "")
    if r.get("error"):
        return f"{eid} error: {_clip(str(r['error']), 160)}"
    if name == "run_baseline":
        reported = next(iter((r.get("reported") or {}).values()), None)
        return f"{eid} reproduced {_n(r.get('measured'))} (reported {_n(reported)})"
    if name == "run_check":
        return f"{eid} {str(r.get('status', '')).upper()}: {_clip(r.get('summary', ''), 170)}"
    if name in ("run_with_split", "run_patched"):
        metrics = r.get("metrics") or {}
        metric, value = next(iter(metrics.items()), (None, None))
        evaluation = (r.get("evaluations") or [{}])[0]
        text = f"{eid} {metric} {_n(value)} (change {_signed(r.get('change_vs_baseline'))})"
        if evaluation.get("majority_class_accuracy_test") is not None:
            text += f", majority baseline {_n(evaluation['majority_class_accuracy_test'])}"
        if r.get("checks_after_patch"):
            text += ", checks after fix: " + ", ".join(f"{k} {v}" for k, v in r["checks_after_patch"].items())
        return text
    if name == "propose_patch":
        if r.get("accepted"):
            return f"{eid} fix {r['patch_id']} accepted ({r['lines_changed']} lines; model untouched)"
        return f"{eid} fix rejected: {_clip(str(r.get('problems')), 160)}"
    if name == "finalize":
        return "verdict accepted" if r.get("accepted") else f"verdict rejected: {_clip(str(r.get('problems')), 160)}"
    if name == "run_ablation":
        return f"{eid} " + "; ".join(f"{x['column']} {_signed(x.get('change'))}" for x in r.get("results", []))
    return ""


def _verdict(data: dict[str, Any]) -> None:
    metric = (data.get("baseline") or {}).get("metric", "metric")
    print("\n" + "-" * WIDTH)
    print(f"  VERDICT: {data['verdict'].replace('_', ' ').upper()}")
    print(_wrap(data["summary"], "  "))
    for f in data.get("findings", []):
        print(f"\n  Finding [{f['family']}] {f['title']}")
        for c in f.get("code", [])[:3]:
            print(f"    line {c['line']:>3}: {_clip(c['code'], 60)}   ({c['role']})")
    rows = [e for e in data.get("evaluations", []) if isinstance(e.get(metric), (int, float))]
    if rows:
        print(f"\n  {'Evaluation':44s} {metric:>9s}  {'majority':>9s}")
        for e in rows:
            majority = e.get("majority_class_accuracy_test")
            print(f"  {_clip(e['label'], 44):44s} {e[metric]:9.4f}  {_n(majority):>9s}")
    cost = data.get("cost") or {}
    print(f"\n  {cost.get('steps')} steps, {cost.get('duration_s')} s, about ${float(cost.get('est_cost_usd') or 0):.3f}")
    print("-" * WIDTH)


def _summary(summaries: list[tuple[DemoProject, dict[str, Any], Path]]) -> None:
    _banner("Summary", "")
    print(f"  {'Project':32s} {'Verdict':14s} {'Reported':>9s} {'Corrected':>10s}")
    for project, data, _ in summaries:
        metric = (data.get("baseline") or {}).get("metric", "metric")
        patched = [e for e in data.get("evaluations", []) if e["label"].startswith("Patched")]
        corrected = _n(patched[-1].get(metric)) if patched else "-"
        print(f"  {project.title:32s} {data['verdict'].replace('_', ' '):14s} "
              f"{_n((data.get('baseline') or {}).get('reported')):>9s} {corrected:>10s}")
    print("\n  Full reports (HTML, Markdown, evidence):")
    for _, _, report_dir in summaries:
        print(f"    {report_dir / 'report.html'}")
    print()


# --- formatting -------------------------------------------------------------------------


def _banner(title: str, subtitle: str) -> None:
    print("=" * WIDTH)
    print(f"  {title}")
    if subtitle:
        print(_wrap(subtitle, "  "))
    print("=" * WIDTH + "\n")


def _section(title: str, story: str) -> None:
    print("\n" + "#" * WIDTH)
    print(f"  {title}")
    print(_wrap(story, "  "))
    print("#" * WIDTH)


def _wrap(text: str, indent: str) -> str:
    import textwrap

    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _n(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else ("-" if value is None else str(value))


def _signed(value: Any) -> str:
    return f"{value:+.4f}" if isinstance(value, (int, float)) else "-"
