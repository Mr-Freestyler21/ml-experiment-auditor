"""The agent loop: the model asks for tools, our code runs them, until it finalizes.

    send task + tool menu  ->  model replies with tool calls (or text)
            ^                              |
            |                              v
    send tool results  <-  our code validates arguments and runs each tool

It stops when finalize is accepted, or when a limit is hit (steps, tokens). Hitting a
limit ends with an honest "inconclusive" verdict, never a guess.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auditor.agent import prompts
from auditor.agent.llm_client import BudgetExceeded, LLMClient
from auditor.agent.tools import TOOL_SCHEMAS, AuditSession, to_json_text

DEFAULT_MAX_STEPS = 20


@dataclass
class AgentResult:
    verdict: dict[str, Any]
    stopped_because: str  # "finalized", "max_steps", "token_budget", "model_error"
    steps: int
    duration_s: float
    usage: dict[str, Any]
    audit_dir: Path
    evidence: list[dict[str, Any]] = field(default_factory=list)


def run_agent(
    session: AuditSession,
    client: LLMClient,
    audit_dir: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_event: Any = None,
) -> AgentResult:
    """Run one audit. `on_event(kind, data)` is called for progress output (optional)."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    transcript = (audit_dir / "transcript.jsonl").open("w", encoding="utf-8")
    emit = on_event or (lambda kind, data: None)
    start = time.perf_counter()

    def log(kind: str, data: dict[str, Any]) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        transcript.write(json.dumps({"time": stamp, "kind": kind, **data}, default=str) + "\n")
        transcript.flush()
        emit(kind, data)

    next_input: str | list[dict[str, Any]] = prompts.task_message(session.config)
    previous_id: str | None = None
    stopped_because = "max_steps"
    steps = 0
    try:
        while steps < max_steps:
            steps += 1
            # 1. Ask the model what to do next.
            try:
                response = client.respond(
                    instructions=prompts.SYSTEM,
                    input=next_input,
                    tools=TOOL_SCHEMAS,
                    previous_response_id=previous_id,
                    reasoning_effort="medium",
                    purpose=f"agent_step_{steps}",
                )
            except BudgetExceeded as exc:
                stopped_because = "token_budget"
                log("stopped", {"reason": str(exc)})
                break
            except Exception as exc:
                stopped_because = "model_error"
                log("stopped", {"reason": f"{type(exc).__name__}: {exc}"})
                break
            previous_id = response.id
            log("model", {"step": steps, "text": response.text, "tool_calls": [
                {"name": c.name, "arguments": c.arguments if c.arguments is not None else c.raw_arguments}
                for c in response.tool_calls
            ], "usage": response.usage})

            # 2. No tool call: remind it to use tools or finalize.
            if not response.tool_calls:
                next_input = [{"role": "user", "content": prompts.NUDGE}]
                continue

            # 3. Run every requested tool; send all results back together, in order.
            outputs = []
            for call in response.tool_calls:
                result = session.call(call.name, call.arguments if call.error is None else None)
                log("tool", {"step": steps, "name": call.name, "result": result})
                outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": to_json_text(result)})
            if session.final is not None:
                stopped_because = "finalized"
                break
            next_input = outputs
    finally:
        transcript.close()

    verdict = session.final or {
        "verdict": "inconclusive",
        "summary": f"The audit stopped before a verdict was accepted ({stopped_because}).",
        "findings": [],
        "caveats": [],
    }
    evidence = [{"id": e.id, "tool": e.tool, "arguments": e.arguments, "result": e.result} for e in session.evidence]
    result = AgentResult(
        verdict=verdict,
        stopped_because=stopped_because,
        steps=steps,
        duration_s=round(time.perf_counter() - start, 1),
        usage=client.budget.summary(),
        audit_dir=audit_dir,
        evidence=evidence,
    )
    (audit_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    (audit_dir / "verdict.json").write_text(json.dumps({
        "verdict": verdict,
        "stopped_because": stopped_because,
        "steps": steps,
        "duration_s": result.duration_s,
        "usage": result.usage,
        "model": client.model,
        "llm_calls": client.budget.calls,
    }, indent=2, default=str), encoding="utf-8")
    return result
