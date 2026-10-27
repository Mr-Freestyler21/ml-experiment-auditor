"""Phase 0 smoke test: one tool-calling round trip with the Meta Model API.

The loop, in four steps:
  1. Send the question plus a tool definition.     (request 1)
  2. The model replies with a function_call item instead of an answer.
  3. Our code runs the tool and computes the number.  (the model never does maths)
  4. Send the tool result back; the model writes the final answer.  (request 2)

Env vars:
  META_MODEL_API_KEY   required (not needed for --dry-run)
  AUDITOR_MAX_TOKENS   total token cap for the whole run, default 4000
  META_MODEL_NAME      default muse-spark-1.3

Usage:
  python scripts/phase0_tool_call.py --dry-run
  python scripts/phase0_tool_call.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
DEFAULT_MAX_TOKENS = 4000
MAX_OUTPUT_PER_REQUEST = 1500

# USD per 1M tokens: (input, cached input, output). From dev.meta.ai/docs/pricing-rate-limits.
PRICES: dict[str, tuple[float, float, float]] = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.1": (1.25, 0.15, 4.25),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
}

INSTRUCTIONS = (
    "You are an ML evaluation auditor. You never compute numbers yourself. "
    "When a number is needed, call a tool and quote only the number it returns."
)

QUESTION = (
    "In the 2024 F1 test set, 33.5% of laps are pit laps. A model scored 66% accuracy. "
    "What accuracy does always predicting the majority class get, and is 66% impressive?"
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "majority_class_accuracy",
        "description": (
            "Accuracy of a classifier that always predicts the majority class, "
            "given the share of positive labels in the evaluation set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positive_rate": {
                    "type": "number",
                    "description": "Share of positive labels, between 0 and 1 (e.g. 0.335).",
                }
            },
            "required": ["positive_rate"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TokenBudget:
    """Per-run cap. The billing spend limit only notifies, so this is the real stop."""

    cap: int
    used: int = 0
    calls: list[dict[str, int]] = field(default_factory=list)

    def max_output_for_next_call(self) -> int:
        remaining = self.cap - self.used
        if remaining <= 0:
            raise BudgetExceeded(f"token cap reached: used {self.used} of {self.cap}")
        return min(remaining, MAX_OUTPUT_PER_REQUEST)

    def record(self, usage: Any) -> dict[str, int]:
        entry = {
            "input": usage.input_tokens,
            "cached": usage.input_tokens_details.cached_tokens,
            "output": usage.output_tokens,
            "reasoning": usage.output_tokens_details.reasoning_tokens,
            "total": usage.total_tokens,
        }
        self.used += entry["total"]
        self.calls.append(entry)
        return entry

    def cost_usd(self, model: str) -> float | None:
        if model not in PRICES:
            return None
        p_in, p_cached, p_out = PRICES[model]
        cost = 0.0
        for c in self.calls:
            cost += (c["input"] - c["cached"]) * p_in + c["cached"] * p_cached + c["output"] * p_out
        return cost / 1_000_000


def load_dotenv(path: Path) -> None:
    """Minimal .env reader. Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def majority_class_accuracy(positive_rate: float) -> dict[str, Any]:
    """The tool itself: plain code, no LLM."""
    majority = "negative" if positive_rate <= 0.5 else "positive"
    return {
        "majority_class": majority,
        "accuracy": round(max(positive_rate, 1 - positive_rate), 4),
    }


def run_tool(name: str, raw_arguments: str) -> dict[str, Any]:
    """Validate model-supplied arguments before running anything (strict mode checks only the schema)."""
    if name != "majority_class_accuracy":
        return {"error": f"unknown tool {name!r}"}
    try:
        args = json.loads(raw_arguments)
        rate = float(args["positive_rate"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {"error": f"bad arguments: {exc}"}
    if not 0.0 <= rate <= 1.0:
        return {"error": "positive_rate must be between 0 and 1"}
    return majority_class_accuracy(rate)


def log_usage(label: str, entry: dict[str, int], budget: TokenBudget) -> None:
    print(
        f"[{label}] tokens in={entry['input']} (cached {entry['cached']}) "
        f"out={entry['output']} (reasoning {entry['reasoning']}) "
        f"total={entry['total']} | run total {budget.used}/{budget.cap}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print request 1 and exit without calling the API")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    model = os.environ.get("META_MODEL_NAME") or DEFAULT_MODEL
    budget = TokenBudget(cap=int(os.environ.get("AUDITOR_MAX_TOKENS") or DEFAULT_MAX_TOKENS))

    first_request: dict[str, Any] = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": QUESTION,
        "tools": TOOLS,
        "reasoning": {"effort": "low"},
        "max_output_tokens": budget.max_output_for_next_call(),
    }

    if args.dry_run:
        print(f"POST {BASE_URL}/responses")
        print(json.dumps(first_request, indent=2))
        return 0

    api_key = os.environ.get("META_MODEL_API_KEY")
    if not api_key:
        print("META_MODEL_API_KEY is not set. Put it in .env or your environment.", file=sys.stderr)
        return 2

    from openai import OpenAI  # imported here so --dry-run works without the SDK

    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=60)

    # Request 1: question + tool definition.
    first = client.responses.create(**first_request)
    log_usage("request 1", budget.record(first.usage), budget)

    calls = [item for item in first.output if item.type == "function_call"]
    if not calls:
        print("The model answered without calling the tool:\n" + first.output_text)
        return 1

    # Step 3: our code runs each tool call. One result per call, in call order.
    tool_outputs = []
    for call in calls:
        print("\nfunction_call item from the model:")
        print(json.dumps(call.model_dump(), indent=2))
        result = run_tool(call.name, call.arguments)
        print(f"tool result: {json.dumps(result)}")
        tool_outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)})

    # Request 2: send results back, chained to request 1 by previous_response_id.
    second = client.responses.create(
        model=model,
        previous_response_id=first.id,
        input=tool_outputs,
        tools=TOOLS,
        reasoning={"effort": "low"},
        max_output_tokens=budget.max_output_for_next_call(),
    )
    log_usage("request 2", budget.record(second.usage), budget)

    print("\nFinal answer:\n" + second.output_text)
    cost = budget.cost_usd(model)
    print(f"\nRun total: {budget.used} tokens, est. cost " + (f"${cost:.5f}" if cost is not None else "unknown"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BudgetExceeded as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(3)
