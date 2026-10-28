"""The model behind an interface, so it can be swapped (Meta API, a cheaper model, a fake for tests).

Every call goes through a TokenBudget: the billing page's spend limit only notifies, so
this per-run cap is the real protection. The API key is read from the environment only
and never logged.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
DEFAULT_MAX_TOKENS = 20_000
MAX_OUTPUT_PER_REQUEST = 4_000

# USD per 1M tokens: (input, cached input, output). From dev.meta.ai/docs/pricing-rate-limits.
PRICES: dict[str, tuple[float, float, float]] = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.1": (1.25, 0.15, 4.25),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TokenBudget:
    cap: int
    used: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def max_output_for_next_call(self) -> int:
        remaining = self.cap - self.used
        if remaining <= 0:
            raise BudgetExceeded(f"token cap reached: used {self.used} of {self.cap}")
        return min(remaining, MAX_OUTPUT_PER_REQUEST)

    def record(self, model: str, usage: dict[str, int], duration_s: float, purpose: str) -> None:
        self.used += usage.get("total", 0)
        self.calls.append({"model": model, "purpose": purpose, "duration_s": round(duration_s, 2), **usage})

    def cost_usd(self) -> float:
        total = 0.0
        for c in self.calls:
            p_in, p_cached, p_out = PRICES.get(c["model"], (0.0, 0.0, 0.0))
            total += (c["input"] - c["cached"]) * p_in + c["cached"] * p_cached + c["output"] * p_out
        return round(total / 1_000_000, 6)

    def summary(self) -> dict[str, Any]:
        return {"calls": len(self.calls), "tokens": self.used, "cap": self.cap, "est_cost_usd": self.cost_usd()}


@dataclass(frozen=True)
class ToolCall:
    name: str
    call_id: str
    arguments: dict[str, Any] | None
    raw_arguments: str
    error: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    id: str
    text: str
    tool_calls: list[ToolCall]
    usage: dict[str, int]


class LLMClient(ABC):
    model: str
    budget: TokenBudget

    @abstractmethod
    def respond(
        self,
        *,
        instructions: str,
        input: str | list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
        previous_response_id: str | None = None,
        reasoning_effort: str = "low",
        purpose: str = "",
    ) -> LLMResponse:
        """One Responses API round trip. Tool calls come back parsed; their arguments still need validating."""


def parse_tool_call(name: str, call_id: str, raw: str) -> ToolCall:
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("arguments are not a JSON object")
        return ToolCall(name, call_id, args, raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return ToolCall(name, call_id, None, raw, error=f"could not parse arguments: {exc}")


class MetaLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=120)
        self.model = model
        self.budget = TokenBudget(cap=max_tokens)

    def respond(
        self,
        *,
        instructions: str,
        input: str | list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
        previous_response_id: str | None = None,
        reasoning_effort: str = "low",
        purpose: str = "",
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": self.budget.max_output_for_next_call(),
        }
        if tools:
            request["tools"] = list(tools)
        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        start = time.perf_counter()
        response = self._client.responses.create(**request)
        u = response.usage
        usage = {
            "input": u.input_tokens,
            "cached": u.input_tokens_details.cached_tokens,
            "output": u.output_tokens,
            "reasoning": u.output_tokens_details.reasoning_tokens,
            "total": u.total_tokens,
        }
        self.budget.record(self.model, usage, time.perf_counter() - start, purpose)
        calls = [
            parse_tool_call(item.name, item.call_id, item.arguments)
            for item in response.output
            if item.type == "function_call"
        ]
        return LLMResponse(id=response.id, text=response.output_text or "", tool_calls=calls, usage=usage)


class ScriptedLLMClient(LLMClient):
    """Replays prepared responses, for tests and offline development. Records what it was sent."""

    def __init__(self, responses: Sequence[LLMResponse], model: str = "scripted") -> None:
        self._responses = list(responses)
        self.model = model
        self.budget = TokenBudget(cap=10**9)
        self.requests: list[dict[str, Any]] = []

    def respond(self, **request: Any) -> LLMResponse:
        self.budget.max_output_for_next_call()  # enforce the cap exactly like the real client
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedLLMClient ran out of responses")
        response = self._responses.pop(0)
        self.budget.record(self.model, response.usage, 0.0, request.get("purpose", ""))
        return response


def scripted_tool_response(name: str, arguments: dict[str, Any], text: str = "") -> LLMResponse:
    raw = json.dumps(arguments)
    usage = {"input": 100, "cached": 0, "output": 50, "reasoning": 0, "total": 150}
    return LLMResponse(id="resp_scripted", text=text, tool_calls=[parse_tool_call(name, "call_1", raw)], usage=usage)


def load_dotenv(path: Path) -> None:
    """Minimal .env reader. Variables already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def meta_client_from_env(repo_root: Path) -> MetaLLMClient | None:
    """The Meta client configured from .env / the environment, or None when no key is set."""
    load_dotenv(repo_root / ".env")
    key = os.environ.get("META_MODEL_API_KEY")
    if not key:
        return None
    return MetaLLMClient(
        api_key=key,
        model=os.environ.get("META_MODEL_NAME") or DEFAULT_MODEL,
        max_tokens=int(os.environ.get("AUDITOR_MAX_TOKENS") or DEFAULT_MAX_TOKENS),
    )
