import pytest

from auditor.agent.llm_client import (
    BudgetExceeded,
    ScriptedLLMClient,
    TokenBudget,
    meta_client_from_env,
    parse_tool_call,
    scripted_tool_response,
)


def test_budget_caps_output_and_stops_when_spent():
    budget = TokenBudget(cap=5000)
    assert budget.max_output_for_next_call() == 4000

    budget.record("muse-spark-1.3", {"input": 4000, "cached": 0, "output": 900, "reasoning": 0, "total": 4900}, 1.0, "x")
    assert budget.max_output_for_next_call() == 100

    budget.record("muse-spark-1.3", {"input": 100, "cached": 0, "output": 0, "reasoning": 0, "total": 100}, 1.0, "x")
    with pytest.raises(BudgetExceeded):
        budget.max_output_for_next_call()


def test_cost_uses_cached_and_output_prices():
    budget = TokenBudget(cap=10**6)
    budget.record("muse-spark-1.3", {"input": 1_000_000, "cached": 200_000, "output": 100_000, "reasoning": 0,
                                     "total": 1_100_000}, 1.0, "x")

    # 800k uncached at $1.25 + 200k cached at $0.15 + 100k output at $4.25 per million
    assert budget.cost_usd() == pytest.approx(1.0 + 0.03 + 0.425)


def test_bad_tool_arguments_are_reported_not_raised():
    call = parse_tool_call("t", "c1", "{not json")

    assert call.arguments is None
    assert "could not parse" in call.error


def test_scripted_client_records_requests_and_usage():
    client = ScriptedLLMClient([scripted_tool_response("t", {"a": 1})])

    response = client.respond(instructions="i", input="hello", purpose="test")

    assert response.tool_calls[0].arguments == {"a": 1}
    assert client.requests[0]["input"] == "hello"
    assert client.budget.summary()["calls"] == 1


def test_no_key_means_no_client(tmp_path, monkeypatch):
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    (tmp_path / ".env").write_text("META_MODEL_API_KEY=\n", encoding="utf-8")

    assert meta_client_from_env(tmp_path) is None
