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


def test_history_leaves_out_expiring_reasoning_items():
    from auditor.agent.llm_client import LLMResponse

    items = [{"type": "reasoning", "id": "rs_1"}, {"type": "message", "content": []},
             {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"}]
    response = LLMResponse(id="r", text="", tool_calls=[], usage={}, output_items=items)

    assert [i["type"] for i in response.history_items()] == ["message", "function_call"]


def _client_with_fake_api(monkeypatch, make_failures):
    """A MetaLLMClient whose API raises make_failures(request) in order, then succeeds."""
    import httpx2

    from auditor.agent import llm_client

    monkeypatch.setattr(llm_client, "OUTAGE_WAITS_S", (0, 0, 0))
    client = llm_client.MetaLLMClient(api_key="fake-key-for-tests")
    failures = make_failures(httpx2.Request("POST", "https://api.meta.ai/v1/responses"))
    calls = []

    def create(**request):
        calls.append(request)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return "response"

    monkeypatch.setattr(client._client.responses, "create", create)
    return client, calls


def test_network_outages_are_retried_patiently(monkeypatch):
    import openai

    client, calls = _client_with_fake_api(
        monkeypatch, lambda req: [openai.APIConnectionError(request=req), openai.APIConnectionError(request=req)])

    assert client._create_with_patience({"model": "m"}) == "response"
    assert len(calls) == 3


def test_request_errors_are_not_retried(monkeypatch):
    import httpx2
    import openai

    client, calls = _client_with_fake_api(monkeypatch, lambda req: [
        openai.BadRequestError("bad", response=httpx2.Response(400, request=req), body=None)])

    with pytest.raises(openai.BadRequestError):
        client._create_with_patience({"model": "m"})
    assert len(calls) == 1


def test_no_key_means_no_client(tmp_path, monkeypatch):
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    (tmp_path / ".env").write_text("META_MODEL_API_KEY=\n", encoding="utf-8")

    assert meta_client_from_env(tmp_path) is None
