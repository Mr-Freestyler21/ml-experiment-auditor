"""The agent's tools and loop, driven by a scripted model (no API calls, no credits spent)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auditor.agent.llm_client import LLMResponse, ScriptedLLMClient, parse_tool_call
from auditor.agent.loop import run_agent
from auditor.agent.tools import load_session
from auditor.config import load_config
from auditor.sandbox import LocalSandbox

CONFIG = """
entry_script: train.py
data_path: data.csv
target: y
reported_metric: accuracy
reported_value: 0.9
prediction_time: Predict for a new site, before any of its measurements are labelled.
group_columns: [site]
time_column: year
random_seed: 0
timeout_seconds: 120
"""

# Each site has its own label and a site-specific feature value: a random split lets the
# model memorise sites, a site-grouped split cannot.
SCRIPT = """
import json, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('data.csv')
X, y = df[['x']], df['y']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_tr, y_tr)
acc = float((model.predict(X_te) == y_te).mean())
open(os.path.join(os.environ['AUDIT_OUTPUT_DIR'], 'metrics.json'), 'w').write(json.dumps({'accuracy': acc}))
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for site in range(12):
        label, centre = int(rng.random() < 0.5), rng.normal(0, 5)
        for i in range(15):
            rows.append({"site": f"s{site}", "year": 2020 + i % 3, "x": centre + rng.normal(0, 0.1), "y": label})
    folder = tmp_path / "project"
    folder.mkdir()
    pd.DataFrame(rows).to_csv(folder / "data.csv", index=False)
    (folder / "audit.yaml").write_text(CONFIG, encoding="utf-8")
    (folder / "train.py").write_text(SCRIPT, encoding="utf-8")
    return folder


@pytest.fixture
def session(project: Path, tmp_path: Path):
    sandbox = LocalSandbox(python=Path(sys.executable), runs_dir=tmp_path / "runs")
    return load_session(load_config(project), sandbox)


def turn(n: int, *calls: tuple[str, dict]) -> LLMResponse:
    tool_calls = [parse_tool_call(name, f"c{n}_{i}", json.dumps(args)) for i, (name, args) in enumerate(calls)]
    return LLMResponse(id=f"r{n}", text="", tool_calls=tool_calls,
                       usage={"input": 100, "cached": 0, "output": 20, "reasoning": 0, "total": 120})


# --- tools ------------------------------------------------------------------------------


def test_read_file_numbers_lines_and_stays_inside_the_project(session):
    script = session.call("read_file", {"path": "train.py"})
    csv = session.call("read_file", {"path": "data.csv"})
    outside = session.call("read_file", {"path": "../secrets.txt"})

    assert "   8  X_tr, X_te" in script["content"]
    assert csv["total_lines"] == 181 and len(csv["first_lines"]) == 6
    assert "no such file" in outside["error"]
    assert [e.id for e in session.evidence] == ["E1", "E2", "E3"]


def test_bad_arguments_and_unknown_tools_come_back_as_errors(session):
    assert "must be one of" in session.call("run_check", {"check": "Z"})["error"]
    assert "unknown tool" in session.call("delete_everything", {})["error"]
    assert "not valid JSON" in session.call("run_check", None)["error"]


def test_group_split_removes_overlap_and_lowers_the_score(session):
    random = session.call("run_with_split", {"strategy": "random"})
    group = session.call("run_with_split", {"strategy": "group"})

    assert random["evaluations"][0]["groups"][0]["shared_groups"] > 0
    assert group["evaluations"][0]["groups"][0]["shared_groups"] == 0
    assert group["metrics"]["accuracy"] < random["metrics"]["accuracy"]
    assert group["change_vs_baseline"] < 0


def test_temporal_and_restricted_splits(session):
    temporal = session.call("run_with_split", {"strategy": "temporal"})
    restricted = session.call("run_with_split", {"strategy": "random", "restrict_to": {"column": "year", "value": 2021}})

    t = temporal["evaluations"][0]["time"]
    assert t["test_range"] == [2022, 2022] and t["train_range"][1] == 2021
    assert t["test_rows_before_train_end"] == 0
    r = restricted["evaluations"][0]
    assert r["n_train"] + r["n_test"] == 60
    assert r["time"]["train_range"] == [2021, 2021]


def test_target_rates_and_majority_baseline_are_reported(session):
    result = session.call("run_with_split", {"strategy": "group"})

    ev = result["evaluations"][0]
    assert 0 <= ev["target_rate_test"] <= 1
    assert ev["majority_class_accuracy_test"] >= 0.5


# --- loop -------------------------------------------------------------------------------


def test_scripted_audit_rejects_an_invented_number_then_finalizes(session, tmp_path):
    good = {
        "verdict": "confirmed",
        "summary": "Sites appear in both train and test; a site-grouped split lowers accuracy.",
        "findings": [{"family": "C", "title": "Sites in both sets", "explanation": "Check C is flagged.",
                      "evidence_ids": ["E2", "E3"]}],
        "caveats": [],
    }
    invented = {**good, "summary": "Accuracy falls by 37.5 points."}
    client = ScriptedLLMClient([
        turn(1, ("run_baseline", {})),
        turn(2, ("run_check", {"check": "C"}), ("run_with_split", {"strategy": "group"})),
        turn(3, ("finalize", invented)),
        turn(4, ("finalize", good)),
    ])

    result = run_agent(session, client, tmp_path / "audit")

    assert result.stopped_because == "finalized"
    assert result.verdict["verdict"] == "confirmed"
    assert result.steps == 4
    assert [e["tool"] for e in result.evidence] == ["run_baseline", "run_check", "run_with_split"]
    assert result.evidence[1]["result"]["status"] == "flagged"
    rejected = [json.loads(line) for line in (tmp_path / "audit" / "transcript.jsonl").open()
                if '"finalize"' in line and '"accepted": false' in line]
    assert rejected and "37.5" in json.dumps(rejected)
    sent_back = client.requests[3]["input"][0]
    assert sent_back["type"] == "function_call_output" and "37.5" in sent_back["output"]
    assert json.loads((tmp_path / "audit" / "verdict.json").read_text())["verdict"]["verdict"] == "confirmed"


def test_results_go_back_with_matching_call_ids_and_previous_response(session, tmp_path):
    client = ScriptedLLMClient([turn(1, ("list_files", {}), ("read_file", {"path": "train.py"})), turn(2)])

    run_agent(session, client, tmp_path / "audit", max_steps=2)

    second = client.requests[1]
    assert second["previous_response_id"] == "r1"
    assert [o["call_id"] for o in second["input"]] == ["c1_0", "c1_1"]


def test_text_without_tool_calls_gets_a_nudge(session, tmp_path):
    client = ScriptedLLMClient([turn(1), turn(2)])

    run_agent(session, client, tmp_path / "audit", max_steps=2)

    assert client.requests[1]["input"][0]["role"] == "user"


def test_step_cap_ends_inconclusive(session, tmp_path):
    client = ScriptedLLMClient([turn(i, ("list_files", {})) for i in range(1, 4)])

    result = run_agent(session, client, tmp_path / "audit", max_steps=3)

    assert result.stopped_because == "max_steps"
    assert result.verdict["verdict"] == "inconclusive"


def test_token_budget_ends_inconclusive(session, tmp_path):
    client = ScriptedLLMClient([turn(i, ("list_files", {})) for i in range(1, 6)])
    client.budget.cap = 250  # turns of 120 tokens: the third still fits (240 used), the fourth is refused

    result = run_agent(session, client, tmp_path / "audit")

    assert result.stopped_because == "token_budget"
    assert result.usage["calls"] == 3
    assert result.verdict["verdict"] == "inconclusive"
