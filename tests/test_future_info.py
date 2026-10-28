import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auditor.agent.llm_client import LLMResponse, ScriptedLLMClient, scripted_tool_response
from auditor.checks.future_info import (
    SUSPECT_TOOL,
    Suspect,
    check_future_info,
    feature_power,
    llm_suspects,
    rules_suspects,
)
from auditor.config import load_config
from auditor.sandbox import LocalSandbox

CONFIG = """
entry_script: train.py
data_path: data.csv
target: y
reported_metric: accuracy
reported_value: 0.9
prediction_time: When a customer signs up, predict whether they will churn within a year.
random_seed: 0
timeout_seconds: 120
"""

SCRIPT = """
import json, os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('data.csv')
X, y = df[{features}], df['y']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
acc = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)
open(os.path.join(os.environ['AUDIT_OUTPUT_DIR'], 'metrics.json'), 'w').write(json.dumps({{'accuracy': acc}}))
"""


def make_data(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    age = rng.normal(40, 10, n)
    y = (age + rng.normal(0, 10, n) > 45).astype(int)
    # Planted leak: recorded after the outcome, so it mirrors y almost exactly.
    cancelled_after = np.where(rng.random(n) < 0.97, y, 1 - y)
    return pd.DataFrame({"age": age, "noise": rng.normal(size=n), "cancelled_after": cancelled_after, "y": y})


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(python=Path(sys.executable), runs_dir=tmp_path / "runs")


def make_project(folder: Path, features: list[str]) -> Path:
    folder.mkdir()
    make_data().to_csv(folder / "data.csv", index=False)
    (folder / "audit.yaml").write_text(CONFIG, encoding="utf-8")
    (folder / "train.py").write_text(SCRIPT.format(features=features), encoding="utf-8")
    return folder


def baseline_metrics(project: Path, sandbox: LocalSandbox) -> dict:
    from auditor.baseline import run_baseline

    return run_baseline(load_config(project), sandbox).metrics


# --- feature_power and rules ---------------------------------------------------------


def test_feature_power_separates_leak_from_noise():
    data = make_data()
    power = feature_power(data, ["age", "noise", "cancelled_after"], "y")

    assert power["cancelled_after"] >= 0.95
    assert power["noise"] < 0.6
    assert 0.6 < power["age"] < 0.95


def test_feature_power_handles_text_and_numeric_targets():
    data = pd.DataFrame({"city": ["a", "b"] * 10, "x": range(20), "price": range(20)})

    assert feature_power(data, ["city"], "price")["city"] is not None
    assert feature_power(data, ["x"], "price")["x"] == 1.0


def test_rules_suspect_names_target_name_and_high_power():
    power = {"age": 0.7, "final_status": 0.6, "y_lag": 0.6, "score": 0.93}

    suspects = {s.column: s.reason for s in rules_suspects(list(power), "y_lag", power)}

    assert set(suspects) == {"final_status", "y_lag", "score"}
    assert "final" in suspects["final_status"]
    assert "target's name" in suspects["y_lag"]


# --- llm_suspects with a scripted model ----------------------------------------------


def test_llm_suspects_keeps_valid_columns_and_rejects_others(tmp_path):
    project = make_project(tmp_path / "p", ["age", "cancelled_after"])
    client = ScriptedLLMClient([scripted_tool_response(SUSPECT_TOOL, {"suspects": [
        {"column": "cancelled_after", "reason": "recorded after the churn happened"},
        {"column": "made_up", "reason": "not a feature"},
    ]})])

    suspects, notes = llm_suspects(client, load_config(project), ["age", "cancelled_after"], make_data())

    assert suspects == [Suspect("cancelled_after", "llm", "recorded after the churn happened")]
    assert notes["rejected_items"][0]["column"] == "made_up"
    sent = client.requests[0]["input"]
    assert "When a customer signs up" in sent and '"name": "cancelled_after"' in sent


def test_llm_answering_without_the_tool_gives_no_suspects(tmp_path):
    project = make_project(tmp_path / "p", ["age"])
    usage = {"input": 1, "cached": 0, "output": 1, "reasoning": 0, "total": 2}
    client = ScriptedLLMClient([LLMResponse(id="r", text="I think age is fine.", tool_calls=[], usage=usage)])

    suspects, notes = llm_suspects(client, load_config(project), ["age"], make_data())

    assert suspects == []
    assert "without calling the tool" in notes["error"]


# --- the check, end to end through the sandbox ---------------------------------------


def test_planted_future_column_is_flagged_by_its_shuffle_rerun(tmp_path, sandbox):
    features = ["age", "cancelled_after"]
    project = make_project(tmp_path / "p", features)
    config = load_config(project)
    data = make_data()
    suspects = rules_suspects(features, "y", feature_power(data, features, "y"))

    result = check_future_info(config, sandbox, data, features, baseline_metrics(project, sandbox), suspects)

    assert result.status == "flagged"
    [rerun] = result.evidence["shuffle_reruns"]
    assert rerun["column"] == "cancelled_after"
    assert rerun["metric_drop"] >= 0.15
    assert "Shuffling 'cancelled_after'" in result.summary


def test_suspect_without_effect_is_clear(tmp_path, sandbox):
    features = ["age", "noise"]
    project = make_project(tmp_path / "p", features)
    suspects = [Suspect("noise", "llm", "sounds like it is measured later")]

    result = check_future_info(
        load_config(project), sandbox, make_data(), features, baseline_metrics(project, sandbox), suspects
    )

    assert result.status == "clear"
    assert result.evidence["shuffle_reruns"][0]["dependence"] == "none"


@pytest.mark.parametrize("drop, power, expected", [
    (0.01, 0.6, "none"),     # below the minimum drop
    (0.0255, 0.54, "weak"),  # the F1 case: 0.0255 of a 0.143 margin, weak alone
    (0.08, 0.6, "strong"),   # more than half the margin
    (0.03, 0.95, "strong"),  # near-perfect on its own
])
def test_dependence_strength(drop, power, expected):
    from auditor.checks.future_info import dependence_strength

    assert dependence_strength(drop, power, baseline=0.952, chance=0.809) == expected


def test_no_suspects_means_no_reruns(tmp_path, sandbox):
    project = make_project(tmp_path / "p", ["age"])

    result = check_future_info(load_config(project), sandbox, make_data(), ["age"], {"accuracy": 0.8}, [])

    assert result.status == "clear"
    assert result.evidence["shuffle_reruns"] == []


def test_unknown_features_are_not_applicable(tmp_path, sandbox):
    project = make_project(tmp_path / "p", ["age"])

    assert check_future_info(load_config(project), sandbox, make_data(), None, {}, []).status == "not_applicable"
