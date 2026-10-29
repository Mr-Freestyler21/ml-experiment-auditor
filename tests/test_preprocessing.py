"""Check A on tiny projects with planted flaws, run through the real tracer and sandbox."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auditor.checks.preprocessing import check_preprocessing
from auditor.config import load_config
from auditor.sandbox import LocalSandbox
from auditor.trace import EvaluationSplit, run_traced

CONFIG = """
entry_script: train.py
data_path: data.csv
target: y
reported_metric: accuracy
reported_value: 0.9
timeout_seconds: 120
"""

HEADER = """
import json, os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
df = pd.read_csv('data.csv')
X, y = df[['a', 'b', 'c']], df['y']
"""

FOOTER = """
open(os.path.join(os.environ['AUDIT_OUTPUT_DIR'], 'metrics.json'), 'w').write(json.dumps({'accuracy': float(acc)}))
"""

SCRIPTS = {
    # Planted: scaler learns the mean and spread of all rows, test rows included.
    "scaler_before_split": """
from sklearn.preprocessing import StandardScaler
X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
acc = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)
""",
    # Planted, severe: feature selection looks at the labels of every row.
    "selector_before_split": """
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=2).fit(X, y)
X = pd.DataFrame(selector.transform(X), index=X.index)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
acc = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)
""",
    # Planted, numpy: no row labels, but the scaler saw as many rows as train + test.
    "numpy_scaler_before_split": """
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X.to_numpy())
X_tr, X_te, y_tr, y_te = train_test_split(Xs, y.to_numpy(), test_size=0.25, random_state=0)
acc = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)
""",
    # Clean: scaler fitted on the training rows only.
    "scaler_after_split": """
from sklearn.preprocessing import StandardScaler
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
scaler = StandardScaler().fit(X_tr)
model = LogisticRegression().fit(scaler.transform(X_tr), y_tr)
acc = model.score(scaler.transform(X_te), y_te)
""",
    # Clean: scaler inside a pipeline, refitted on each training fold.
    "pipeline_in_cv": """
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=4).mean()
""",
    # Clean with a note: the encoder only learns category names, like the F1 project.
    "label_encoder_before_split": """
from sklearn.preprocessing import LabelEncoder
X = X.copy()
X['c'] = LabelEncoder().fit_transform(X['c'])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
acc = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)
""",
}


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(python=Path(sys.executable), runs_dir=tmp_path / "runs")


def audit(name: str, tmp_path: Path, sandbox: LocalSandbox):
    project = tmp_path / name
    project.mkdir()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80), "c": rng.choice(["x", "y", "z"], size=80)})
    df["y"] = (df["a"] + 0.5 * rng.normal(size=80) > 0).astype(int)
    if name != "label_encoder_before_split":
        df["c"] = rng.normal(size=80)
    df.to_csv(project / "data.csv", index=False)
    (project / "audit.yaml").write_text(CONFIG, encoding="utf-8")
    (project / "train.py").write_text(HEADER + SCRIPTS[name] + FOOTER, encoding="utf-8")
    traced = run_traced(load_config(project), sandbox)
    assert traced.error is None, traced.run.stderr
    return check_preprocessing(traced.events, traced.splits, traced.incomplete_reason)


def test_scaler_before_split_is_flagged_with_line_and_counts(tmp_path, sandbox):
    result = audit("scaler_before_split", tmp_path, sandbox)

    assert result.status == "flagged"
    [leak] = result.evidence["leaks"]
    assert (leak["class"], leak["kind"], leak["basis"]) == ("StandardScaler", "statistics", "row_labels")
    assert leak["line"] == 10
    assert (leak["n_rows_seen"], leak["test_rows_seen"], leak["n_test"]) == (80, 20, 20)
    assert "StandardScaler.fit_transform at line 10" in result.summary


def test_label_informed_selector_is_flagged_as_label_informed(tmp_path, sandbox):
    result = audit("selector_before_split", tmp_path, sandbox)

    assert result.status == "flagged"
    assert result.evidence["leaks"][0]["kind"] == "label_informed"
    assert result.measurements["label_informed_leaks"] == 1
    assert "using the labels" in result.summary


def test_numpy_scaler_is_matched_by_row_count(tmp_path, sandbox):
    result = audit("numpy_scaler_before_split", tmp_path, sandbox)

    assert result.status == "flagged"
    assert result.evidence["leaks"][0]["basis"] == "row_count"


@pytest.mark.parametrize("name", ["scaler_after_split", "pipeline_in_cv"])
def test_correct_preprocessing_is_clear(tmp_path, sandbox, name):
    result = audit(name, tmp_path, sandbox)

    assert result.status == "clear"
    assert result.evidence.get("leaks", []) == []


def test_label_encoder_on_all_rows_is_clear_with_a_note(tmp_path, sandbox):
    result = audit("label_encoder_before_split", tmp_path, sandbox)

    assert result.status == "clear"
    assert result.evidence["notes"][0]["class"] == "LabelEncoder"
    assert "only learns category names" in result.summary


def test_pandas_standardisation_before_split_is_found_by_reading_the_code():
    from auditor.checks.preprocessing import whole_data_statistics

    script = "\n".join([
        "X = df[features]",                                   # 1
        "y = df['y']",                                        # 2
        "X = (X - X.mean()) / X.std()",                       # 3  leak
        "df['a'] = df['a'] / df['a'].max()",                  # 4  leak
        "X = X.fillna(X.median())",                           # 5  leak
        "rate = y.mean()",                                    # 6  target, not preprocessing
        "scale = other.std()",                                # 7  other variable
        "X_train, X_test = X[:80], X[80:]",                   # 8  split
        "X_train = (X_train - X_train.mean()) / X_train.std()",  # 9  after the split, train only
    ])

    found = whole_data_statistics(script, before_line=8)

    assert [(f["line"], f["variable"], f["statistic"]) for f in found] == [
        (3, "X", "mean"), (4, "df", "max"), (5, "X", "median")]
    assert whole_data_statistics(script, before_line=None) == []


def test_pandas_scaler_project_is_flagged_through_the_script(tmp_path, sandbox):
    project = tmp_path / "pandas"
    project.mkdir()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80), "c": rng.normal(size=80)})
    df["y"] = (df["a"] > 0).astype(int)
    df.to_csv(project / "data.csv", index=False)
    (project / "audit.yaml").write_text(CONFIG, encoding="utf-8")
    script = HEADER + "X = (X - X.mean()) / X.std()\n" + \
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)\n" + \
        "acc = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)\n" + FOOTER
    (project / "train.py").write_text(script, encoding="utf-8")
    traced = run_traced(load_config(project), sandbox)

    without_script = check_preprocessing(traced.events, traced.splits)
    with_script = check_preprocessing(traced.events, traced.splits, script=script)

    assert without_script.status == "clear"  # the tracer alone cannot see pandas
    assert with_script.status == "flagged"
    assert with_script.evidence["static_code"][0]["line"] == 8
    assert "plain pandas" in with_script.summary


def test_incomplete_trace_is_not_judged():
    split = EvaluationSplit("M", [0], [1], 1, 2, ("predict",), 1, 1, fit_seq=2, predict_seq=3, model_obj=9)

    result = check_preprocessing([], [split], incomplete_reason="workers ran elsewhere")

    assert result.status == "not_applicable"
    assert result.summary.startswith("Trace incomplete")


def test_fit_after_the_evaluation_is_ignored():
    split = EvaluationSplit("M", [0, 1], [2, 3], 1, 2, ("predict",), 2, 2, fit_seq=2, predict_seq=3, model_obj=9)
    late_scaler = [
        {"event": "call", "seq": 4, "obj": 5, "class": "StandardScaler", "method": "fit_transform", "depth": 0,
         "from_script": True, "has_y": False, "line": 30, "n_rows": 4, "rows": [0, 1, 2, 3]},
    ]

    assert check_preprocessing(late_scaler, [split]).status == "clear"
