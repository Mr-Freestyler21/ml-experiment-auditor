import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from auditor.checks.group_overlap import check_group_overlap
from auditor.config import load_config
from auditor.sandbox import LocalSandbox, collect_files
from auditor.trace import compare_metrics, extract_splits, run_traced


def call(seq, obj, method, rows=None, has_y=False, cls="M", depth=0, line=1):
    return {"event": "call", "seq": seq, "obj": obj, "method": method, "rows": rows, "has_y": has_y,
            "class": cls, "depth": depth, "line": line, "n_rows": len(rows) if rows else None}


def test_fit_then_predict_and_proba_on_same_rows_is_one_split():
    events = [
        call(1, 7, "fit_transform", rows=[0, 1, 2, 3], cls="LabelEncoder"),
        call(2, 9, "fit", rows=[0, 1, 2], has_y=True, cls="RF", line=20),
        call(3, 9, "predict", rows=[3], line=22),
        call(4, 9, "predict_proba", rows=[3], line=25),
        call(5, 9, "predict", rows=[3], depth=1),
    ]

    [s] = extract_splits(events)

    assert (s.model, s.train_rows, s.test_rows) == ("RF", [0, 1, 2], [3])
    assert (s.fit_line, s.predict_line, s.predict_methods) == (20, 22, ("predict", "predict_proba"))


def test_each_cross_validation_fold_is_its_own_split():
    events = [
        call(1, 1, "fit", rows=[0, 1], has_y=True), call(2, 1, "predict", rows=[2]),
        call(3, 2, "fit", rows=[1, 2], has_y=True), call(4, 2, "predict", rows=[0]),
    ]

    splits = extract_splits(events)

    assert [(s.train_rows, s.test_rows) for s in splits] == [([0, 1], [2]), ([1, 2], [0])]


def test_refitting_the_same_object_starts_a_new_split():
    events = [
        call(1, 1, "fit", rows=[0, 1], has_y=True), call(2, 1, "predict", rows=[2]),
        call(3, 1, "fit", rows=[0, 1, 2], has_y=True), call(4, 1, "predict", rows=[3]),
    ]

    assert [s.test_rows for s in extract_splits(events)] == [[2], [3]]


def split_event(seq, train_rows, test_rows, line=5):
    return {"event": "split", "seq": seq, "line": line, "outputs": [
        {"n_rows": len(train_rows), "rows": train_rows}, {"n_rows": len(test_rows), "rows": test_rows}]}


def numpy_call(seq, obj, method, n, has_y=False):
    return {**call(seq, obj, method, has_y=has_y), "n_rows": n}


def test_rows_lost_to_numpy_are_recovered_from_the_matching_split_call():
    events = [
        split_event(1, [0, 1, 2], [3]),
        numpy_call(2, 9, "fit", 3, has_y=True),
        numpy_call(3, 9, "predict", 1),
    ]

    [s] = extract_splits(events)

    assert (s.train_rows, s.test_rows) == ([0, 1, 2], [3])
    assert s.rows_source == "train_test_split at line 5"


def test_rows_are_not_recovered_when_sizes_do_not_match():
    events = [split_event(1, [0, 1], [2, 3]), numpy_call(2, 9, "fit", 3, has_y=True), numpy_call(3, 9, "predict", 1)]

    [s] = extract_splits(events)

    assert not s.has_row_identity
    assert s.rows_source == "model_inputs"


def test_unsupervised_fits_and_unevaluated_models_give_no_split():
    events = [call(1, 1, "fit", rows=[0, 1]), call(2, 2, "fit", rows=[0, 1], has_y=True)]

    assert extract_splits(events) == []


# --- End to end through the sandbox -------------------------------------------------

CONFIG = """
entry_script: train.py
data_path: data.csv
target: y
reported_metric: accuracy
reported_value: 0.9
group_columns: [g]
timeout_seconds: 120
"""

RANDOM_SPLIT = """
import json, os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('data.csv')
X, y = df[['x']], df['y']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
model = LogisticRegression().fit(X_tr, y_tr)
acc = float((model.predict(X_te) == y_te).mean())
open(os.path.join(os.environ['AUDIT_OUTPUT_DIR'], 'metrics.json'), 'w').write(json.dumps({'accuracy': acc}))
"""

GROUP_KFOLD = """
import json, os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
df = pd.read_csv('data.csv')
scores = cross_val_score(LogisticRegression(), df[['x']], df['y'], groups=df['g'], cv=GroupKFold(n_splits=4))
open(os.path.join(os.environ['AUDIT_OUTPUT_DIR'], 'metrics.json'), 'w').write(json.dumps({'accuracy': float(scores.mean())}))
"""


PARALLEL_CV = GROUP_KFOLD.replace("cv=GroupKFold(n_splits=4))", "cv=GroupKFold(n_splits=4), n_jobs=-1)")

# A script that insists on worker processes, overriding the tracer's sequential setting.
HARD_CODED_PROCESSES = GROUP_KFOLD.replace(
    "scores = cross_val_score(",
    "import joblib\nwith joblib.parallel_config(backend='loky'):\n    scores = cross_val_score(",
).replace("cv=GroupKFold(n_splits=4))", "cv=GroupKFold(n_splits=4), n_jobs=2)")


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(python=Path(sys.executable), runs_dir=tmp_path / "runs")


def make_project(folder: Path, script: str) -> Path:
    folder.mkdir()
    rows = [{"g": f"group{i // 5}", "x": i % 5 + (i // 5) * 0.1, "y": int(i % 5 > 2)} for i in range(40)]
    pd.DataFrame(rows).to_csv(folder / "data.csv", index=False)
    (folder / "audit.yaml").write_text(CONFIG, encoding="utf-8")
    (folder / "train.py").write_text(script, encoding="utf-8")
    return folder


def run_check(project: Path, sandbox: LocalSandbox):
    config = load_config(project)
    traced = run_traced(config, sandbox)
    assert traced.error is None, traced.run.stderr
    data = pd.read_csv(project / "data.csv")
    result = check_group_overlap(data, traced.splits, config.group_columns, config.time_column, traced.incomplete_reason)
    return traced, result


def test_random_split_is_traced_and_flagged(sandbox, tmp_path):
    traced, result = run_check(make_project(tmp_path / "p", RANDOM_SPLIT), sandbox)

    [s] = traced.splits
    assert (s.model, s.n_train, s.n_test, s.fit_line, s.predict_line) == ("LogisticRegression", 30, 10, 9, 10)
    assert any(e["event"] == "split" and e["line"] == 8 for e in traced.events)
    assert result.status == "flagged"


def test_group_kfold_is_traced_per_fold_and_clear(sandbox, tmp_path):
    traced, result = run_check(make_project(tmp_path / "p", GROUP_KFOLD), sandbox)

    assert len(traced.splits) == 4
    assert all(s.n_train == 30 and s.n_test == 10 for s in traced.splits)
    assert result.status == "clear"
    assert result.measurements["splits_checked"] == 4


def test_n_jobs_minus_one_is_traced_in_process(sandbox, tmp_path):
    traced, result = run_check(make_project(tmp_path / "p", PARALLEL_CV), sandbox)

    assert len(traced.splits) == 4
    assert traced.incomplete_reason is None
    assert result.status == "clear"


def test_hard_coded_worker_processes_make_the_check_refuse_to_judge(sandbox, tmp_path):
    traced, result = run_check(make_project(tmp_path / "p", HARD_CODED_PROCESSES), sandbox)

    assert "LokyBackend" in traced.incomplete_reason
    assert result.status == "not_applicable"
    assert result.summary.startswith("Trace incomplete")


def test_tracing_does_not_change_results(sandbox, tmp_path):
    project = make_project(tmp_path / "p", RANDOM_SPLIT)
    config = load_config(project)

    plain = sandbox.run(["python", "train.py"], files=collect_files(project), timeout=60)
    traced = run_traced(config, sandbox)

    plain_metrics = json.loads(plain.artifacts["metrics.json"].read_text())
    assert compare_metrics(plain_metrics, traced.metrics)["match"]


def test_compare_metrics_reports_differences():
    assert compare_metrics({"accuracy": 0.9}, {"accuracy": 0.9}) == {"match": True, "differences": {}}
    assert compare_metrics({"accuracy": 0.9}, {"accuracy": 0.8})["differences"] == {"accuracy": [0.9, 0.8]}
    assert compare_metrics({"accuracy": 0.9}, None)["match"] is False


def test_project_files_are_left_untouched(sandbox, tmp_path):
    project = make_project(tmp_path / "p", RANDOM_SPLIT)
    before = sorted(p.name for p in project.iterdir())

    run_check(project, sandbox)

    assert sorted(p.name for p in project.iterdir()) == before
