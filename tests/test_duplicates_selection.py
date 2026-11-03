import numpy as np
import pandas as pd

from auditor.checks.duplicates import check_shared_rows
from auditor.checks.selection import check_selection
from auditor.trace import EvaluationSplit

RNG = np.random.default_rng(0)
DATA = pd.DataFrame({
    "pump_id": [f"P{i % 10}" for i in range(100)],
    "date": [f"2024-01-{i // 10 + 1:02d}" for i in range(100)],
    "x": RNG.normal(size=100),
    "y": RNG.integers(0, 2, size=100),
})


def split(train, test, model="RandomForestClassifier", obj=1, line=10):
    return EvaluationSplit(
        model=model, train_rows=list(train), test_rows=list(test), fit_line=line, predict_line=line + 1,
        predict_methods=("predict",), n_train=len(train), n_test=len(test), model_obj=obj, features=("x",),
    )


# --- check D: shared rows ---


def test_disjoint_split_of_unique_rows_is_clear():
    result = check_shared_rows(DATA, [split(range(0, 80), range(80, 100))], "date")

    assert result.status == "clear"
    assert result.measurements["max_share_of_test_rows_shared"] == 0.0


def test_rows_copied_before_the_split_are_flagged():
    # Oversampling before the split: row labels 0-9 appear on both sides.
    train = list(range(0, 80)) + list(range(0, 10))
    test = list(range(80, 100)) + list(range(0, 10))

    result = check_shared_rows(DATA, [split(train, test)], "date")

    assert result.status == "flagged"
    s = result.evidence["splits"][0]
    assert s["test_rows_copied_from_training"] == 10
    assert s["share_of_test_rows_shared"] == round(10 / 30, 4)


def test_duplicated_records_from_a_join_are_flagged_by_key_and_features():
    # A join duplicated the last 20 rows under new labels (100-119).
    dup = DATA.iloc[80:100].copy()
    dup.index = range(100, 120)
    data = pd.concat([DATA, dup])

    result = check_shared_rows(data, [split(range(0, 100), range(100, 120))], "date")

    s = result.evidence["splits"][0]
    assert result.status == "flagged"
    assert s["test_rows_copied_from_training"] == 0
    assert s["test_rows_identical_to_training"] == 20
    assert s["test_rows_with_key_in_training"] == 20
    assert result.evidence["key_columns"] == ["pump_id", "date"]
    assert result.measurements["duplicated_key_rows_in_data"] == 40


def test_low_cardinality_features_do_not_count_as_identical_rows():
    data = DATA.assign(x=DATA["x"].round().clip(-1, 1))  # three distinct values: matches by chance

    result = check_shared_rows(data, [split(range(0, 80), range(80, 100))], None)

    assert result.evidence["splits"][0]["test_rows_identical_to_training"] == 0
    assert result.status == "clear"


def test_d_needs_row_labels():
    s = EvaluationSplit(model="M", train_rows=None, test_rows=None, fit_line=1, predict_line=2,
                        predict_methods=("predict",), n_train=80, n_test=20)

    assert check_shared_rows(DATA, [s], "date").status == "not_applicable"


# --- check E: choices made on the test rows ---


def test_several_candidates_scored_on_the_same_test_rows_are_flagged():
    splits = [split(range(80), range(80, 100), obj=i, line=20) for i in (1, 2, 3)]

    result = check_selection(splits)

    assert result.status == "flagged"
    assert result.evidence["candidates"][0]["candidates"] == 3
    assert "3 RandomForestClassifier candidates" in result.summary


def test_one_model_per_class_is_clear():
    splits = [split(range(80), range(80, 100), model="LogisticRegression", obj=1),
              split(range(80), range(80, 100), model="RandomForestClassifier", obj=2)]

    assert check_selection(splits).status == "clear"


def test_static_threshold_search_is_reported_but_not_flagged_alone():
    items = [{"family": "E", "line": 42, "pattern": "choice made by scoring on the evaluation data"}]

    result = check_selection([split(range(80), range(80, 100))], items)

    assert result.status == "clear"
    assert "line 42" in result.summary
