import pandas as pd
import pytest

from auditor.checks.group_overlap import check_group_overlap
from auditor.trace import EvaluationSplit

# 12 laps: races A and B in 2023, races A and C in 2024 (race "A" exists in both seasons).
DATA = pd.DataFrame({
    "Year": [2023] * 6 + [2024] * 6,
    "Race": ["A", "A", "A", "B", "B", "B", "A", "A", "A", "C", "C", "C"],
    "y": [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
})


def split(train, test, model="RandomForestClassifier"):
    return EvaluationSplit(
        model=model, train_rows=list(train), test_rows=list(test), fit_line=10, predict_line=12,
        predict_methods=("predict",), n_train=len(train), n_test=len(test),
    )


RACE = [("Year", "Race")]


def test_clean_grouped_split_is_clear():
    result = check_group_overlap(DATA, [split(range(0, 6), range(6, 12))], RACE, "Year")

    assert result.status == "clear"
    g = result.evidence["splits"][0]["groups"][0]
    assert g["shared_groups"] == 0
    assert result.evidence["splits"][0]["time"]["test_rows_before_train_end"] == 0


def test_random_split_across_races_is_flagged_with_counts():
    # Every race has laps on both sides, like a random 80/20 split.
    train, test = [0, 1, 3, 4, 6, 7, 9, 10], [2, 5, 8, 11]

    result = check_group_overlap(DATA, [split(train, test)], RACE, None)

    assert result.status == "flagged"
    g = result.evidence["splits"][0]["groups"][0]
    assert (g["shared_groups"], g["test_groups"], g["test_rows_in_shared_groups"]) == (4, 4, 4)
    assert g["share_of_test_rows_in_shared_groups"] == 1.0
    assert result.measurements["max_share_of_test_rows_in_shared_groups[Year+Race]"] == 1.0
    assert "4 of 4 test groups" in result.summary


def test_composite_group_keeps_same_name_in_different_seasons_apart():
    # Train on 2023, test on 2024: race "A" appears on both sides, but they are different races.
    train, test = range(0, 6), range(6, 12)

    by_year_and_race = check_group_overlap(DATA, [split(train, test)], RACE, None)
    by_name_only = check_group_overlap(DATA, [split(train, test)], [("Race",)], None)

    assert by_year_and_race.status == "clear"
    assert by_name_only.status == "flagged"


def test_test_rows_earlier_than_training_data_are_flagged():
    # Both seasons on both sides: the 2023 test rows were judged by a model that saw 2024.
    train, test = [0, 1, 6, 7], [2, 3, 8, 9]

    result = check_group_overlap(DATA, [split(train, test)], [], "Year")

    t = result.evidence["splits"][0]["time"]
    assert result.status == "flagged"
    assert t["test_rows_before_train_end"] == 2
    assert t["share_of_test_rows_before_train_end"] == 0.5
    assert t["test_range"] == [2023, 2024]
    assert "trained on their future" in result.summary


def test_worst_fold_is_reported_across_cross_validation():
    clean = split(range(0, 6), range(6, 12), model="fold1")
    leaky = split([0, 1, 3, 4], [2, 5], model="fold2")

    result = check_group_overlap(DATA, [clean, leaky], RACE, None)

    assert result.status == "flagged"
    assert result.measurements["splits_checked"] == 2
    assert result.measurements["splits_flagged"] == 1
    assert result.summary.startswith("Worst of 2 splits")


def test_numpy_inputs_are_not_applicable():
    no_rows = EvaluationSplit("M", None, None, 1, 2, ("predict",), 8, 4)

    result = check_group_overlap(DATA, [no_rows], RACE, "Year")

    assert result.status == "not_applicable"
    assert "row labels" in result.summary


@pytest.mark.parametrize("groups, time_col, splits", [([], None, "one"), (RACE, None, "none")])
def test_nothing_to_check_is_not_applicable(groups, time_col, splits):
    splits = [split(range(6), range(6, 12))] if splits == "one" else []

    assert check_group_overlap(DATA, splits, groups, time_col).status == "not_applicable"


def test_unknown_column_is_an_error():
    result = check_group_overlap(DATA, [split(range(6), range(6, 12))], [("race_id",)], None)

    assert result.status == "error"
    assert "race_id" in result.summary


def test_row_labels_outside_the_data_are_not_applicable():
    result = check_group_overlap(DATA, [split(range(6), [100, 101])], RACE, None)

    assert result.status == "not_applicable"
    assert result.evidence["unmatched_row_labels_example"] == [100, 101]
