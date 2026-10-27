"""Check C: train/test overlap between entities that should be separate.

Plain pandas, no LLM. Given the raw data and the rows each model was trained and
evaluated on, it measures:
  - for each group (an entity that will be new when the model is used, e.g. a race),
    how many test rows come from groups that also appear in training;
  - for the time column, how many test rows are earlier than the latest training data,
    i.e. evaluated by a model that was trained on their future.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from auditor.checks.base import CheckResult
from auditor.trace import EvaluationSplit

CHECK_ID = "C_group_overlap"
MAX_EXAMPLES = 5
MAX_SPLITS_IN_EVIDENCE = 10


def check_group_overlap(
    data: pd.DataFrame,
    splits: Sequence[EvaluationSplit],
    group_columns: Sequence[Sequence[str]],
    time_column: str | None,
) -> CheckResult:
    if not group_columns and not time_column:
        return _result("not_applicable", "No group_columns or time_column configured, so there is nothing to compare.")
    if not splits:
        return _result("not_applicable", "No model was fitted with labels and then evaluated, so no train/test split was found.")
    if not all(s.has_row_identity for s in splits):
        return _result(
            "not_applicable",
            "The model was trained or evaluated on arrays without row labels (e.g. numpy), "
            "so its rows cannot be matched to the data.",
            evidence={"splits": [_split_ref(s) for s in splits[:MAX_SPLITS_IN_EVIDENCE]]},
        )

    configured = {c for cols in group_columns for c in cols} | ({time_column} if time_column else set())
    missing = sorted(configured - set(data.columns))
    if missing:
        return _result("error", f"Configured columns are not in the data: {', '.join(missing)}.")
    unknown = _unknown_rows(data, splits)
    if unknown:
        return _result(
            "not_applicable",
            "The row labels the model saw do not match the data's rows (the index may have been reset), "
            "so groups cannot be looked up.",
            evidence={"unmatched_row_labels_example": unknown},
        )

    per_split = [_measure_split(data, split, group_columns, time_column) for split in splits]
    flagged = [m for m in per_split if m["flagged"]]
    status = "flagged" if flagged else "clear"
    return _result(
        status,
        _summary(per_split, group_columns, time_column),
        measurements=_aggregate(per_split, group_columns, time_column),
        evidence={
            "assumption": "Row labels seen by the model are the data file's row numbers (pandas index from read_csv).",
            "n_splits": len(per_split),
            "splits": per_split[:MAX_SPLITS_IN_EVIDENCE],
        },
    )


def _measure_split(
    data: pd.DataFrame, split: EvaluationSplit, group_columns: Sequence[Sequence[str]], time_column: str | None
) -> dict[str, Any]:
    train = data.loc[split.train_rows]
    test = data.loc[split.test_rows]
    record: dict[str, Any] = {**_split_ref(split), "groups": [], "time": None, "flagged": False}

    for cols in group_columns:
        g = _group_overlap(train, test, list(cols))
        record["groups"].append(g)
        record["flagged"] |= g["shared_groups"] > 0
    if time_column:
        t = _time_overlap(train[time_column], test[time_column], time_column)
        record["time"] = t
        record["flagged"] |= t["test_rows_before_train_end"] > 0
    return record


def _group_overlap(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    train_keys = _keys(train, cols)
    test_keys = _keys(test, cols)
    shared = set(train_keys) & set(test_keys)
    test_in_shared = int(test_keys.isin(shared).sum())
    train_counts = train_keys.value_counts()
    test_counts = test_keys.value_counts()
    examples = sorted(shared, key=lambda k: -int(test_counts[k]))[:MAX_EXAMPLES]
    return {
        "columns": cols,
        "train_groups": int(train_keys.nunique()),
        "test_groups": int(test_keys.nunique()),
        "shared_groups": len(shared),
        "test_rows": len(test_keys),
        "test_rows_in_shared_groups": test_in_shared,
        "share_of_test_rows_in_shared_groups": round(test_in_shared / len(test_keys), 4) if len(test_keys) else 0.0,
        "examples": [
            {"group": k, "train_rows": int(train_counts[k]), "test_rows": int(test_counts[k])} for k in examples
        ],
    }


def _time_overlap(train: pd.Series, test: pd.Series, column: str) -> dict[str, Any]:
    """Flag when the model was trained on the future of some test rows (test time < latest train time)."""
    train_end = train.max()
    before_train_end = int((test < train_end).sum())
    at_or_after = int((train >= test.min()).sum())
    return {
        "column": column,
        "train_range": [_plain(train.min()), _plain(train_end)],
        "test_range": [_plain(test.min()), _plain(test.max())],
        "test_rows": len(test),
        "test_rows_before_train_end": before_train_end,
        "share_of_test_rows_before_train_end": round(before_train_end / len(test), 4) if len(test) else 0.0,
        "train_rows_at_or_after_test_start": at_or_after,
    }


def _keys(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    """One string key per row, e.g. '2024 | Monaco Grand Prix' for [Year, Race]."""
    return frame[cols].astype(str).agg(" | ".join, axis=1)


def _unknown_rows(data: pd.DataFrame, splits: Sequence[EvaluationSplit]) -> list[Any]:
    index = set(data.index)
    unknown: list[Any] = []
    for s in splits:
        for row in (*s.train_rows, *s.test_rows):  # type: ignore[misc]
            if row not in index:
                unknown.append(row)
                if len(unknown) >= MAX_EXAMPLES:
                    return unknown
    return unknown


def _aggregate(per_split: list[dict[str, Any]], group_columns: Sequence[Sequence[str]], time_column: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"splits_checked": len(per_split), "splits_flagged": sum(m["flagged"] for m in per_split)}
    for i, cols in enumerate(group_columns):
        shares = [m["groups"][i]["share_of_test_rows_in_shared_groups"] for m in per_split]
        out[f"max_share_of_test_rows_in_shared_groups[{'+'.join(cols)}]"] = max(shares)
    if time_column:
        shares = [m["time"]["share_of_test_rows_before_train_end"] for m in per_split]
        out[f"max_share_of_test_rows_before_train_end[{time_column}]"] = max(shares)
    return out


def _summary(per_split: list[dict[str, Any]], group_columns: Sequence[Sequence[str]], time_column: str | None) -> str:
    worst = max(per_split, key=lambda m: max([g["share_of_test_rows_in_shared_groups"] for g in m["groups"]] or [0]))
    parts = []
    for g in worst["groups"]:
        name = " + ".join(g["columns"])
        parts.append(
            f"{g['shared_groups']} of {g['test_groups']} test groups by {name} also appear in training "
            f"({g['share_of_test_rows_in_shared_groups']:.1%} of test rows)"
        )
    if time_column and worst["time"]:
        t = worst["time"]
        parts.append(
            f"{t['share_of_test_rows_before_train_end']:.1%} of test rows have {time_column} earlier than "
            f"the latest training data ({t['train_range'][1]}), so the model was trained on their future"
        )
    prefix = f"Worst of {len(per_split)} splits: " if len(per_split) > 1 else ""
    return prefix + "; ".join(parts) + "."


def _split_ref(split: EvaluationSplit) -> dict[str, Any]:
    return {
        "model": split.model,
        "fit_line": split.fit_line,
        "predict_line": split.predict_line,
        "predict_methods": list(split.predict_methods),
        "n_train": split.n_train,
        "n_test": split.n_test,
    }


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _result(status: str, summary: str, **kwargs: Any) -> CheckResult:
    return CheckResult(check_id=CHECK_ID, status=status, summary=summary, **kwargs)  # type: ignore[arg-type]
