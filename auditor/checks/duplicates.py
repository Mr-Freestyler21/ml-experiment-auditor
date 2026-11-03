"""Check D: the same rows on both sides of the split.

Plain pandas, no LLM. A test row that also sits in the training set is not a test of
anything: the model has seen it. This happens when rows are copied before the split
(upsampling, oversampling) or when a join duplicates rows. Three measurements per
evaluation:
  - copies: test rows whose row label is also a training row label (made inside the script);
  - identical rows: test rows whose model features are exactly those of a training row;
  - duplicated keys: in the data file, the same id + date pair on more than one row (a join
    that duplicated rows), and how many of those land on both sides.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from auditor.checks.base import CheckResult
from auditor.trace import EvaluationSplit

CHECK_ID = "D_shared_rows"
# Flag when at least this share of test rows also appears in training.
MIN_SHARED_SHARE = 0.005
MAX_SPLITS_IN_EVIDENCE = 5


def check_shared_rows(
    data: pd.DataFrame,
    splits: Sequence[EvaluationSplit],
    time_column: str | None,
    incomplete_reason: str | None = None,
) -> CheckResult:
    if incomplete_reason:
        return _result("not_applicable", f"Trace incomplete: {incomplete_reason}.")
    if not splits:
        return _result("not_applicable", "No model was fitted with labels and then evaluated.")
    usable = [s for s in splits if s.has_row_identity]
    if not usable:
        return _result("not_applicable", "The model's inputs had no row labels, so shared rows cannot be counted.")

    keys = _key_columns(data, time_column)
    duplicated_keys = int(data.duplicated(subset=keys, keep=False).sum()) if keys else 0
    per_split = [_measure(data, s, keys) for s in usable]
    worst = max(per_split, key=lambda m: m["share_of_test_rows_shared"])
    status = "flagged" if worst["share_of_test_rows_shared"] >= MIN_SHARED_SHARE else "clear"

    parts = []
    if worst["test_rows_copied_from_training"]:
        parts.append(f"{worst['test_rows_copied_from_training']} of {worst['n_test']} test rows are copies of "
                     "training rows made inside the script (same row label on both sides)")
    if worst["test_rows_identical_to_training"]:
        parts.append(f"{worst['test_rows_identical_to_training']} test rows have exactly the features of a "
                     "training row")
    if keys and duplicated_keys:
        parts.append(f"the data file has {duplicated_keys} rows sharing a {' + '.join(keys)} key with another row, "
                     f"{worst['test_rows_with_key_in_training']} of them in the test set with their twin in training")
    summary = "; ".join(parts) + "." if parts else "No test row is shared with or identical to a training row."
    if status == "flagged":
        summary = f"{worst['share_of_test_rows_shared']:.1%} of test rows also appear in training: " + summary
    return _result(
        status,
        summary,
        measurements={"max_share_of_test_rows_shared": worst["share_of_test_rows_shared"],
                      "duplicated_key_rows_in_data": duplicated_keys},
        evidence={"key_columns": keys, "splits": per_split[:MAX_SPLITS_IN_EVIDENCE]},
    )


def _measure(data: pd.DataFrame, split: EvaluationSplit, keys: list[str]) -> dict[str, Any]:
    train, test = list(split.train_rows), list(split.test_rows)
    train_labels = set(train)
    copied = sum(1 for r in test if r in train_labels)

    identical = 0
    features = [f for f in (split.features or ()) if f in data.columns]
    # Rows described by a few categories match by chance; only count identical rows when at least
    # one feature takes many distinct values, so a match means a duplicated record.
    varied = any(data[f].nunique() >= 0.2 * len(data) for f in features)
    if features and varied:
        known = set(pd.util.hash_pandas_object(data.loc[train, features], index=False))
        hashed = pd.util.hash_pandas_object(data.loc[test, features], index=False)
        identical = int(hashed.isin(known).sum())

    key_overlap = 0
    if keys:
        train_keys = set(data.loc[train, keys].astype(str).agg("|".join, axis=1))
        test_keys = data.loc[test, keys].astype(str).agg("|".join, axis=1)
        key_overlap = int(test_keys.isin(train_keys).sum())

    shared = max(copied, identical, key_overlap)
    return {
        "model": split.model,
        "fit_line": split.fit_line,
        "predict_line": split.predict_line,
        "n_test": len(test),
        "test_rows_copied_from_training": copied,
        "test_rows_identical_to_training": identical,
        "test_rows_with_key_in_training": key_overlap,
        "share_of_test_rows_shared": round(shared / len(test), 4) if test else 0.0,
        "features_compared": len(features),
    }


def _key_columns(data: pd.DataFrame, time_column: str | None) -> list[str]:
    """An id column plus the time column, when both exist: one row per entity per period."""
    ids = [c for c in data.columns if c.lower() == "id" or c.lower().endswith("_id")]
    if not ids or not time_column or time_column not in data.columns:
        return []
    return [ids[0], time_column]


def _result(status: str, summary: str, **kwargs: Any) -> CheckResult:
    return CheckResult(check_id=CHECK_ID, status=status, summary=summary, **kwargs)  # type: ignore[arg-type]
