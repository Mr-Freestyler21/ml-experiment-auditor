"""Check A: preprocessing fitted on rows that include the test set.

Plain code over the call trace, no LLM. A preprocessing fit is any fit / fit_transform
the script itself made on an object other than an evaluated model. It leaks when it saw
test rows of an evaluation and happened before that evaluation's predictions.

What the fitted object learns decides how serious that is:
  - label_informed: fitted with y (SelectKBest, TargetEncoder, ...): learned from the answers.
  - statistics: means, variances, components, ... of the test rows (StandardScaler, PCA, ...).
  - vocabulary: only the set of category names (LabelEncoder, OneHotEncoder, ...). Noted, not
    flagged: knowing a category exists says nothing about the labels.
Unknown classes count as statistics, the cautious choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from auditor.checks.base import CheckResult
from auditor.trace import EvaluationSplit

CHECK_ID = "A_preprocessing_before_split"
FIT_METHODS = {"fit", "fit_transform"}
VOCABULARY_ONLY = {"LabelEncoder", "OrdinalEncoder", "OneHotEncoder", "LabelBinarizer", "MultiLabelBinarizer"}
MAX_SPLITS_IN_EVIDENCE = 10


def check_preprocessing(
    events: Sequence[dict[str, Any]],
    splits: Sequence[EvaluationSplit],
    incomplete_reason: str | None = None,
) -> CheckResult:
    if incomplete_reason:
        return _result("not_applicable", f"Trace incomplete: {incomplete_reason}.")
    if not splits:
        return _result("not_applicable", "No model was fitted with labels and then evaluated, so no test set was found.")

    model_objs = {s.model_obj for s in splits}
    calls = [e for e in events if e.get("event") == "call" and e.get("depth") == 0]
    transformers = {e["obj"] for e in calls if e["method"] in ("transform", "fit_transform")}
    fits = [
        e for e in calls
        if e.get("from_script") and e["method"] in FIT_METHODS
        and e["obj"] in transformers and e["obj"] not in model_objs
    ]
    if not fits:
        return _result("clear", "No preprocessing was fitted outside the evaluated models (e.g. all of it sits inside a pipeline).")

    findings, unverifiable = [], []
    for fit in fits:
        for split in splits:
            if split.predict_seq is not None and fit["seq"] > split.predict_seq:
                continue  # fitted after this evaluation, so it cannot have affected it
            finding = _compare(fit, split)
            if finding is None:
                unverifiable.append(_fit_ref(fit))
            elif finding["test_rows_seen"] > 0:
                findings.append(finding)

    leaks = [f for f in findings if f["kind"] != "vocabulary"]
    notes = [f for f in findings if f["kind"] == "vocabulary"]
    evidence: dict[str, Any] = {
        "preprocessing_fits": [_fit_ref(f) for f in fits],
        "leaks": _dedupe(leaks)[:MAX_SPLITS_IN_EVIDENCE],
        "notes": _dedupe(notes)[:MAX_SPLITS_IN_EVIDENCE],
    }
    if unverifiable:
        evidence["unverifiable"] = _dedupe(unverifiable)

    if leaks:
        return _result("flagged", _summary(leaks), measurements=_measurements(leaks), evidence=evidence)
    if unverifiable and not notes:
        return _result(
            "not_applicable",
            "Preprocessing was fitted on arrays without row labels whose size does not match the data, "
            "so the rows it saw cannot be compared with the test set.",
            evidence=evidence,
        )
    summary = "No preprocessing that learns from data saw test rows before the evaluation."
    if notes:
        n = _dedupe(notes)[0]
        summary += (
            f" Note: {n['class']} at line {n['line']} was fitted on rows including the test set, but it only "
            "learns category names, which carry no information about the labels."
        )
    return _result("clear", summary, evidence=evidence)


def _compare(fit: dict[str, Any], split: EvaluationSplit) -> dict[str, Any] | None:
    """How many of the split's test rows the fit saw, by row labels or, failing that, by row count."""
    if fit.get("rows") is not None and split.test_rows is not None:
        seen = set(fit["rows"])
        test_seen = sum(1 for r in split.test_rows if r in seen)
        basis = "row_labels"
    elif split.n_train is not None and split.n_test is not None and fit.get("n_rows") == split.n_train + split.n_test:
        # No row labels, but it saw exactly as many rows as train and test together.
        test_seen = split.n_test
        basis = "row_count"
    else:
        return None
    return {
        **_fit_ref(fit),
        "kind": _kind(fit),
        "basis": basis,
        "model": split.model,
        "rows_source": split.rows_source,
        "model_fit_line": split.fit_line,
        "predict_line": split.predict_line,
        "n_test": split.n_test,
        "test_rows_seen": test_seen,
        "share_of_test_rows_seen": round(test_seen / split.n_test, 4) if split.n_test else 0.0,
    }


def _kind(fit: dict[str, Any]) -> str:
    if fit.get("has_y"):
        return "label_informed"
    if fit["class"] in VOCABULARY_ONLY:
        return "vocabulary"
    return "statistics"


def _fit_ref(fit: dict[str, Any]) -> dict[str, Any]:
    return {"class": fit["class"], "method": fit["method"], "line": fit.get("line"), "n_rows_seen": fit.get("n_rows")}


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per fitted object and line (cross-validation repeats the same finding per fold)."""
    seen, out = set(), []
    for item in items:
        key = (item["class"], item.get("line"), item.get("method"))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _measurements(leaks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "leaking_fits": len(_dedupe(leaks)),
        "label_informed_leaks": len(_dedupe([f for f in leaks if f["kind"] == "label_informed"])),
        "max_share_of_test_rows_seen": max(f["share_of_test_rows_seen"] for f in leaks),
    }


def _summary(leaks: list[dict[str, Any]]) -> str:
    worst = sorted(leaks, key=lambda f: (f["kind"] != "label_informed", -f["share_of_test_rows_seen"]))[0]
    what = "using the labels" if worst["kind"] == "label_informed" else "learning statistics"
    return (
        f"{worst['class']}.{worst['method']} at line {worst['line']} was fitted on {worst['n_rows_seen']} rows "
        f"including {worst['test_rows_seen']} of the {worst['n_test']} test rows ({worst['share_of_test_rows_seen']:.0%}), "
        f"{what}, before {worst['model']} was evaluated at line {worst['predict_line']}."
        + (" (Matched by row count; the inputs had no row labels.)" if worst["basis"] == "row_count" else "")
    )


def _result(status: str, summary: str, **kwargs: Any) -> CheckResult:
    return CheckResult(check_id=CHECK_ID, status=status, summary=summary, **kwargs)  # type: ignore[arg-type]
