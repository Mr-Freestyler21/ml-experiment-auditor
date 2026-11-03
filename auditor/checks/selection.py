"""Check E: choices made on the test set.

Plain code over the call trace, no LLM. When several candidate models of the same kind are
fitted and each is scored on the same held-out rows, and the best is reported, the held-out
rows were used to choose: the reported score is the maximum of several noisy scores, so it is
optimistic. (Comparing two different kinds of model, such as a baseline and the final model,
on the test set is normal reporting and is not flagged.)

The code review adds the static view: a threshold or setting picked inside a loop or a
max(...) by its score on the evaluation rows.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from auditor.checks.base import CheckResult
from auditor.trace import EvaluationSplit

CHECK_ID = "E_selection_on_test"


def check_selection(
    splits: Sequence[EvaluationSplit],
    code_items: Sequence[dict[str, Any]] = (),
    incomplete_reason: str | None = None,
) -> CheckResult:
    if incomplete_reason:
        return _result("not_applicable", f"Trace incomplete: {incomplete_reason}.")
    if not splits:
        return _result("not_applicable", "No model was fitted with labels and then evaluated.")

    by_test_rows: dict[str, list[EvaluationSplit]] = defaultdict(list)
    for s in splits:
        key = json.dumps(s.test_rows) if s.test_rows is not None else f"n={s.n_test}"
        by_test_rows[key].append(s)

    candidates = []
    for group in by_test_rows.values():
        per_class: dict[str, set] = defaultdict(set)
        for s in group:
            per_class[s.model].add(s.model_obj)
        for model, objs in per_class.items():
            if len(objs) >= 2:
                lines = sorted({s.fit_line for s in group if s.model == model and s.fit_line})
                candidates.append({"model": model, "candidates": len(objs), "n_test": group[0].n_test,
                                   "fit_lines": lines})
    static = [i for i in code_items if i.get("family") == "E"]

    if candidates:
        c = max(candidates, key=lambda c: c["candidates"])
        summary = (f"{c['candidates']} {c['model']} candidates were each scored on the same {c['n_test']} held-out "
                   "rows, so those rows were used to choose the model; the reported score is the best of several.")
        if static:
            summary += f" The code also picks a value by its score on those rows (line {static[0]['line']})."
        status = "flagged"
    elif static:
        summary = (f"The code picks a setting or threshold by its score on the evaluation rows (line "
                   f"{static[0]['line']}); a single model was fitted, so the effect is probably small. Read as a "
                   "suspicion.")
        status = "clear"
    else:
        summary = "No choice between candidates was made on the evaluation rows."
        status = "clear"
    return _result(status, summary, measurements={"candidate_groups": len(candidates)},
                   evidence={"candidates": candidates, "code_review_items": static[:5]})


def _result(status: str, summary: str, **kwargs: Any) -> CheckResult:
    return CheckResult(check_id=CHECK_ID, status=status, summary=summary, **kwargs)  # type: ignore[arg-type]
