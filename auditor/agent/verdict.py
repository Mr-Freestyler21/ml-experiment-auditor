"""Validate the model's final verdict before it is accepted.

Rules (CLAUDE.md section 4):
  - every finding and caveat cites evidence ids that exist;
  - a finding of family X cites a run_check(X) result whose status is "flagged";
  - confirmed needs at least one finding; not_found and inconclusive have none;
  - not_found needs checks A, B and C all to have run without error;
  - every number in the text appears in some tool result (allowing rounding, percentages
    and sign), so the model cannot introduce numbers of its own.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from auditor.agent.tools import EvidenceItem

VERDICTS = ("confirmed", "not_found", "inconclusive")
FAMILIES = ("A", "B", "C")
# Small whole numbers ("three checks", "2 splits") are allowed without a source.
FREE_INTEGER_MAX = 10
NUMBER = re.compile(r"(?<![A-Za-z_\d])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


def validate_verdict(verdict: dict[str, Any], evidence: Sequence[EvidenceItem]) -> list[str]:
    problems: list[str] = []
    by_id = {e.id: e for e in evidence}
    v = verdict.get("verdict")
    if v not in VERDICTS:
        return [f"verdict must be one of {list(VERDICTS)}"]
    findings = verdict.get("findings") or []
    caveats = verdict.get("caveats") or []
    if not isinstance(verdict.get("summary"), str) or not verdict["summary"].strip():
        problems.append("summary must be a non-empty string")

    for i, f in enumerate(findings):
        label = f"finding {i + 1}"
        if f.get("family") not in FAMILIES:
            problems.append(f"{label}: family must be A, B or C")
            continue
        ids = _check_ids(f.get("evidence_ids"), by_id, label, problems)
        if not any(_is_flagged_check(by_id[i_], f["family"]) for i_ in ids):
            problems.append(
                f"{label}: a family {f['family']} finding must cite a run_check({f['family']}) result whose status "
                "is flagged. A score change alone is not proof; put observations without a flagged check in caveats."
            )
    for i, c in enumerate(caveats):
        _check_ids(c.get("evidence_ids"), by_id, f"caveat {i + 1}", problems)

    if v == "confirmed" and not findings:
        problems.append("confirmed needs at least one finding")
    if v in ("not_found", "inconclusive") and findings:
        problems.append(f"{v} must have no findings; move observations to caveats")
    if v == "not_found":
        missing = [fam for fam in FAMILIES if not any(_is_check(e, fam) and not e.result.get("error") for e in evidence)]
        if missing:
            problems.append(f"not_found needs checks A, B and C all run without error; missing: {missing}")

    texts = [verdict.get("summary", "")]
    texts += [f.get("title", "") + " " + f.get("explanation", "") for f in findings]
    texts += [c.get("text", "") for c in caveats]
    unsupported = unsupported_numbers(texts, evidence)
    if unsupported:
        problems.append(
            f"these numbers do not appear in any tool result: {unsupported}. Use only numbers returned by tools "
            "(rounding is fine); if you need a difference or percentage that no tool returned, describe it in words."
        )
    return problems


def unsupported_numbers(texts: Iterable[str], evidence: Sequence[EvidenceItem]) -> list[str]:
    known = sorted({abs(x) for e in evidence for x in _numbers_in(e.result)})
    bad = []
    for text in texts:
        for match in NUMBER.finditer(text or ""):
            token = match.group(1)
            value = float(token.replace(",", ""))
            if value.is_integer() and value <= FREE_INTEGER_MAX:
                continue
            decimals = len(token.split(".")[1]) if "." in token else 0
            tolerance = 0.5 * 10 ** -decimals + 1e-9
            if not any(abs(value - k) <= tolerance or abs(value - 100 * k) <= tolerance for k in known):
                bad.append(token)
    return sorted(set(bad))


def _numbers_in(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        for m in NUMBER.finditer(obj):
            yield float(m.group(1).replace(",", ""))
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _numbers_in(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _numbers_in(value)


def _check_ids(ids: Any, by_id: dict[str, EvidenceItem], label: str, problems: list[str]) -> list[str]:
    if not isinstance(ids, list) or not ids:
        problems.append(f"{label}: must cite at least one evidence id")
        return []
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        problems.append(f"{label}: unknown evidence ids {unknown}")
    return [i for i in ids if i in by_id]


def _is_check(e: EvidenceItem, family: str) -> bool:
    return e.tool == "run_check" and e.arguments.get("check") == family


def _is_flagged_check(e: EvidenceItem, family: str) -> bool:
    return _is_check(e, family) and e.result.get("status") == "flagged"
