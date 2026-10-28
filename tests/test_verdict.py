from auditor.agent.tools import EvidenceItem
from auditor.agent.verdict import unsupported_numbers, validate_verdict

EVIDENCE = [
    EvidenceItem("E1", "run_baseline", {}, {"status": "reproduced", "metrics": {"accuracy": 0.952, "n_test": 10491}}),
    EvidenceItem("E2", "run_check", {"check": "C"}, {"status": "flagged", "summary": "48 of 48 test groups shared"}),
    EvidenceItem("E3", "run_check", {"check": "A"}, {"status": "clear"}),
    EvidenceItem("E4", "run_with_split", {"strategy": "group"}, {"metrics": {"accuracy": 0.8132}, "change_vs_baseline": -0.1388}),
]

GOOD = {
    "verdict": "confirmed",
    "summary": "Accuracy of 95.2% falls to 81.3% with a race-grouped split; 48 of 48 test races were in training.",
    "findings": [{"family": "C", "title": "Races in both train and test", "explanation": "See E2 and E4.",
                  "evidence_ids": ["E2", "E4"]}],
    "caveats": [{"text": "The grouped score drops by 13.9 points.", "evidence_ids": ["E4"]}],
}


def test_a_well_supported_verdict_is_accepted():
    assert validate_verdict(GOOD, EVIDENCE) == []


def test_invented_numbers_are_rejected():
    bad = {**GOOD, "summary": "Accuracy falls by 17% to 78.5%."}

    problems = validate_verdict(bad, EVIDENCE)

    assert any("17" in p and "78.5" in p for p in problems)


def test_rounding_percent_sign_and_thousands_are_understood():
    assert unsupported_numbers(["0.95, 95%, 95.20%, 10,491 rows, -0.14, 13.88 points"], EVIDENCE) == []


def test_small_integers_and_evidence_ids_are_not_numbers_to_check():
    assert unsupported_numbers(["We ran 3 checks (E4, E12) on F1 data."], EVIDENCE) == []


def test_finding_must_cite_a_flagged_check_of_its_family():
    bad = {**GOOD, "findings": [{"family": "A", "title": "x", "explanation": "y", "evidence_ids": ["E3", "E4"]}]}

    problems = validate_verdict(bad, EVIDENCE)

    assert any("run_check(A)" in p and "flagged" in p for p in problems)


def test_unknown_evidence_ids_are_rejected():
    bad = {**GOOD, "caveats": [{"text": "x", "evidence_ids": ["E99"]}]}

    assert any("E99" in p for p in validate_verdict(bad, EVIDENCE))


def test_confirmed_needs_findings_and_not_found_needs_all_checks():
    assert any("at least one finding" in p for p in validate_verdict({**GOOD, "findings": []}, EVIDENCE))

    not_found = {"verdict": "not_found", "summary": "Nothing found.", "findings": [], "caveats": []}
    assert any("missing: ['B']" in p for p in validate_verdict(not_found, EVIDENCE))


def test_inconclusive_has_no_findings():
    bad = {**GOOD, "verdict": "inconclusive"}

    assert any("no findings" in p for p in validate_verdict(bad, EVIDENCE))
