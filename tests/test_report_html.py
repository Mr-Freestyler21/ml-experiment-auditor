import json
from pathlib import Path

from auditor.report_html import render_html, write_html

REPO = Path(__file__).resolve().parents[1]
INJECTION = "<script>alert('x')</script>"


def report(**overrides):
    data = {
        "project": "demo",
        "verdict": "confirmed",
        "summary": "Groups leak.",
        "baseline": {"metric": "accuracy", "reported": 0.95, "measured": 0.952, "difference": 0.002,
                     "status": "reproduced", "random_seed": 42, "environment": {"python": "3.14.3", "packages": {}}},
        "context_metrics": {"majority_class_accuracy": 0.809},
        "findings": [{"family": "C", "family_name": "Same entities", "title": "Shared races", "explanation": "x",
                      "evidence_ids": ["E2"], "code": [{"line": 47, "role": "split", "code": "train_test_split(X, y)"}],
                      "evidence": [{"id": "E2", "tool": "run_check", "brief": "flagged"}]}],
        "patch": {"id": "P1", "path": "train.py", "rationale": "group it", "lines_changed": 2,
                  "diff": "--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-old line\n+new line\n"},
        "evaluations": [
            {"label": "Original evaluation (rerun)", "evidence_id": "E1", "accuracy": 0.952, "change_vs_baseline": 0.0,
             "majority_class_accuracy_test": None},
            {"label": "Patched script (P1)", "evidence_id": "E3", "accuracy": 0.705, "change_vs_baseline": -0.247,
             "majority_class_accuracy_test": 0.651, "target_rate_train": 0.15, "target_rate_test": 0.35,
             "shared_groups": 0, "checks_after_patch": {"A": "clear", "C_groups": "clear"}},
        ],
        "caveats": [{"text": "Label shift.", "evidence_ids": ["E3"]}],
        "limits": [],
        "cost": {"model": "muse-spark-1.3", "steps": 8, "duration_s": 157.1, "calls": 8, "tokens": 57006,
                 "cap": 200000, "est_cost_usd": 0.0515},
        "evidence_index": [{"id": "E1", "tool": "run_baseline", "arguments": {}, "brief": "reproduced"}],
    }
    data.update(overrides)
    return data


def test_page_has_verdict_chart_fix_and_evidence_anchors():
    page = render_html(report())

    assert "CONFIRMED" in page
    assert "<svg" in page and page.count("class='bar'") == 2
    assert page.count("class='majority'") == 2  # the original row falls back to the context baseline
    assert "<span class='del'>-old line</span>" in page and "<span class='add'>+new line</span>" in page
    assert "<tr id='E1'>" in page and "href='#E2'" in page
    assert "train_test_split(X, y)" in page


def test_untrusted_text_is_escaped_everywhere():
    page = render_html(report(
        project=INJECTION, summary=INJECTION,
        findings=[{**report()["findings"][0], "title": INJECTION, "explanation": INJECTION,
                   "code": [{"line": 1, "role": INJECTION, "code": INJECTION}]}],
        caveats=[{"text": INJECTION, "evidence_ids": ["E1"]}],
        evaluations=[{**report()["evaluations"][1], "label": INJECTION}],
        evidence_index=[{"id": "E1", "tool": "read_file", "arguments": {"path": INJECTION}, "brief": INJECTION}],
    ))

    assert INJECTION not in page
    assert "&lt;script&gt;" in page
    assert page.count("<script>") == 1  # only the page's own tooltip script


def test_not_found_and_inconclusive_pages_render_without_findings_or_patch():
    for verdict, label in (("not_found", "NOT FOUND"), ("inconclusive", "INCONCLUSIVE")):
        page = render_html(report(verdict=verdict, findings=[], patch=None, evaluations=[]))
        assert label in page
        assert "No evaluation flaw was confirmed" in page
        assert "The fix" not in page and "<svg" not in page


def test_committed_examples_render(tmp_path):
    for name in ("f1", "clean"):
        source = REPO / "examples" / name / "report.json"
        copy = tmp_path / name / "report.json"
        copy.parent.mkdir()
        copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        page = write_html(copy).read_text(encoding="utf-8")

        assert json.loads(source.read_text(encoding="utf-8"))["verdict"].replace("_", " ").upper() in page
