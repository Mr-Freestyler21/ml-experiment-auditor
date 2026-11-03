import json

from auditor.agent.loop import KEEP_BULKY_STEPS, _short_output, _trim_old_outputs


def test_old_outputs_are_swapped_for_their_short_version_and_recent_ones_kept():
    history = [{"type": "function_call_output", "call_id": "a", "output": "x" * 1000},
               {"type": "function_call_output", "call_id": "b", "output": "y" * 1000}]
    bulky = [(0, 1, "short a"), (1, 3, "short b")]

    kept = _trim_old_outputs(history, bulky, step=1 + KEEP_BULKY_STEPS)

    assert kept == [(1, 3, "short b")]
    assert history[0] == {"type": "function_call_output", "call_id": "a", "output": "short a"}
    assert history[1]["output"] == "y" * 1000


def test_short_versions_keep_ids_verdict_and_summary():
    read = json.loads(_short_output("read_file", {"evidence_id": "E2", "content": "..."}))
    check = json.loads(_short_output("run_check", {"evidence_id": "E5", "status": "flagged", "summary": "s",
                                                   "measurements": {"m": 1}, "evidence": {"big": [1] * 100}}))
    patch = json.loads(_short_output("propose_patch", {"evidence_id": "E7", "accepted": True, "patch_id": "P1",
                                                       "diff": "---"}))

    assert read["evidence_id"] == "E2" and "call it again" in read["note"]
    assert (check["status"], check["summary"], check["measurements"]) == ("flagged", "s", {"m": 1})
    assert "evidence" not in check
    assert patch == {"evidence_id": "E7", "accepted": True, "patch_id": "P1"}
    assert _short_output("run_patched", {"metrics": {}}) is None


def test_budget_warnings_come_once_each_and_finalize_wins():
    from auditor.agent import prompts
    from auditor.agent.loop import budget_warning

    warned: set[str] = set()
    assert budget_warning(100, 1000, 20, warned) is None
    assert budget_warning(700, 1000, 20, warned) == prompts.WRAP_UP
    assert budget_warning(750, 1000, 19, warned) is None
    assert budget_warning(760, 1000, 2, warned) == prompts.FINALIZE_NOW
    assert budget_warning(900, 1000, 1, warned) is None
    assert budget_warning(900, 1000, 1, set()) == prompts.FINALIZE_NOW


def test_the_latest_read_of_a_file_is_kept_whole():
    history = [{"type": "function_call_output", "call_id": c, "output": "text " + c} for c in "abc"]
    bulky = [(0, 1, "short a"), (1, 1, "short b"), (2, 1, "short c")]

    kept = _trim_old_outputs(history, bulky, step=1 + KEEP_BULKY_STEPS, keep={1})

    assert [h["output"] for h in history] == ["short a", "text b", "short c"]
    assert kept == [(1, 1, "short b")]
