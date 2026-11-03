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
