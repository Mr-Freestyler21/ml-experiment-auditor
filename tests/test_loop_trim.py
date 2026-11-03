import json

from auditor.agent.loop import KEEP_BULKY_STEPS, _trim_old_outputs


def test_old_file_contents_are_replaced_by_a_note_and_recent_ones_kept():
    history = [{"type": "function_call_output", "call_id": "a", "output": "x" * 1000},
               {"type": "function_call_output", "call_id": "b", "output": "y" * 1000}]
    bulky = [(0, 1, "read_file (E2)"), (1, 3, "read_file (E5)")]

    kept = _trim_old_outputs(history, bulky, step=1 + KEEP_BULKY_STEPS)

    assert kept == [(1, 3, "read_file (E5)")]
    assert history[0]["call_id"] == "a"
    assert "read_file (E2)" in json.loads(history[0]["output"])["note"]
    assert history[1]["output"] == "y" * 1000
