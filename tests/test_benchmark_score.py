from benchmark.score import consistency, score

ANSWERS = {
    "a": {"split": "dev", "families": ["A"], "correct_metric": 0.70},
    "c": {"split": "dev", "families": ["C"], "correct_metric": 0.60},
    "clean": {"split": "heldout", "families": [], "correct_metric": 0.80},
}


def rec(pid, system, families, repeat=0, **extra):
    return {"id": pid, "system": system, "repeat": repeat, "result": {"families": families, **extra}}


def test_recall_precision_and_false_alarms():
    results = [
        rec("a", "rules", ["A"]), rec("c", "rules", []), rec("clean", "rules", ["B"]),
        rec("a", "agent", ["A"], verdict="confirmed", stopped_because="finalized", patched_metric=0.71),
        rec("c", "agent", ["C"], verdict="confirmed", stopped_because="finalized", patched_metric=0.65),
        rec("clean", "agent", [], verdict="not_found", stopped_because="finalized"),
    ]

    s = score(results, ANSWERS, None)

    assert s["rules"]["recall"] == {"A": "1/1", "B": "n/a", "C": "0/1"}
    assert s["rules"]["precision"] == "1/2"
    assert s["rules"]["clean_false_alarms"] == "1/1"
    assert s["agent"]["exact_match"] == "3/3"
    assert s["agent"]["corrected"] == "1/2"  # 0.65 is more than 0.02 from 0.60
    assert s["agent"]["mean_abs_error_of_corrected_metric"] == 0.03


def test_split_filter_and_failures():
    results = [rec("a", "agent", [], verdict="inconclusive", stopped_because="max_steps"),
               rec("clean", "agent", [], verdict="not_found", stopped_because="finalized")]

    dev = score(results, ANSWERS, "dev")

    assert dev["agent"]["projects"] == 1
    assert dev["agent"]["failures"] == 1
    assert dev["agent"]["recall"]["A"] == "0/1"


def test_only_first_repeat_is_scored_and_repeats_feed_consistency():
    results = [rec("a", "agent", ["A"], 0, verdict="confirmed"), rec("a", "agent", [], 1, verdict="not_found"),
               rec("a", "agent", ["A"], 2, verdict="confirmed")]

    assert score(results, ANSWERS, None)["agent"]["projects"] == 1
    c = consistency(results)["a"]
    assert c["runs"] == 3 and c["agree"] is False
    assert c["outcomes"] == ["confirmed A", "not_found -", "confirmed A"]
