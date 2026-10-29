"""Combine benchmark runs into the committed results table.

    python -m benchmark.combine --main dev_run3 heldout_run --repeats repeats

Scores come from the first run of every project in the `--main` folders (a later folder wins
for the same project and system). Consistency comes from the `--repeats` folders.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from benchmark.run import RESULTS, ROOT, load_answers, write_summary

HISTORY = [
    "How the auditor was developed against this benchmark (all on the development set; the held-out "
    "set was run once, at commit 96ab701, after the last change):",
    "Dev run 1 exposed: pandas-only preprocessing invisible to the tracer (check A now also reads the "
    "script); no patch proposed for leaky features (prompt: the fix is removing the feature); runs lost "
    "to network drops (longer retries); prediction_time wording that contradicted the data (reworded; "
    "data and answer keys unchanged).",
    "Dev run 2 exposed: a procedural flaw with no score effect left unconfirmed (prompt: confirm A and C "
    "flaws even when the effect is small); synthetic hospital data that was implausible, which Muse "
    "pointed out (regenerated); expired reasoning items in the kept history (left out).",
    "Dev run 3: check B confirmations calibrated to 'clearly not known at prediction time', otherwise "
    "inconclusive. This removed a false alarm and turned the 3-day window leak into an honest "
    "inconclusive.",
    "Runs that stopped on network outages were rerun; the table counts only completed audits.",
]


def records_in(names: list[str]) -> list[dict[str, Any]]:
    merged: dict[tuple, dict[str, Any]] = {}
    for name in names:
        for path in sorted((RESULTS / name / "raw").glob("*.json")):
            r = json.loads(path.read_text(encoding="utf-8"))
            merged[(r["id"], r["system"], r["repeat"])] = r
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", nargs="+", required=True)
    parser.add_argument("--repeats", nargs="*", default=[])
    args = parser.parse_args()
    answers = load_answers()
    main_records = [r for r in records_in(args.main) if r["repeat"] == 0]
    ids = [i for i in answers if any(r["id"] == i for r in main_records)]
    write_summary(ROOT, main_records, answers, ids, consistency_records=records_in(args.repeats),
                  extra_notes=HISTORY, name="RESULTS")


if __name__ == "__main__":
    main()
