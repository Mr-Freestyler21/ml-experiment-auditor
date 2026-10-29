"""Score each system's flags against the answer keys and write the results table."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

FAMILIES = ("A", "B", "C")
SYSTEM_NAMES = {"rules": "Rules only", "agent": "Muse + tools (auditor)", "llm_only": "Muse only (no tools)"}
# A patched evaluation counts as corrected when it lands this close to the answer key's correct metric.
CORRECTION_TOLERANCE = 0.02


def score(results: list[dict[str, Any]], answers: dict[str, dict[str, Any]], split: str | None) -> dict[str, Any]:
    """results: [{"id", "system", "repeat", "result"}]; only the first repeat is scored here."""
    rows = [r for r in results if r["repeat"] == 0 and (split is None or answers[r["id"]]["split"] == split)]
    out: dict[str, Any] = {}
    for system in SYSTEM_NAMES:
        mine = [r for r in rows if r["system"] == system]
        if not mine:
            continue
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        clean_alarms = clean_total = exact = failures = 0
        corrected = flawed = 0
        estimate_errors, durations, costs = [], [], []
        for r in mine:
            truth = set(answers[r["id"]]["families"])
            res = r["result"]
            flagged = set(res.get("families", []))
            if res.get("error") or res.get("stopped_because") not in (None, "finalized"):
                failures += 1
            for f in FAMILIES:
                tp[f] += f in flagged and f in truth
                fp[f] += f in flagged and f not in truth
                fn[f] += f in truth and f not in flagged
            exact += flagged == truth
            if not truth:
                clean_total += 1
                clean_alarms += bool(flagged)
            else:
                flawed += 1
                correct = answers[r["id"]]["correct_metric"]
                if res.get("patched_metric") is not None and abs(res["patched_metric"] - correct) <= CORRECTION_TOLERANCE:
                    corrected += 1
            estimate = res.get("patched_metric") if system == "agent" else res.get("estimated_correct_metric")
            if truth and isinstance(estimate, (int, float)):
                estimate_errors.append(abs(estimate - answers[r["id"]]["correct_metric"]))
            durations.append(res.get("duration_s") or 0)
            costs.append(res.get("cost_usd") or 0)
        total_tp, total_fp = sum(tp.values()), sum(fp.values())
        out[system] = {
            "projects": len(mine),
            "recall": {f: _ratio(tp[f], tp[f] + fn[f]) for f in FAMILIES},
            "precision": _ratio(total_tp, total_tp + total_fp),
            "clean_false_alarms": f"{clean_alarms}/{clean_total}",
            "exact_match": f"{exact}/{len(mine)}",
            "corrected": f"{corrected}/{flawed}" if system == "agent" else "n/a",
            "mean_abs_error_of_corrected_metric": round(statistics.mean(estimate_errors), 4) if estimate_errors else None,
            "failures": failures,
            "median_duration_s": round(statistics.median(durations), 1) if durations else None,
            "mean_cost_usd": round(statistics.mean(costs), 4) if costs else None,
        }
    return out


def consistency(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For repeated agent runs: the verdict and families of each repeat, per project."""
    by_project: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for r in sorted(results, key=lambda r: r["repeat"]):
        if r["system"] == "agent":
            res = r["result"]
            by_project[r["id"]].append((res.get("verdict", "error"), tuple(res.get("families", []))))
    return {
        pid: {"runs": len(runs), "outcomes": [f"{v} {'+'.join(f) or '-'}" for v, f in runs],
              "agree": len(set(runs)) == 1}
        for pid, runs in by_project.items() if len(runs) > 1
    }


def _ratio(a: int, b: int) -> str:
    return f"{a}/{b}" if b else "n/a"


def markdown(scores: dict[str, dict[str, Any]], per_project: list[dict[str, Any]],
             consistency_rows: dict[str, dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = ["# Benchmark results", "", meta["description"], ""]
    for split, table in scores.items():
        if not table:
            continue
        lines += [f"## {split}", "",
                  "| System | Recall A | Recall B | Recall C | Precision | False alarms (clean) | Exact match "
                  "| Corrected | Mean abs error of corrected metric | Failures | Median time | Mean cost |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for system, s in table.items():
            r = s["recall"]
            err = "" if s["mean_abs_error_of_corrected_metric"] is None else f"{s['mean_abs_error_of_corrected_metric']:.4f}"
            lines.append(
                f"| {SYSTEM_NAMES[system]} | {r['A']} | {r['B']} | {r['C']} | {s['precision']} | "
                f"{s['clean_false_alarms']} | {s['exact_match']} | {s['corrected']} | {err} | {s['failures']} | "
                f"{s['median_duration_s']} s | ${s['mean_cost_usd']:.4f} |"
            )
        lines.append("")
    lines += ["## Per project", "",
              "| Project | Split | Planted flaw (taxonomy) | Correct metric | Rules | Muse + tools | Patched metric "
              "| Muse only | Muse-only estimate |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in per_project:
        lines.append(
            f"| {p['id']} | {p['split']} | {p['truth']} | {p['correct']:.4f} | {p['rules']} | {p['agent']} | "
            f"{p['patched']} | {p['llm_only']} | {p['estimate']} |"
        )
    if consistency_rows:
        lines += ["", "## Consistency of repeated audits", "", "| Project | Runs | Outcomes | Same every time |",
                  "|---|---|---|---|"]
        for pid, c in consistency_rows.items():
            lines.append(f"| {pid} | {c['runs']} | {'; '.join(c['outcomes'])} | {'yes' if c['agree'] else 'no'} |")
    lines += ["", "## Notes", ""] + [f"- {n}" for n in meta["notes"]]
    return "\n".join(lines) + "\n"
