"""Check B: target or future information in features.

Suspects come from two sources, kept apart so the benchmark can compare them:
  - rules: column names that sound like the future or the answer, and features that
    predict the target almost perfectly on their own;
  - llm: the model reads column names, a few example values and the prediction_time
    sentence, and names columns that would not be known at that moment, with a reason.
Code then measures each suspect: its single-feature power, and a shuffle rerun that
scrambles that one column in a copy of the data and reruns the script unchanged. A
suspect is flagged only when the rerun shows the reported metric depends on it.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from auditor.agent.llm_client import BudgetExceeded, LLMClient
from auditor.checks.base import CheckResult
from auditor.config import AuditConfig
from auditor.sandbox import Sandbox, collect_files

CHECK_ID = "B_future_information"
# A shuffle rerun must cost the reported metric at least this much to count as dependence.
MIN_METRIC_DROP = 0.02
# Single-feature power at or above this is "near-perfect separability".
NEAR_PERFECT_POWER = 0.98
RULES_POWER_THRESHOLD = 0.90
MAX_RERUNS = 3
FUTURE_WORDS = re.compile(r"(next|future|after|final|result|outcome|label|target|later|leak|post_|_post)", re.I)


@dataclass(frozen=True)
class Suspect:
    column: str
    source: str  # "rules" or "llm"
    reason: str


# --- Measurements (plain code) ---------------------------------------------------------


def feature_power(data: pd.DataFrame, features: Sequence[str], target: str) -> dict[str, float | None]:
    """How well each feature alone predicts the target.

    Binary target: ROC-AUC of the raw feature, folded so 0.5 means no signal and 1.0 perfect.
    Numeric target: absolute Spearman correlation. Otherwise None.
    """
    y = data[target]
    binary = y.nunique(dropna=True) == 2
    numeric_target = pd.api.types.is_numeric_dtype(y) and not binary
    power: dict[str, float | None] = {}
    for col in features:
        if col not in data.columns:
            power[col] = None
            continue
        x = data[col]
        if not pd.api.types.is_numeric_dtype(x):
            x = x.astype("category").cat.codes.where(x.notna())
        mask = x.notna() & y.notna()
        if mask.sum() < 10 or x[mask].nunique() < 2:
            power[col] = None
        elif binary:
            labels = (y[mask] == sorted(y[mask].unique())[1]).astype(int)
            auc = roc_auc_score(labels, x[mask])
            power[col] = round(max(auc, 1 - auc), 4)
        elif numeric_target:
            power[col] = round(abs(x[mask].corr(y[mask], method="spearman")), 4)
        else:
            power[col] = None
    return power


def run_shuffled(
    config: AuditConfig, sandbox: Sandbox, column: str, seed: int = 0
) -> tuple[dict[str, Any] | None, str | None, Path]:
    """Rerun the script with one column of the data file shuffled. Returns (metrics, error, log_dir)."""
    data = pd.read_csv(config.project_dir / config.data_path)
    data[column] = data[column].sample(frac=1.0, random_state=seed).to_numpy()
    files = collect_files(config.project_dir)
    with tempfile.TemporaryDirectory(prefix="audit-shuffle-") as tmp:
        shuffled = Path(tmp) / "data.csv"
        data.to_csv(shuffled, index=False)
        files[Path(config.data_path).as_posix()] = shuffled
        run = sandbox.run(["python", config.entry_script], files=files, timeout=config.timeout_seconds)
    (run.log_dir / "shuffled_column.txt").write_text(column, encoding="utf-8")
    if not run.ok:
        return None, f"rerun failed (exit {run.exit_code}, timed out {run.timed_out})", run.log_dir
    path = run.artifacts.get("metrics.json")
    if path is None:
        return None, "rerun wrote no metrics.json", run.log_dir
    return json.loads(path.read_text(encoding="utf-8")), None, run.log_dir


# --- Suspect sources --------------------------------------------------------------------


def rules_suspects(features: Sequence[str], target: str, power: dict[str, float | None]) -> list[Suspect]:
    """The rules-only baseline: suspicious names, the target's own name, or very high single-feature power."""
    suspects = []
    target_key = _normalise(target)
    for col in features:
        reasons = []
        if FUTURE_WORDS.search(col):
            reasons.append(f"name contains '{FUTURE_WORDS.search(col).group(0)}'")
        if target_key and target_key in _normalise(col):
            reasons.append("name contains the target's name")
        p = power.get(col)
        if p is not None and p >= RULES_POWER_THRESHOLD:
            reasons.append(f"predicts the target alone with power {p}")
        if reasons:
            suspects.append(Suspect(col, "rules", "; ".join(reasons)))
    return suspects


SUSPECT_TOOL = "report_suspect_columns"
SUSPECT_INSTRUCTIONS = """You audit how a machine learning experiment was evaluated.
You are given the features a model was trained on, a few example values for each, the
target, and when predictions are made in real use. Name every feature whose value would
NOT be known at that moment: information from after the prediction time, or derived from
the target itself. Give a concrete reason for each. Do not list features just because they
are strongly predictive; being useful is fine if the value is available in time.
An empty list is a good answer when nothing is suspicious. Do not state any numbers.
Answer only by calling the report_suspect_columns tool."""


def llm_suspects(
    client: LLMClient, config: AuditConfig, features: Sequence[str], data: pd.DataFrame
) -> tuple[list[Suspect], dict[str, Any]]:
    """Ask the model which features would not be known at prediction time. Returns (suspects, notes)."""
    tool = {
        "type": "function",
        "name": SUSPECT_TOOL,
        "description": "Report features that would not be known at prediction time.",
        "parameters": {
            "type": "object",
            "properties": {
                "suspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "enum": list(features)},
                            "reason": {"type": "string"},
                        },
                        "required": ["column", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["suspects"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    prompt = json.dumps({
        "target": config.target,
        "prediction_time": config.prediction_time or "not stated",
        "features": [
            {"name": c, "examples": [_plain(v) for v in data[c].dropna().head(3)]} for c in features if c in data
        ],
    }, indent=2)
    notes: dict[str, Any] = {"model": client.model}
    try:
        response = client.respond(instructions=SUSPECT_INSTRUCTIONS, input=prompt, tools=[tool], purpose="check_b_suspects")
    except BudgetExceeded as exc:
        return [], {**notes, "error": str(exc)}
    notes["usage"] = response.usage
    calls = [c for c in response.tool_calls if c.name == SUSPECT_TOOL]
    if not calls or calls[0].arguments is None:
        notes["error"] = calls[0].error if calls else "the model answered without calling the tool"
        notes["text"] = response.text[:500]
        return [], notes

    suspects, rejected = [], []
    for item in calls[0].arguments.get("suspects", []):
        column = item.get("column") if isinstance(item, dict) else None
        if column in features:
            suspects.append(Suspect(column, "llm", str(item.get("reason", ""))[:500]))
        else:
            rejected.append(item)
    if rejected:
        notes["rejected_items"] = rejected
    return suspects, notes


# --- The check --------------------------------------------------------------------------


def check_future_info(
    config: AuditConfig,
    sandbox: Sandbox,
    data: pd.DataFrame,
    features: Sequence[str] | None,
    baseline_metrics: dict[str, Any],
    suspects: Sequence[Suspect],
    notes: dict[str, Any] | None = None,
    max_reruns: int = MAX_RERUNS,
) -> CheckResult:
    if not features:
        return _result("not_applicable", "The model's feature names are unknown (it was not trained on a DataFrame).")
    if config.target not in data.columns:
        return _result("error", f"Target column {config.target!r} is not in the data.")
    baseline = baseline_metrics.get(config.reported_metric)
    if not isinstance(baseline, (int, float)):
        return _result("error", f"Baseline metrics have no numeric {config.reported_metric!r}.")

    in_data = [f for f in features if f in data.columns]
    power = feature_power(data, in_data, config.target)
    by_column: dict[str, list[Suspect]] = {}
    for s in suspects:
        if s.column in in_data:
            by_column.setdefault(s.column, []).append(s)
    for col in in_data:
        p = power.get(col)
        if p is not None and p >= NEAR_PERFECT_POWER and col not in by_column:
            by_column[col] = [Suspect(col, "rules", f"near-perfect single-feature power {p}")]

    evidence: dict[str, Any] = {
        "features": list(features),
        "feature_power": power,
        "suspects": [asdict(s) for s in suspects],
        "suspect_notes": notes or {},
        "shuffle_reruns": [],
    }
    if not by_column:
        return _result("clear", "No feature was suspected of being unavailable at prediction time.", evidence=evidence)

    ranked = sorted(by_column, key=lambda c: -(power.get(c) or 0))
    findings, skipped = [], ranked[max_reruns:]
    for col in ranked[:max_reruns]:
        metrics, error, log_dir = run_shuffled(config, sandbox, col, seed=config.random_seed or 0)
        shuffled_value = metrics.get(config.reported_metric) if metrics else None
        drop = round(float(baseline) - float(shuffled_value), 4) if isinstance(shuffled_value, (int, float)) else None
        findings.append({
            "column": col,
            "sources": sorted({s.source for s in by_column[col]}),
            "reasons": [s.reason for s in by_column[col]],
            "single_feature_power": power.get(col),
            "baseline": baseline,
            "shuffled": shuffled_value,
            "metric_drop": drop,
            "depends_on_column": drop is not None and drop >= MIN_METRIC_DROP,
            "error": error,
            "log_dir": str(log_dir),
        })
    evidence["shuffle_reruns"] = findings
    if skipped:
        evidence["not_rerun"] = skipped

    flagged = [f for f in findings if f["depends_on_column"]]
    if flagged:
        worst = max(flagged, key=lambda f: f["metric_drop"])
        summary = (
            f"Shuffling '{worst['column']}' drops {config.reported_metric} from {worst['baseline']:.4f} to "
            f"{worst['shuffled']:.4f} ({worst['metric_drop']:+.4f} lost), and it is suspected of not being known at "
            f"prediction time ({'; '.join(worst['reasons'])[:200]})."
        )
        return _result("flagged", summary, measurements=_measurements(findings), evidence=evidence)
    if all(f["error"] for f in findings):
        return _result("error", "Every shuffle rerun failed, so no suspect could be measured.", evidence=evidence)
    names = ", ".join(f"'{f['column']}'" for f in findings)
    return _result(
        "clear",
        f"Suspected column(s) {names} were measured; shuffling them changed {config.reported_metric} by less than "
        f"{MIN_METRIC_DROP}, so the reported result does not depend on them.",
        measurements=_measurements(findings),
        evidence=evidence,
    )


def _measurements(findings: list[dict[str, Any]]) -> dict[str, Any]:
    drops = [f["metric_drop"] for f in findings if f["metric_drop"] is not None]
    return {
        "suspects_measured": len(findings),
        "suspects_with_dependence": sum(f["depends_on_column"] for f in findings),
        "max_metric_drop": max(drops) if drops else None,
    }


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _result(status: str, summary: str, **kwargs: Any) -> CheckResult:
    return CheckResult(check_id=CHECK_ID, status=status, summary=summary, **kwargs)  # type: ignore[arg-type]
