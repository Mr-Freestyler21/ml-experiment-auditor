"""Code review: read every Python file in the project for patterns that often leak.

Plain code, no model. Many leaks are created before training, in the script that builds the
data file, where the call tracer never looks. This reads every .py file in the project (the
training script and any preparation scripts) and reports suspicious lines with what they
usually mean and how to verify them.

Every item is a suspicion, not a verdict: a centred window is a leak in a feature but fine
in a label, a whole-history average is harmless for a fixed property. The agent weighs each
one and verifies it with measurements (a shuffle rerun under an honest split, or a patched
rerun of the pipeline). The check is "flagged" when it finds at least one item of high
severity, so the agent always looks; on its own it never confirms a flaw.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from auditor.checks.base import CheckResult

CHECK_ID = "R_code_review"
MAX_ITEMS = 30
RESAMPLERS = {"resample", "fit_resample", "SMOTE", "RandomOverSampler", "ADASYN"}
AGGREGATES = {"mean", "median", "sum", "max", "min", "std", "count", "size", "nunique", "quantile", "var"}
STATISTICS = {"mean", "std", "median", "var", "min", "max", "quantile"}
METRIC_FUNCTIONS = {"f1_score", "accuracy_score", "roc_auc_score", "precision_score", "recall_score",
                    "balanced_accuracy_score", "r2_score", "mean_squared_error", "mean_absolute_error"}
SPLITTERS = {"train_test_split", "GroupShuffleSplit", "GroupKFold", "TimeSeriesSplit", "KFold",
             "StratifiedKFold", "ShuffleSplit", "cross_val_score", "cross_validate"}


@dataclass(frozen=True)
class Item:
    file: str
    line: int
    code: str
    pattern: str
    family: str  # A, B, C, D, E or "quality"
    severity: str  # high, medium, low
    why: str
    verify: str
    column: str | None = None  # the data column this line builds, when it assigns one (df["x"] = ...)


def review_project(project_dir: Path, entry_script: str, time_column: str | None = None) -> CheckResult:
    files = sorted(p for p in Path(project_dir).rglob("*.py")
                   if not ({".venv", "__pycache__", ".ipynb_checkpoints"} & set(p.parts)))
    items: list[Item] = []
    unreadable = []
    for path in files:
        rel = path.relative_to(project_dir).as_posix()
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError:
            unreadable.append(rel)
            continue
        items += review_source(rel, source, tree, is_entry=(rel == entry_script), time_series=bool(time_column))
    order = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda i: (order[i.severity], i.file, i.line))
    high = [i for i in items if i.severity == "high"]
    status = "flagged" if high else "clear"
    summary = (f"Read {len(files)} Python file(s): {len(high)} high-severity and "
               f"{len(items) - len(high)} other suspicious line(s).")
    if high:
        first = high[0]
        summary += f" First: {first.file} line {first.line}, {first.pattern}."
    summary += " These are suspicions to verify with measurements, not proof."
    return CheckResult(
        check_id=CHECK_ID,
        status=status,  # type: ignore[arg-type]
        summary=summary,
        measurements={"files_read": len(files), "high": len(high), "total": len(items)},
        evidence={"items": [asdict(i) for i in items[:MAX_ITEMS]], "files": [p.relative_to(project_dir).as_posix()
                                                                             for p in files],
                  "unparseable_files": unreadable},
    )


def review_source(file: str, source: str, tree: ast.AST, is_entry: bool, time_series: bool) -> list[Item]:
    lines = source.splitlines()
    split_line = _first_split_line(tree)
    found: list[Item] = []
    # Variables holding a per-group aggregate (`per_pump = failures.groupby("pump_id").size()`),
    # so mapping them onto rows later is recognised too.
    aggregate_names = {
        t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) and _is_groupby_aggregate(n.value)
        for t in n.targets if isinstance(t, ast.Name)
    }

    columns_by_line = _assigned_columns(tree)

    def add(node: ast.AST, pattern: str, family: str, severity: str, why: str, verify: str) -> None:
        line = getattr(node, "lineno", 0)
        code = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        if not any(i.line == line and i.pattern == pattern and i.file == file for i in found):
            found.append(Item(file, line, code[:160], pattern, family, severity, why, verify,
                              columns_by_line.get(line)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            kw = {k.arg: k.value for k in node.keywords if k.arg}

            if name == "rolling" and _is_true(kw.get("center")):
                add(node, "centred rolling window", "B", "high",
                    "A centred window averages values after the current row, so the feature sees the future.",
                    "Recompute it as a trailing window (center=False) and rerun; or shuffle the feature under an "
                    "honest split.")

            if name == "shift" and _negative_number(node.args[0] if node.args else kw.get("periods")):
                add(node, "shift to future rows", "B", "medium",
                    "A negative shift reads later rows. Correct for building a label, a leak in a feature.",
                    "Check whether the result is a feature or the label.")

            if name in RESAMPLERS and (split_line is None or node.lineno < split_line):
                add(node, "resampling before the split", "D", "high",
                    "Copies of the same rows are made before train and test are separated, so identical rows "
                    "can land on both sides.",
                    "Run check D (shared rows); resample only the training rows.")

            if name == "sample" and _is_true(kw.get("replace")) and (split_line is None or node.lineno < split_line):
                add(node, "sampling with replacement before the split", "D", "high",
                    "Duplicated rows created before the split can appear in both train and test.",
                    "Run check D (shared rows).")

            if name == "transform" and node.args and _const_str(node.args[0]) in AGGREGATES and time_series:
                add(node, "per-group statistic over the whole history", "B", "medium",
                    "A group-wise mean, median or count over all rows uses values from later dates.",
                    "Compute it from earlier rows only (expanding window or shift(1)), then rerun.")

            if name in ("map", "merge") and time_series and (
                    _uses_full_groupby_aggregate(node) or _uses_names(node, aggregate_names)):
                add(node, "per-group aggregate mapped onto every row", "B", "medium",
                    "A count or rate per entity over the whole period includes events after each row's date.",
                    "Compute the rate from events before each row's date only.")

            if name == "merge" and "validate" not in kw:
                add(node, "join without a uniqueness check", "D", "low",
                    "If the right-hand key is not unique, the join silently duplicates rows.",
                    "Run check D for duplicated rows and duplicated keys.")

            if name == "filterwarnings" and node.args and _const_str(node.args[0]) == "ignore":
                add(node, "all warnings hidden", "quality", "low",
                    "Hides convergence and data warnings that explain weak or odd results.",
                    "Remove it and read the warnings.")

            if name == "LogisticRegression":
                c = _number(kw.get("C"))
                iters = _number(kw.get("max_iter"))
                if (c is not None and c <= 1e-3) or (iters is not None and iters < 100):
                    add(node, "handicapped baseline", "quality", "medium",
                        "Very strong regularisation or very few iterations make the baseline weak, so the "
                        "model's gain over it looks larger than it is.",
                        "Rerun the baseline with default settings.")

            if name in METRIC_FUNCTIONS and _inside_loop_or_selection(tree, node):
                add(node, "choice made by scoring on the evaluation data", "E", "medium",
                    "Settings or a threshold picked by their score on the same rows that are then reported "
                    "make the reported score optimistic.",
                    "Run check E; pick settings on a separate validation split or with cross-validation.")

        before_split = split_line is not None and getattr(node, "lineno", 0) < split_line
        if isinstance(node, ast.Assign) and (not is_entry or before_split):
            stat = _statistic_over_column(node)
            if stat:
                add(node, "statistic over all rows", "A", "medium" if not is_entry else "high",
                    f"{stat}() is computed over every row, including rows that end up in the test set, and used "
                    "to filter, scale or fill values.",
                    "Compute the statistic on training rows only and rerun.")

    return found


# --- AST helpers ------------------------------------------------------------------------


def _assigned_columns(tree: ast.AST) -> dict[int, str]:
    """Line -> column name, for every line of a statement like `df["x"] = ...`."""
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                key = target.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                        out[line] = key.value
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.JoinedStr):
                pass  # f-string column names cannot be resolved statically
    return out


def code_evidence_by_column(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Columns that the code shows are built from later rows (family B, high or medium)."""
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.get("family") == "B" and item.get("column") and item.get("severity") in ("high", "medium"):
            out.setdefault(item["column"], item)
    return out


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _const_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        return -inner if inner is not None else None
    return None


def _negative_number(node: ast.AST | None) -> bool:
    value = _number(node)
    return value is not None and value < 0


def _first_split_line(tree: ast.AST) -> int | None:
    lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and _call_name(n) in SPLITTERS]
    return min(lines) if lines else None


def _uses_full_groupby_aggregate(node: ast.Call) -> bool:
    """map(df.groupby(...).size()) or map(per_entity_series) built from a groupby aggregate."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and _call_name(sub) in AGGREGATES:
            inner = sub.func.value if isinstance(sub.func, ast.Attribute) else None
            if inner is not None and any(isinstance(n, ast.Call) and _call_name(n) == "groupby" for n in ast.walk(inner)):
                return True
    return False


def _is_groupby_aggregate(value: ast.AST) -> bool:
    if not (isinstance(value, ast.Call) and _call_name(value) in AGGREGATES and isinstance(value.func, ast.Attribute)):
        return False
    return any(isinstance(n, ast.Call) and _call_name(n) == "groupby" for n in ast.walk(value.func.value))


def _uses_names(node: ast.AST, names: set[str]) -> bool:
    return bool(names) and any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(node))


def _statistic_over_column(node: ast.Assign) -> str | None:
    """`z = (df[c] - df[c].mean()) / df[c].std()` or `df[x] = df[x].fillna(df[x].median())`."""
    for sub in ast.walk(node.value):
        if isinstance(sub, ast.Call) and _call_name(sub) in STATISTICS and isinstance(sub.func, ast.Attribute):
            base = sub.func.value
            if isinstance(base, (ast.Subscript, ast.Name)) and not _looks_like_split_name(base):
                parent_uses = [n for n in ast.walk(node.value) if isinstance(n, ast.BinOp)]
                fill = [n for n in ast.walk(node.value) if isinstance(n, ast.Call) and _call_name(n) == "fillna"]
                if parent_uses or fill:
                    return _call_name(sub)
    return None


def _looks_like_split_name(node: ast.AST) -> bool:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return isinstance(node, ast.Name) and any(w in node.id.lower() for w in ("train", "test", "val"))


def _inside_loop_or_selection(tree: ast.AST, target: ast.Call) -> bool:
    """A metric call inside a for loop, a comprehension, or a max/min/sorted key."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.ListComp, ast.GeneratorExp, ast.DictComp)) or (
                isinstance(node, ast.Call) and _call_name(node) in ("max", "min", "sorted")):
            if node is not target and any(sub is target for sub in ast.walk(node)):
                return True
    return False


def items_of(result: CheckResult) -> Iterable[dict[str, Any]]:
    return result.evidence.get("items", [])
