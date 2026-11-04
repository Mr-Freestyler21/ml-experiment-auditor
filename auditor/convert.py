"""Turn a project without an audit.yaml (a notebook, a script, a folder or a zip) into an audit target.

Plain code does the work, no model: it finds the training code and the CSV it reads, makes the
script runnable in the sandbox (plots and display-only lines removed, data copied in, a short
marked block that writes metrics.json), then reruns it once to measure the reported value.

What code cannot know for sure is only suggested, never switched on: prediction_time,
group_columns and time_column go into audit.yaml as comments for a person to confirm. A wrong
group column causes false alarms, so a guess there would do harm.

The originals are never modified: everything is written to a new output folder.
"""

from __future__ import annotations

import ast
import functools
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import yaml
from pyflakes.api import check as pyflakes_check
from pyflakes.reporter import Reporter

from auditor.baseline import run_baseline
from auditor.config import CONFIG_NAME, ConfigError, load_config
from auditor.sandbox import Sandbox
from auditor.sandbox.base import SKIP_DIRS

ENTRY_NAME = "train.py"
NOTES_NAME = "CONVERSION.md"

# Import names available in the sandbox: sandbox-requirements.txt plus scikit-learn's own dependencies.
SANDBOX_MODULES = {"numpy", "pandas", "sklearn", "scipy", "joblib", "threadpoolctl"}
PLOT_MODULES = {"matplotlib", "seaborn", "plotly", "bokeh", "altair", "missingno"}
PLOT_METHODS = {"plot", "hist", "boxplot", "scatter_matrix"}
# Notebook lines like `df.head()` only display something; they are dropped from notebooks.
DISPLAY_METHODS = {"head", "tail", "describe", "info", "sample", "value_counts", "nunique", "unique",
                   "isnull", "isna", "sum", "mean", "corr", "count", "dtypes", "to_frame"}
READ_FUNCTIONS = {"read_csv", "read_table", "read_parquet", "read_excel", "read_json", "read_feather"}
SKIP_SOURCE_DIRS = SKIP_DIRS | {".ipynb_checkpoints"}

# sklearn metric function -> (metrics.json key, kind). "label" metrics compare predicted labels,
# "score" metrics take probabilities or decision scores.
METRICS = {
    "accuracy_score": ("accuracy", "label"),
    "balanced_accuracy_score": ("balanced_accuracy", "label"),
    "precision_score": ("precision", "label"),
    "recall_score": ("recall", "label"),
    "f1_score": ("f1", "label"),
    "classification_report": ("accuracy", "label"),
    "confusion_matrix": ("accuracy", "label"),
    "roc_auc_score": ("roc_auc", "score"),
    "average_precision_score": ("pr_auc", "score"),
    "r2_score": ("r2", "regression"),
    "mean_absolute_error": ("mae", "regression"),
    "mean_squared_error": ("mse", "regression"),
    "root_mean_squared_error": ("rmse", "regression"),
}

TARGET_VARIABLES = {"target", "TARGET", "target_col", "target_column", "target_name", "label", "LABEL", "label_col"}
Y_VARIABLES = {"y", "Y", "y_all", "labels"}
TIME_NAME = re.compile(r"(?i)(^|[_\s])(year|season|date|time|timestamp|month|week|day|period)s?($|[_\s])")
ID_NAME = re.compile(r"(?i)(^id$|[_\s]id$|^id[_\s]|[a-z]Id$)")
ENTITY_WORDS = {"patient", "user", "customer", "driver", "race", "store", "session", "subject", "account",
                "player", "team", "school", "hospital", "site", "device", "household", "company", "client",
                "member", "student", "match", "game", "event", "region", "city"}


class ConvertError(ValueError):
    """The input cannot be converted at all (as opposed to a problem listed in the result)."""


@dataclass
class ConversionResult:
    out_dir: Path
    source: str
    config: dict[str, Any] = field(default_factory=dict)
    suggestions: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)      # what the converter did
    problems: list[str] = field(default_factory=list)   # what blocks an audit until fixed
    metrics: dict[str, Any] = field(default_factory=dict)
    already_configured: bool = False

    @property
    def ready(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()}
        data["out_dir"] = str(self.out_dir)
        data["ready"] = self.ready
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ConversionResult:
        fields = {k: v for k, v in data.items() if k != "ready"}
        return cls(**{**fields, "out_dir": Path(data["out_dir"])})


# --- entry point ---

def convert(source: Path, out_dir: Path, sandbox: Sandbox | None = None, *, script: str | None = None,
            target: str | None = None, metric: str | None = None, reported_value: float | None = None,
            cells: int | None = None) -> ConversionResult:
    """Convert `source` (folder, .zip, .py or .ipynb) into an audit target in `out_dir`.

    With a sandbox, the converted script is run once to check it works and to measure the
    reported value. Without one, `reported_value` must be given or it is listed as a problem.
    """
    source = Path(source).resolve()
    out_dir = Path(out_dir).resolve()
    if not source.exists():
        raise ConvertError(f"{source} does not exist")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ConvertError(f"output folder {out_dir} is not empty")
    if source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="audit-convert-") as tmp:
            root = extract_zip(source.read_bytes(), Path(tmp))
            return convert(root, out_dir, sandbox, script=script, target=target, metric=metric,
                           reported_value=reported_value, cells=cells)

    root, entry = _find_entry(source, script)
    result = ConversionResult(out_dir=out_dir, source=source.name)
    if (root / CONFIG_NAME).is_file() and entry is None:
        result.already_configured = True
        result.notes.append(f"{CONFIG_NAME} already exists, so nothing was converted")
        return result
    assert entry is not None

    if entry.suffix == ".ipynb":
        code = _notebook_to_script(entry, cells, result)
        entry_rel = PurePosixPath(ENTRY_NAME)
    else:
        code = entry.read_text(encoding="utf-8")
        entry_rel = PurePosixPath(entry.relative_to(root).as_posix())
        result.notes.append(f"Entry script: {entry_rel} (kept as written, apart from the changes below)")
    if not script and source.is_dir():
        others = [p for p in _training_files(root) if p != entry]
        if others:
            result.notes.append(f"Chose {entry.relative_to(root).as_posix()} as the experiment over "
                                f"{', '.join(p.relative_to(root).as_posix() for p in others)} "
                                "(not a test, runs at the top level, named like a training script); "
                                "change with --script")

    code = _remove_plotting(code, result)
    code = _modernise_imports(code, result)
    if entry.suffix == ".ipynb":
        code = _tidy_cells(code)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(code)
    constants = {**_imported_constants(tree, root, entry), **_string_constants(tree)}
    hint = target or next((key for key, _ in _target_candidates(tree, constants)), None)
    before = len(result.problems)
    code, data_rel, columns = _bring_data(code, root, entry.parent, out_dir, result, constants, hint)
    explained = {"data_path": len(result.problems) > before}  # a specific reason was already given
    _copy_local_modules(code, root, out_dir, result)
    _check_imports(code, root, result)
    _check_package(code, entry_rel, result)

    tree = ast.parse(code)
    constants = {**_imported_constants(tree, root, entry), **_string_constants(tree)}
    detected_target = target or _detect_target(tree, constants, columns, result)
    if target:
        result.notes.append(f"Target column: {target} (given on the command line)")
    if detected_target and columns and detected_target not in columns:
        result.problems.append(f"target {detected_target!r} is not a column of {data_rel}")

    if "AUDIT_OUTPUT_DIR" in code:
        result.notes.append("The script already writes metrics.json, so no output block was added")
        default_metric = None
    else:
        before = len(result.problems)
        footer, default_metric, place = _metrics_footer(tree, code, result)
        explained["reported_metric"] = len(result.problems) > before
        if footer and place is None:
            code = code.rstrip() + "\n\n" + footer
        elif footer:
            code = _insert_block(code, footer, *place)
    _check_undefined_names(code, result)

    (out_dir / Path(*entry_rel.parts)).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / Path(*entry_rel.parts)).write_text(code, encoding="utf-8")

    result.config = {
        "entry_script": str(entry_rel),
        "data_path": data_rel or "",
        "target": detected_target or "",
        "reported_metric": metric or default_metric or "",
        "reported_value": reported_value,
        "random_seed": _random_seed(tree),
        "timeout_seconds": 600,
    }
    missing = {"data_path": "the data file could not be found",
               "target": "the target column could not be detected (command line: --target)",
               "reported_metric": "the metric to audit could not be detected; give the name of a value your "
                                  "script writes to metrics.json (command line: --metric)"}
    for key, message in missing.items():
        if not result.config[key] and not explained.get(key):
            result.problems.append(message)
    if columns:
        result.suggestions = suggest_settings(out_dir / data_rel, detected_target)

    _measure(result, sandbox, reported_value)
    write_outputs(result)
    return result


# --- finding the code ---

def _find_entry(source: Path, script: str | None) -> tuple[Path, Path | None]:
    """Return (project root, entry file). Entry is None when the folder already has audit.yaml."""
    if source.is_file():
        if source.suffix not in (".py", ".ipynb"):
            raise ConvertError("input must be a folder, a .zip, a .py script or a .ipynb notebook")
        return source.parent, source
    if script:
        entry = (source / script).resolve()
        if not entry.is_file() or source not in entry.parents:
            raise ConvertError(f"--script {script!r} is not a file inside {source}")
        return source, entry
    if (source / CONFIG_NAME).is_file():
        return source, None

    candidates = _training_files(source)
    if not candidates:
        raise ConvertError("found no script or notebook that trains a model (.fit) and scores it")
    if len(candidates) > 1:
        ranked = sorted(candidates, key=lambda p: _entry_rank(p, source))
        if _entry_rank(ranked[0], source) == _entry_rank(ranked[1], source):
            names = ", ".join(c.relative_to(source).as_posix() for c in candidates)
            raise ConvertError(f"several files train and score a model equally ({names}); keep only the one "
                               "to audit, or choose one with --script")
        return source, ranked[0]
    return source, candidates[0]


def _training_files(root: Path) -> list[Path]:
    """Scripts and notebooks that fit a model and score it."""
    found = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.suffix not in (".py", ".ipynb") or SKIP_SOURCE_DIRS.intersection(rel.parts):
            continue
        text = _searchable_text(path)
        if ".fit(" in text and any(f"{name}(" in text for name in [*METRICS, ".score"]):
            found.append(path)
    return found


ENTRY_WORDS = {"train", "training", "model", "main", "run", "experiment", "pipeline", "analysis", "fit"}


def _entry_rank(path: Path, root: Path) -> tuple[int, int, int, int, int]:
    """Lower is a likelier entry: not a test, runs its experiment at the top level, a telling name,
    shallow, and a script over a notebook (a script is usually the cleaned-up version)."""
    rel = path.relative_to(root)
    is_test = (any(part in ("tests", "test") for part in rel.parts[:-1])
               or path.stem.startswith("test_") or path.stem.endswith("_test") or path.stem == "conftest")
    if path.suffix == ".ipynb":
        top_level = True
    else:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            tree = None
        top_level = tree is not None and any(_call_name(c) in METRICS for c in _module_level_calls(tree))
    named = bool(ENTRY_WORDS & set(re.split(r"[^a-z]+", path.stem.lower())))
    return (int(is_test), int(not top_level), int(not named), len(rel.parts), int(path.suffix == ".ipynb"))


def _code_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        return "\n".join(_notebook_code_cells(path))
    return path.read_text(encoding="utf-8", errors="replace")


def _searchable_text(path: Path) -> str:
    """Code text for searching a folder: a file that cannot be read (an empty or broken notebook)
    is skipped instead of stopping the conversion. Converting it directly still reports why."""
    try:
        return _code_text(path)
    except ConvertError:
        return ""


def _notebook_code_cells(path: Path) -> list[str]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConvertError(f"{path.name} is not a readable notebook: {exc}") from None
    cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            cells.append("".join(source) if isinstance(source, list) else source)
    return cells


# --- notebooks ---

def _notebook_to_script(path: Path, cells: int | None, result: ConversionResult) -> str:
    """Code cells of the first complete experiment, without magics and display-only lines."""
    raw_cells = _notebook_code_cells(path)
    parsed: list[tuple[int, str, ast.Module]] = []
    for number, text in enumerate(raw_cells, start=1):
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("%", "!")))
        try:
            parsed.append((number, text, ast.parse(text)))
        except SyntaxError:
            result.notes.append(f"Skipped code cell {number}: not plain Python (IPython syntax?)")

    last = cells or _first_experiment_end(parsed)
    if last is None:
        raise ConvertError(f"{path.name}: no code cell scores a model after fitting it")
    kept = [(n, t, tree) for n, t, tree in parsed if n <= last]
    result.notes.append(f"Converted {path.name}: code cells 1 to {last} of {len(raw_cells)}"
                        + ("" if cells else " (up to the first cell that scores a fitted model; change with --cells)"))

    parts, dropped = [], 0
    for number, text, tree in kept:
        lines = text.splitlines()
        remove = {i for node in tree.body if _is_display_only(node)
                  for i in range(node.lineno - 1, node.end_lineno)}
        dropped += sum(1 for node in tree.body if _is_display_only(node))
        body = "\n".join(line for i, line in enumerate(lines) if i not in remove).strip()
        if body:
            parts.append(f"# Code cell {number}\n{body}")
    if dropped:
        result.notes.append(f"Removed {dropped} display-only line(s) such as df.head()")
    header = f'"""Converted from {path.name} (code cells 1-{last}) by the auditor\'s converter. See {NOTES_NAME}."""'
    return header + "\n\n" + "\n\n".join(parts) + "\n"


def _tidy_cells(code: str) -> str:
    """Drop markers of cells that removals emptied, and the blank runs removals left behind."""
    blocks = re.split(r"(?m)^(?=# Code cell \d+$)", code)
    code = "".join(b for b in blocks if not re.fullmatch(r"# Code cell \d+\s*", b))
    code = re.sub(r"(?m)^(# Code cell \d+)\n\s*\n", r"\1\n", code)
    return re.sub(r"\n{3,}", "\n\n", code)


def _first_experiment_end(cells: list[tuple[int, str, ast.Module]]) -> int | None:
    fitted = False
    for number, _, tree in cells:
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        fitted = fitted or any(_call_name(c) == "fit" for c in calls)
        if fitted and any(_call_name(c) in METRICS or _call_name(c) == "score" for c in calls):
            return number
    return None


def _is_display_only(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    value = node.value
    if not isinstance(value, ast.Call):
        return not isinstance(value, ast.Constant)  # a bare `df` or `df.shape`; keep docstrings
    name = _call_name(value)
    return name == "display" or (isinstance(value.func, ast.Attribute) and name in DISPLAY_METHODS)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


# --- old scikit-learn modules ---

# Modules scikit-learn merged into sklearn.model_selection in 0.18 and removed in 0.20. Their
# common functions (train_test_split, cross_val_score, GridSearchCV...) kept their names there.
OLD_SKLEARN_MODULES = ("cross_validation", "grid_search", "learning_curve")


def _modernise_imports(code: str, result: ConversionResult) -> str:
    """Point imports of removed scikit-learn modules at sklearn.model_selection, so an older
    tutorial runs on the sandbox's scikit-learn. Only the import lines change."""
    changed = []
    for old in OLD_SKLEARN_MODULES:
        patterns = [
            (rf"^(\s*)from\s+sklearn\.{old}\s+import\b", r"\1from sklearn.model_selection import"),
            (rf"^(\s*)from\s+sklearn\s+import\s+{old}\s*$", rf"\1from sklearn import model_selection as {old}"),
            (rf"^(\s*)import\s+sklearn\.{old}\s+as\s+(\w+)\s*$", r"\1import sklearn.model_selection as \2"),
        ]
        for pattern, replacement in patterns:
            code, n = re.subn(pattern, replacement, code, flags=re.MULTILINE)
            if n:
                changed.append(old)
    if changed:
        names = ", ".join(f"sklearn.{m}" for m in dict.fromkeys(changed))
        result.notes.append(f"Updated imports of {names} (removed in scikit-learn 0.20) to sklearn.model_selection, "
                            "where the same functions live now; old-style calls such as KFold(n, n_folds=...) are "
                            "not translated")
    return code


# --- plotting ---

def _remove_plotting(code: str, result: ConversionResult) -> str:
    """Remove plotting statements and anything that depends on their results.

    Top-level statements are deleted; nested ones become `pass` so blocks stay valid.
    """
    tree = ast.parse(code)
    tainted = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            for alias in node.names:
                top = (module or alias.name).split(".")[0]
                if top in PLOT_MODULES:
                    tainted.add((alias.asname or alias.name).split(".")[0])

    removed: list[ast.stmt] = []

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if _is_plotting(node, tainted):
                removed.append(node)
                tainted.update(_stored_names(node))
                continue
            for name in ("body", "orelse", "finalbody"):
                visit(getattr(node, name, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                visit(handler.body)

    visit(tree.body)
    if not removed:
        return code
    lines = code.splitlines()
    for node in sorted(removed, key=lambda n: n.lineno, reverse=True):
        start, end = node.lineno - 1, node.end_lineno
        replacement = [] if node.col_offset == 0 else [" " * node.col_offset + "pass  # converter: removed plotting"]
        lines[start:end] = replacement
    result.notes.append(f"Removed {len(removed)} plotting statement(s) (no plotting libraries in the sandbox)")
    return "\n".join(lines) + "\n"


def _is_plotting(node: ast.stmt, tainted: set[str]) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return all((alias.asname or alias.name).split(".")[0] in tainted for alias in node.names)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    header = _header_nodes(node)
    for part in header:
        for sub in ast.walk(part):
            if isinstance(sub, ast.Name) and sub.id in tainted:
                return True
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in PLOT_METHODS:
                return True
    return False


def _header_nodes(node: ast.stmt) -> list[ast.AST]:
    """The parts of a statement that run before (or instead of) its nested body."""
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return [node.target, node.iter]
    if isinstance(node, (ast.If, ast.While)):
        return [node.test]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return list(node.items)
    if isinstance(node, ast.Try):
        return []
    return [node]


def _stored_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


# --- data ---

def _bring_data(code: str, root: Path, script_dir: Path, out_dir: Path, result: ConversionResult,
                constants: dict[str, str], target: str | None) -> tuple[str, str | None, list[str]]:
    """Copy the file the script reads into the output and point the script at it.

    When the script names no file the converter can resolve (the read happens in another module,
    or the path is built at run time), the project is searched for the CSV instead.
    """
    tree = ast.parse(code)
    reads = sorted((node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and _call_name(node) in READ_FUNCTIONS and node.args),
                   key=lambda n: (n.lineno, n.col_offset))
    if len(reads) > 1:
        result.notes.append(f"The script reads {len(reads)} files; the first one is used as data_path")
    read = reads[0] if reads else None
    literal = _path_literal(read.args[0], constants) if read else None
    if read is not None and _call_name(read) not in ("read_csv", "read_table"):
        result.problems.append(f"the data is read with {_call_name(read)}; the auditor supports CSV files only")

    found = _locate(literal, root, script_dir) if literal else None
    if found is None:
        if read is None and not any(f"{name}(" in _project_code(root) for name in ("read_csv", "read_table")):
            result.problems.append("the script does not read a data file with pandas (pd.read_csv)")
            return code, None, []
        found = _search_data(root, target, literal, result)
        if found is None:
            return code, None, []
        where = f"line {read.lineno} of the script" if read else "another file of the project"
        result.notes.append(f"Found the data by searching the project: {found.relative_to(root).as_posix()} "
                            f"(read in {where}{f', as {literal!r}' if literal else ''})")
    inside = root in found.parents
    data_rel = found.relative_to(root).as_posix() if inside else f"data/{found.name}"
    dest = out_dir / Path(*PurePosixPath(data_rel).parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, dest)
    result.notes.append(f"Data: {data_rel} ({found.stat().st_size // 1024} KB)")

    if read is not None and isinstance(read.args[0], ast.Constant) and literal.replace("\\", "/") != data_rel:
        code = _replace_segment(code, read.args[0], repr(data_rel))
        result.notes.append(f"Changed the data path {literal!r} to {data_rel!r} so it works inside the sandbox")
    elif read is not None and not isinstance(read.args[0], ast.Constant) and not inside:
        result.problems.append(f"line {read.lineno}: the data path points outside the project; "
                               f"change it to {data_rel!r}")
    return code, data_rel, _columns(found)


def _columns(path: Path) -> list[str]:
    try:
        return [str(c) for c in pd.read_csv(path, nrows=0).columns]
    except Exception:  # noqa: BLE001 - any unreadable file is reported, not raised
        return []


def _project_code(root: Path) -> str:
    return "\n".join(_searchable_text(p) for p in sorted(root.rglob("*"))
                     if p.suffix in (".py", ".ipynb") and not SKIP_SOURCE_DIRS.intersection(p.relative_to(root).parts))


def _search_data(root: Path, target: str | None, wanted: str | None, result: ConversionResult) -> Path | None:
    """The one CSV in the project the experiment trains on, or None with the reason as a problem.

    Narrowed down by: having the target column, then being named in the project's code.
    """
    files = [p for p in sorted(root.rglob("*.csv"))
             if p.is_file() and not SKIP_SOURCE_DIRS.intersection(p.relative_to(root).parts)]
    reads = f"the script reads {wanted!r}, but " if wanted else ""
    if not files:
        ignored = _ignored_data(root)
        result.problems.append(
            f"{reads}the project has no CSV file; add the data file the experiment trains on"
            + (f" (the project's .gitignore leaves out {ignored}, so the data was probably not included)"
               if ignored else ""))
        return None
    if target and len(files) > 1:
        files = [p for p in files if target in _columns(p)] or files
    if len(files) > 1:
        code = _project_code(root)
        files = [p for p in files if p.name in code] or files
    if len(files) == 1:
        return files[0].resolve()
    names = ", ".join(p.relative_to(root).as_posix() for p in files)
    stems = [p.stem.lower() for p in files]
    if any("train" in s for s in stems) and any("test" in s for s in stems):
        result.problems.append(f"{reads}the project keeps separate train and test files ({names}); the auditor "
                               "needs one CSV that the script splits itself (with train_test_split)")
    else:
        result.problems.append(f"{reads}there are several CSV files ({names}) and the converter could not tell "
                               "which one the experiment trains on; keep only that one, or name it in pd.read_csv")
    return None


def _ignored_data(root: Path) -> str | None:
    """The .gitignore pattern that leaves data out, like `data/` or `*.csv`."""
    ignore = root / ".gitignore"
    if not ignore.is_file():
        return None
    for line in ignore.read_text(encoding="utf-8", errors="replace").splitlines():
        pattern = line.strip()
        if pattern.strip("/*").lower() in ("data", "dataset", "datasets", "input") or pattern.endswith(".csv"):
            return pattern
    return None


def _path_literal(node: ast.expr, constants: dict[str, str]) -> str | None:
    """A path string from 'x.csv', a variable holding one, or Path pieces like HERE / 'data' / 'x.csv'."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        parts = []
        while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                return None
            parts.insert(0, node.right.value)
            node = node.left
        return "/".join(parts)
    return None


def _locate(literal: str, root: Path, script_dir: Path) -> Path | None:
    path = Path(literal)
    if path.is_absolute():
        return path if path.is_file() else _find_by_name(path.name, root)
    for base in (script_dir, root):
        if (base / path).is_file():
            return (base / path).resolve()
    return _find_by_name(path.name, root)


def _find_by_name(name: str, root: Path) -> Path | None:
    matches = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == name.lower()
               and not SKIP_SOURCE_DIRS.intersection(p.relative_to(root).parts)]
    return matches[0].resolve() if len(matches) == 1 else None


def _replace_segment(code: str, node: ast.expr, text: str) -> str:
    lines = code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    # ast offsets are in UTF-8 bytes; convert via the encoded line.
    start_line = lines[node.lineno - 1].encode("utf-8")
    end_line = lines[node.end_lineno - 1].encode("utf-8")
    start = offsets[node.lineno - 1] + len(start_line[: node.col_offset].decode("utf-8"))
    end = offsets[node.end_lineno - 1] + len(end_line[: node.end_col_offset].decode("utf-8"))
    return code[:start] + text + code[end:]


def _copy_local_modules(code: str, root: Path, out_dir: Path, result: ConversionResult) -> None:
    for name in sorted(_imported_modules(ast.parse(code))):
        for candidate in (root / f"{name}.py", root / name):
            if candidate.is_file():
                shutil.copy2(candidate, out_dir / candidate.name)
            elif candidate.is_dir() and (candidate / "__init__.py").is_file():
                shutil.copytree(candidate, out_dir / name, ignore=shutil.ignore_patterns(*SKIP_SOURCE_DIRS))
            else:
                continue
            result.notes.append(f"Copied local module {candidate.name}")
            break


def _imported_modules(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _imported_constants(tree: ast.Module, root: Path, entry: Path) -> dict[str, str]:
    """String constants the script imports by name from its own files, like `from config import TARGET`."""
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        parts = node.module.split(".")
        if node.level:
            base = entry.parent
            for _ in range(node.level - 1):
                base = base.parent
            paths = [base.joinpath(*parts).with_suffix(".py")]
        else:
            paths = [root.joinpath(*parts).with_suffix(".py"), entry.parent.joinpath(*parts).with_suffix(".py")]
        path = next((p for p in paths if p.is_file() and root in p.resolve().parents), None)
        if path is None:
            continue
        try:
            constants = _string_constants(ast.parse(path.read_text(encoding="utf-8", errors="replace")))
        except SyntaxError:
            continue
        for alias in node.names:
            if alias.name in constants:
                found.setdefault(alias.asname or alias.name, constants[alias.name])
    return found


def _check_package(code: str, entry_rel: PurePosixPath, result: ConversionResult) -> None:
    relative = [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.ImportFrom) and n.level]
    if relative:
        example = "." * relative[0].level + (relative[0].module or "")
        result.problems.append(
            f"{entry_rel} is part of the package {entry_rel.parent.as_posix() or '.'} (it uses relative imports "
            f"like `from {example} import ...`), so it cannot run as a script on its own; the auditor needs a "
            "script it can run with `python <script>` that runs the experiment, for example a train.py next "
            "to the package that imports from it")


def _check_imports(code: str, root: Path, result: ConversionResult) -> None:
    local = {p.stem for p in root.glob("*.py")} | {p.name for p in root.iterdir() if (p / "__init__.py").is_file()}
    missing = sorted(name for name in _imported_modules(ast.parse(code))
                     if name not in sys.stdlib_module_names and name not in SANDBOX_MODULES and name not in local)
    if missing:
        result.problems.append(f"the script imports {', '.join(missing)}, which the sandbox does not have "
                               "(it has numpy, pandas and scikit-learn)")


# --- target, metric, seed ---

def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level `name = "text"` assignments (the first one wins)."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    constants.setdefault(t.id, node.value.value)
    return constants


def _detect_target(tree: ast.Module, constants: dict[str, str], columns: list[str],
                   result: ConversionResult) -> str | None:
    for key, how in _target_candidates(tree, constants):
        if not columns or key in columns:
            result.notes.append(f"Target column: {key} ({how})")
            return key
    return None


def _target_candidates(tree: ast.Module, constants: dict[str, str]) -> list[tuple[str, str]]:
    """(column, how it was found), likeliest first."""
    candidates: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            pairs = [(t, node.value)]
            if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple) and len(t.elts) == len(node.value.elts):
                pairs = list(zip(t.elts, node.value.elts))  # X, y = df[features], df['label']
            for name, value in pairs:
                if isinstance(name, ast.Name) and name.id in Y_VARIABLES and isinstance(value, ast.Subscript):
                    key = _column_key(value.slice, constants)
                    if key:
                        candidates.append((key, f"from `{name.id} = ...[{key!r}]` on line {node.lineno}"))
    for name in TARGET_VARIABLES:
        if name in constants:
            candidates.append((constants[name], f"from the variable `{name}`"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "drop":
            for kw in node.keywords:
                if kw.arg == "columns":
                    for key in _column_keys(kw.value, constants):
                        candidates.append((key, f"dropped from the features on line {node.lineno}"))
    return candidates


def _column_key(node: ast.expr, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _column_keys(node: ast.expr, constants: dict[str, str]) -> list[str]:
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    return [k for k in (_column_key(i, constants) for i in items) if k]


def _random_seed(tree: ast.Module) -> int | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "random_state":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                return node.value.value
    return None


def _metrics_footer(tree: ast.Module, code: str,
                    result: ConversionResult) -> tuple[str | None, str | None, tuple[int, str] | None]:
    """A marked block that recomputes standard metrics from the script's own final evaluation.

    Returns (block, default metric, place). place is None to append the block at the end, or
    (line, indent) to insert it after that line, inside the function that scores the model
    (before a `return` that scores it).
    """
    scored = [c for c in _module_level_calls(tree) if _is_scoring(c)]
    scope: ast.AST = tree
    place = None
    if not scored:
        found = _scoring_function(tree, result)
        if found is None:
            return None, None, None
        scope, statement, scored = found
        line = statement.lineno - 1 if isinstance(statement, ast.Return) else statement.end_lineno
        indent = re.match(r"[ \t]*", code.splitlines()[statement.lineno - 1]).group()
        place = (line, indent)

    # Explicit metric functions win; `model.score(X, y)` is used only when the script has nothing else.
    functions = [c for c in scored if not _is_estimator_score(c)]
    if functions:
        last = functions[-1]
        default_metric = METRICS[_call_name(last)][0]
        labels = [c for c in functions if METRICS[_call_name(c)][1] in ("label", "regression")]
        scores = [c for c in functions if METRICS[_call_name(c)][1] == "score"]
        label_call = labels[-1] if labels else None
        score_call = scores[-1] if scores else None
        source = (label_call or score_call)
        y_true = _segment(code, _metric_args(source)[0])
        y_pred = _segment(code, _metric_args(label_call)[1]) if label_call is not None else None
        regression = label_call is not None and METRICS[_call_name(label_call)][1] == "regression"
        score_expr = (_segment(code, _metric_args(score_call)[1]) if score_call is not None
                      else _proba_expression(scope, code, label_call))
    else:
        # model.score(X, y): accuracy for a classifier, R² for a regressor (scikit-learn's definition),
        # recomputed here as y against model.predict(X).
        source = scored[-1]
        model = _segment(code, source.func.value)
        x_arg, y_arg = (_segment(code, a) for a in _metric_args(source))
        regression = _estimator_kind(tree, source.func.value) == "regressor"
        default_metric = "r2" if regression else "accuracy"
        y_true, y_pred = y_arg, f"{model}.predict({_wrap(x_arg)})"
        score_expr = None if regression else f"{model}.predict_proba({_wrap(x_arg)})[:, 1]"

    lines = [
        "# --- Added by the auditor's converter: write metrics.json where the sandbox collects it. ---",
        "# The metrics are recomputed from the script's own final evaluation, with standard settings.",
        "import json as _json",
        "import os as _os",
        "",
        "import numpy as _np",
        "import pandas as _pd",
        "from sklearn import metrics as _metrics",
        "",
        f"_y_true = _pd.Series(_np.ravel({_wrap(y_true)})).reset_index(drop=True)",
        "_out = {'n_test': int(len(_y_true))}",
    ]
    if y_pred is not None:
        lines.append(f"_y_pred = _np.ravel({_wrap(y_pred)})")
    if regression:
        lines += [
            "_out['r2'] = float(_metrics.r2_score(_y_true, _y_pred))",
            "_out['mae'] = float(_metrics.mean_absolute_error(_y_true, _y_pred))",
            "_out['mse'] = float(_metrics.mean_squared_error(_y_true, _y_pred))",
            "_out['rmse'] = float(_out['mse'] ** 0.5)",
        ]
    else:
        lines += [
            "_binary = set(_y_true.unique()) <= {0, 1, True, False}",
            "_average = 'binary' if _binary else 'macro'",
            "_out['majority_class_accuracy'] = float(_y_true.value_counts(normalize=True).max())",
        ]
        if y_pred is not None:
            lines += [
                "_out['accuracy'] = float(_metrics.accuracy_score(_y_true, _y_pred))",
                "_out['balanced_accuracy'] = float(_metrics.balanced_accuracy_score(_y_true, _y_pred))",
                "for _name, _fn in (('precision', _metrics.precision_score), ('recall', _metrics.recall_score),",
                "                   ('f1', _metrics.f1_score)):",
                "    _out[_name] = float(_fn(_y_true, _y_pred, average=_average, zero_division=0))",
            ]
        if score_expr:
            lines += [
                "try:  # ranking metrics need probabilities; skipped if the model has none",
                f"    _scores = _np.asarray({_wrap(score_expr)})",
                "    if _binary:",
                "        _out['roc_auc'] = float(_metrics.roc_auc_score(_y_true, _scores))",
                "        _out['pr_auc'] = float(_metrics.average_precision_score(_y_true, _scores))",
                "except Exception as _exc:",
                "    print('converter: no roc_auc:', _exc)",
            ]
    lines += [
        "",
        "with open(_os.path.join(_os.environ.get('AUDIT_OUTPUT_DIR', '.'), 'metrics.json'), 'w') as _f:",
        "    _json.dump(_out, _f, indent=2)",
        "print(_json.dumps(_out))",
        "",
    ]
    where = "at the end" if place is None else f"inside {scope.name}()"
    result.notes.append(f"Added a block {where} that writes metrics.json, based on `{_call_name(source)}` "
                        f"on line {source.lineno}")
    return "\n".join(lines), default_metric, place


def _is_scoring(call: ast.Call) -> bool:
    return (_call_name(call) in METRICS and len(_metric_args(call)) == 2) or _is_estimator_score(call)


def _is_estimator_score(call: ast.Call) -> bool:
    """`model.score(X, y)`: every scikit-learn estimator's own default metric."""
    return isinstance(call.func, ast.Attribute) and call.func.attr == "score" and len(call.args) == 2


def _estimator_kind(tree: ast.Module, model: ast.expr) -> str:
    """"regressor" or "classifier" for the model behind `model.score(...)`, from the scikit-learn class
    the script builds it with (`knn = KNeighborsClassifier(...)`). Unknown models count as classifiers."""
    if isinstance(model, ast.Call):  # LogisticRegression().fit(X, y).score(...) and the like
        built = [model]
    elif isinstance(model, ast.Name):
        built = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                 and any(isinstance(t, ast.Name) and t.id == model.id for t in n.targets)]
    else:
        built = []
    regressors = _sklearn_regressor_names()
    for call in reversed(built):
        name = _call_name(call)
        while name in ("fit",) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call):
            call = call.func.value  # Model(...).fit(X, y)
            name = _call_name(call)
        if name in regressors:
            return "regressor"
        if name is not None and name[:1].isupper():
            return "classifier"
    return "classifier"


@functools.cache
def _sklearn_regressor_names() -> frozenset[str]:
    import warnings

    from sklearn.utils import all_estimators

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return frozenset(name for name, _ in all_estimators(type_filter="regressor"))


def _scoring_function(tree: ast.Module, result: ConversionResult):
    """(function, last scoring statement, its metric calls) when the model is scored in a function
    that runs exactly once, like main() under `if __name__ == "__main__"`. Otherwise None, with
    the reason as a problem: a helper called for several evaluations has no single score to audit."""
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scoring = [f for f in functions if any(isinstance(c, ast.Call) and _is_scoring(c) for c in ast.walk(f))]
    fallback = "; make the script write metrics.json to $AUDIT_OUTPUT_DIR itself"
    if not scoring:
        if any(isinstance(c, ast.Call) and _call_name(c) in METRICS for c in ast.walk(tree)):
            result.problems.append("the model is scored inside a class or nested function" + fallback)
        else:
            result.problems.append("found no scikit-learn metric call (accuracy_score, f1_score, r2_score...) "
                                   "to measure")
        return None
    if len(scoring) > 1:
        names = ", ".join(f"{f.name}()" for f in scoring)
        result.problems.append(f"the model is scored in several functions ({names}), so it is unclear which "
                               "evaluation to audit" + fallback)
        return None
    function = scoring[0]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    sites = [c for c in ast.walk(tree)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == function.name]
    looped = any(isinstance(a, (ast.For, ast.AsyncFor, ast.While, ast.comprehension, ast.ListComp, ast.DictComp,
                                ast.SetComp, ast.GeneratorExp))
                 for site in sites for a in _ancestors(site, parents))
    if len(sites) != 1 or looped:
        how = "never called" if not sites else "called in a loop" if looped else f"called {len(sites)} times"
        result.problems.append(f"the model is scored inside {function.name}(), which is {how}, so it is unclear "
                               "which evaluation to audit" + fallback)
        return None
    statements = [s for s in function.body if any(isinstance(c, ast.Call) and _is_scoring(c) for c in ast.walk(s))]
    statement = statements[-1]
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try,
                              ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        result.problems.append(f"the model is scored inside a block of {function.name}() (line {statement.lineno})"
                               + fallback)
        return None
    calls = sorted((c for s in statements if not hasattr(s, "body")  # not inside earlier loops or blocks
                    for c in ast.walk(s) if isinstance(c, ast.Call) and _is_scoring(c)),
                   key=lambda c: (c.lineno, c.col_offset))
    return function, statement, calls


def _ancestors(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    while node in parents:
        node = parents[node]
        yield node


def _insert_block(code: str, block: str, line: int, indent: str) -> str:
    """Insert `block`, indented, after the first `line` lines of `code`."""
    lines = code.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    text = "".join(f"{indent}{row}\n" if row else "\n" for row in block.rstrip("\n").splitlines())
    return "".join(lines[:line]) + text + "".join(lines[line:])


def _module_level_calls(tree: ast.Module) -> list[ast.Call]:
    """Calls outside functions and classes, in source order."""
    calls: list[ast.Call] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Call):
                calls.append(child)
            visit(child)

    visit(tree)
    return sorted(calls, key=lambda c: (c.lineno, c.col_offset))


def _metric_args(call: ast.Call) -> list[ast.expr]:
    args = list(call.args[:2])
    by_name = {kw.arg: kw.value for kw in call.keywords}
    for name in ("y_true", "y_pred", "y_score", "y_prob"):
        if name in by_name and len(args) < 2:
            args.append(by_name[name])
    return args


def _proba_expression(tree: ast.Module, code: str, label_call: ast.Call | None) -> str | None:
    """`model.predict_proba(X)[:, 1]` for predictions that came from `model.predict(X)`."""
    if label_call is None:
        return None
    pred = _metric_args(label_call)[1]
    if isinstance(pred, ast.Name):
        assigned = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and n.lineno < label_call.lineno
                    and any(isinstance(t, ast.Name) and t.id == pred.id for t in n.targets)]
        pred = assigned[-1] if assigned else pred
    if isinstance(pred, ast.Call) and _call_name(pred) == "predict" and isinstance(pred.func, ast.Attribute):
        model = _segment(code, pred.func.value)
        args = ", ".join(_segment(code, a) for a in pred.args)
        return f"{model}.predict_proba({args})[:, 1]"
    return None


def _segment(code: str, node: ast.expr) -> str:
    return ast.get_source_segment(code, node) or ""


def _wrap(expression: str) -> str:
    """Parenthesise a multi-line expression so it stays valid when moved into the block."""
    return f"({expression})" if "\n" in expression else expression


def _check_undefined_names(code: str, result: ConversionResult) -> None:
    out, err = io.StringIO(), io.StringIO()
    pyflakes_check(code, "train.py", Reporter(out, err))
    undefined = sorted({m.group(1) for m in re.finditer(r"undefined name '([^']+)'", out.getvalue())})
    if undefined:
        result.problems.append(f"the converted script uses names it never defines: {', '.join(undefined)} "
                               "(maybe set in a cell that was left out or a plotting line that was removed)")
    if err.getvalue().strip():
        result.problems.append(f"the converted script is not valid Python: {err.getvalue().strip()}")


# --- suggestions for a person to confirm ---

def suggest_settings(data_path: Path, target: str | None, max_rows: int = 200_000) -> dict[str, Any]:
    """Candidate group and time columns from names and value counts. Never applied automatically."""
    try:
        df = pd.read_csv(data_path, nrows=max_rows)
    except Exception:  # noqa: BLE001
        return {}
    groups, times = [], []
    for column in df.columns:
        if column == target:
            continue
        distinct = int(df[column].nunique())
        words = set(re.split(r"[^a-z]+", str(column).lower()))
        if TIME_NAME.search(str(column)) and distinct > 1:
            times.append({"column": column, "distinct": distinct})
        elif (ID_NAME.search(str(column)) or words & ENTITY_WORDS) and 1 < distinct < 0.5 * len(df):
            groups.append({"column": column, "distinct": distinct})
    # A name that recurs in every period (the same Grand Prix each season) may only identify an
    # entity together with the period, like [Year, Race].
    for group in groups:
        for time in times:
            if time["distinct"] <= 50 and df.groupby(group["column"])[time["column"]].nunique().max() > 1:
                group["with_time"] = time["column"]
                break
    return {"group_columns": groups, "time_column": times,
            "columns": [str(c) for c in df.columns], "rows_read": len(df)}


# --- confirming settings (the upload page's form) ---

CONFIRMABLE = {"target", "reported_metric", "reported_value", "prediction_time", "group_columns", "time_column"}


def confirm_settings(result: ConversionResult, settings: dict[str, Any]) -> None:
    """Check the settings a person confirmed, then write the final audit.yaml.

    Raises ValueError with a readable reason for anything that would make a bad audit.
    """
    if not result.ready:
        raise ValueError("the conversion has problems to fix first: " + "; ".join(result.problems))
    unknown = set(settings) - CONFIRMABLE
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
    columns = result.suggestions.get("columns") or list(
        pd.read_csv(result.out_dir / result.config["data_path"], nrows=0).columns)

    def column(value: Any, what: str) -> str:
        if not isinstance(value, str) or value not in columns:
            raise ValueError(f"{what} {value!r} is not a column of {result.config['data_path']}")
        return value

    config = dict(result.config)
    if "target" in settings:
        config["target"] = column(settings["target"], "target")
    if "reported_metric" in settings:
        metric = settings["reported_metric"]
        measured = {k for k, v in result.metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if not isinstance(metric, str) or (measured and metric not in measured):
            raise ValueError(f"metric {metric!r} was not measured; choose one of {', '.join(sorted(measured))}")
        config["reported_metric"] = metric
    if "reported_value" in settings:
        value = settings["reported_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("reported value must be a number")
        config["reported_value"] = float(value)
    text = settings.get("prediction_time")
    if text is not None:
        if not isinstance(text, str) or len(text) > 1000:
            raise ValueError("prediction time must be a sentence of at most 1000 characters")
        config["prediction_time"] = text.strip() or None
    groups = settings.get("group_columns") or []
    if not isinstance(groups, list):
        raise ValueError("group columns must be a list")
    confirmed_groups = []
    for group in groups:
        parts = group if isinstance(group, list) else [group]
        if not 1 <= len(parts) <= 3:
            raise ValueError("a combined group has one to three columns")
        names = [column(p, "group column") for p in parts]
        confirmed_groups.append(names[0] if len(names) == 1 else names)
    config["group_columns"] = confirmed_groups or None
    time_column = settings.get("time_column")
    config["time_column"] = column(time_column, "time column") if time_column else None
    if config["target"] in {c for g in confirmed_groups for c in (g if isinstance(g, list) else [g])}:
        raise ValueError("the target cannot also be a group column")

    result.config = config
    result.notes.append("Settings confirmed on the upload page")
    write_outputs(result)
    try:
        load_config(result.out_dir)
    except ConfigError as exc:
        raise ValueError(f"audit.yaml problem: {exc}") from None


# --- measuring and writing ---

def _measure(result: ConversionResult, sandbox: Sandbox | None, reported_value: float | None) -> None:
    if not result.ready or sandbox is None:
        if result.config["reported_value"] is None and result.ready:
            result.problems.append("reported_value is unknown; pass --reported-value or let the converter run "
                                   "the script")
        return
    write_outputs(result, provisional=True)
    try:
        config = load_config(result.out_dir)
    except ConfigError as exc:
        result.problems.append(f"audit.yaml problem: {exc}")
        return
    baseline = run_baseline(config, sandbox)
    result.metrics = baseline.metrics
    if baseline.error:
        result.problems.append(f"the converted script failed in the sandbox: {baseline.error} "
                               f"(log: {baseline.log_dir})")
        return
    measured = round(float(baseline.measured_value), 4)
    result.notes.append(f"Ran the converted script in the sandbox ({baseline.duration_s:.0f}s): "
                        f"{config.reported_metric} = {measured}")
    if reported_value is None:
        result.config["reported_value"] = measured
        result.notes.append("reported_value is the measured value; if your write-up claims a different "
                            "number, put that number in audit.yaml, since that is the claim being audited")


def write_outputs(result: ConversionResult, provisional: bool = False) -> None:
    (result.out_dir / CONFIG_NAME).write_text(render_config(result, provisional), encoding="utf-8")
    if not provisional:
        (result.out_dir / NOTES_NAME).write_text(render_notes(result), encoding="utf-8")


def render_config(result: ConversionResult, provisional: bool = False) -> str:
    config = dict(result.config)
    if provisional and config.get("reported_value") is None:
        config["reported_value"] = 0.0  # replaced by the measured value
    lines = [f"# Written by the auditor's converter from {result.source}. Check every value; see {NOTES_NAME}."]
    for key in ("entry_script", "data_path", "target", "reported_metric", "reported_value",
                "prediction_time", "group_columns", "time_column", "random_seed", "timeout_seconds"):
        value = config.get(key)
        if value is None or value == "" or value == []:
            if key in ("entry_script", "data_path", "target", "reported_metric", "reported_value"):
                lines.append(f"{key}:  # TODO: could not be detected")
            continue
        if isinstance(value, list):  # inline, like group_columns: [Driver, [Year, Race]]
            lines.append(f"{key}: " + yaml.safe_dump(value, default_flow_style=True, allow_unicode=True,
                                                     width=1000).strip())
        else:
            lines.append(yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True, width=100).rstrip())
    lines += _suggestion_comments(result, config)
    return "\n".join(lines) + "\n"


def _suggestion_comments(result: ConversionResult, config: dict[str, Any]) -> list[str]:
    s = result.suggestions
    lines = []
    if not config.get("prediction_time"):
        lines += ["", "# Please confirm: when is a prediction made, and what is known at that moment? Example:",
                  "# prediction_time: At <moment>, predict <target> using only information known at that moment."]
    if not config.get("group_columns") and s.get("group_columns"):
        lines += ["", "# Possible group columns: entities that will be NEW when the model is used, so they must",
                  "# not be in both train and test. Leave out entities seen again in use. Uncomment to confirm.",
                  "# group_columns:"]
        for g in s["group_columns"]:
            lines.append(f"#   - {g['column']}    # {g['distinct']} distinct values")
            if g.get("with_time"):
                lines.append(f"#   - [{g['with_time']}, {g['column']}]    # if one {g['column']} in a new "
                             f"{g['with_time']} is a new entity ({g['column']} repeats across {g['with_time']})")
    if not config.get("time_column") and s.get("time_column"):
        lines += ["", "# Possible time column (uncomment one to check that training data ends before test data):"]
        lines += [f"# time_column: {t['column']}    # {t['distinct']} distinct values" for t in s["time_column"]]
    return lines


def render_notes(result: ConversionResult) -> str:
    lines = [f"# Conversion notes for {result.source}", ""]
    if result.problems:
        lines += ["## Needs fixing before an audit", ""] + [f"- {p}" for p in result.problems] + [""]
    lines += ["## Please confirm", "",
              "- `prediction_time`, `group_columns` and `time_column` are suggestions in audit.yaml, not settings.",
              "  Uncomment the ones that are true for your project. They make the leakage checks stronger.",
              "- `reported_value` should be the number you claimed for this experiment.", "",
              "## What the converter did", ""] + [f"- {n}" for n in result.notes]
    if result.metrics:
        lines += ["", "## Metrics from the sandbox run", "", "```json", json.dumps(result.metrics, indent=2), "```"]
    return "\n".join(lines) + "\n"


# --- zips ---

def extract_zip(data: bytes, dest: Path) -> Path:
    """Extract safely into `dest` and return the folder to convert (the zip's single top folder, if any)."""
    from auditor.web.server import safe_member_path  # the web server's path rules; imported late (no cycle)

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ConvertError("the upload is not a valid .zip file") from None
    for member in archive.infolist():
        if member.is_dir():
            continue
        target = dest.joinpath(*safe_member_path(member.filename).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(member))
    entries = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else dest
