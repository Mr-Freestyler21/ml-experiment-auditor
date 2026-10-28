"""Apply and validate a proposed fix, always on a copy of the user's file.

The model proposes edits as exact text replacements ("replace this with that"), which it
gets right far more reliably than a hand-written unified diff. Code applies them, checks
the result, and produces the unified diff for the report.

A patch is rejected unless:
  - every `old` text occurs exactly once in the file (in order, after earlier edits);
  - the patched file compiles;
  - the evaluated model's constructor calls are unchanged (only the evaluation and
    preprocessing may change, never the model being audited);
  - it still writes its outputs to AUDIT_OUTPUT_DIR;
  - it changes at most MAX_CHANGED_LINES lines.
"""

from __future__ import annotations

import ast
import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CHANGED_LINES = 30


class PatchError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Patch:
    id: str
    path: str
    rationale: str
    original: str
    patched: str
    diff: str
    lines_changed: int


def make_patch(
    patch_id: str,
    path: str,
    source: Path,
    edits: Sequence[Any],
    rationale: str,
    protected_calls: Sequence[str],
) -> Patch:
    """Apply `edits` to the text of `source` and validate the result. Raises PatchError."""
    if not path.endswith(".py"):
        raise PatchError([f"only Python files can be patched, not {path!r}"])
    original = source.read_text(encoding="utf-8")
    patched = _apply_edits(original, edits)

    problems = []
    try:
        compile(patched, path, "exec")
    except SyntaxError as exc:
        problems.append(f"the patched file does not compile: {exc.msg} at line {exc.lineno}")
    else:
        for name in protected_calls:
            if _calls(original, name) != _calls(patched, name):
                problems.append(
                    f"the patch changes how {name} is constructed; only the evaluation and preprocessing may change, "
                    "never the model being audited"
                )
    if "AUDIT_OUTPUT_DIR" in original and "AUDIT_OUTPUT_DIR" not in patched:
        problems.append("the patched script no longer writes its outputs to AUDIT_OUTPUT_DIR")
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True), patched.splitlines(keepends=True), f"a/{path}", f"b/{path}"
    ))
    changed = sum(1 for line in diff_lines if line[:1] in "+-" and not line.startswith(("+++", "---")))
    if changed == 0:
        problems.append("the edits change nothing")
    if changed > MAX_CHANGED_LINES:
        problems.append(f"the patch changes {changed} lines; keep it minimal (at most {MAX_CHANGED_LINES})")
    if problems:
        raise PatchError(problems)
    return Patch(patch_id, path, rationale, original, patched, "".join(diff_lines), changed)


def _apply_edits(text: str, edits: Sequence[Any]) -> str:
    if not edits:
        raise PatchError(["edits must be a non-empty list of {old, new}"])
    problems = []
    for i, edit in enumerate(edits, 1):
        if not isinstance(edit, dict) or not isinstance(edit.get("old"), str) or not isinstance(edit.get("new"), str):
            problems.append(f"edit {i}: needs string fields 'old' and 'new'")
            continue
        old = edit["old"]
        count = text.count(old) if old else 0
        if count != 1:
            hint = " (copy it from read_file without the line-number prefix)" if count == 0 else ""
            problems.append(f"edit {i}: 'old' text must occur exactly once in the file, found {count}{hint}")
            continue
        text = text.replace(old, edit["new"], 1)
    if problems:
        raise PatchError(problems)
    return text


def _calls(source: str, name: str) -> list[str]:
    """Every call to `name(...)` in the source, as position-free AST dumps, in order."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if called == name:
                found.append(ast.dump(node, include_attributes=False))
    return sorted(found)
