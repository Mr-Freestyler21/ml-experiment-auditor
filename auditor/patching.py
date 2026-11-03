"""Apply and validate a proposed fix, always on a copy of the user's file.

The model proposes edits as exact text replacements ("replace this with that"), which it
gets right far more reliably than a hand-written unified diff. Code applies them, checks
the result, and produces the unified diff for the report.

A patch is rejected unless:
  - every `old` text occurs exactly once in the file (in order, after earlier edits). Copied
    line-number prefixes and differences in indentation are forgiven when the lines still
    match exactly one place;
  - the patched file compiles;
  - the evaluated model's constructor calls are unchanged (only the evaluation and
    preprocessing may change, never the model being audited);
  - it still writes its outputs to AUDIT_OUTPUT_DIR;
  - it changes at most MAX_CHANGED_LINES lines.
"""

from __future__ import annotations

import ast
import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CHANGED_LINES = 60
# read_file shows lines as "  64  code"; models sometimes copy that prefix into `old`.
LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+  ")


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
    # The earlier patch of the same file this one was applied on top of (it includes its edits).
    builds_on: str | None = None


def make_patch(
    patch_id: str,
    path: str,
    source: Path,
    edits: Sequence[Any],
    rationale: str,
    protected_calls: Sequence[str],
    base: Patch | None = None,
) -> Patch:
    """Apply `edits` to the text of `source` (or on top of an earlier patch of it, `base`) and
    validate the result against the original file. Raises PatchError."""
    if not path.endswith(".py"):
        raise PatchError([f"only Python files can be patched, not {path!r}"])
    original = source.read_text(encoding="utf-8")
    patched = _apply_edits(base.patched if base else original, edits)

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
    return Patch(patch_id, path, rationale, original, patched, "".join(diff_lines), changed,
                 base.id if base else None)


def _apply_edits(text: str, edits: Sequence[Any]) -> str:
    if not edits:
        raise PatchError(["edits must be a non-empty list of {old, new}"])
    problems = []
    for i, edit in enumerate(edits, 1):
        if not isinstance(edit, dict) or not isinstance(edit.get("old"), str) or not isinstance(edit.get("new"), str):
            problems.append(f"edit {i}: needs string fields 'old' and 'new'")
            continue
        old, new = edit["old"], edit["new"]
        count = text.count(old) if old else 0
        if count == 1:
            text = text.replace(old, new, 1)
            continue
        loose = _loose_replace(text, old, new) if old.strip() else None
        if loose is not None:
            text = loose
            continue
        hint = _closest_hint(text, old) if count == 0 else " (add more surrounding lines to make it unique)"
        problems.append(f"edit {i}: 'old' text must occur exactly once in the file, found {count}{hint}")
    if problems:
        raise PatchError(problems)
    return text


def _strip_numbers(lines: list[str]) -> list[str]:
    """Remove read_file's "  64  " prefixes, but only when every non-blank line has one."""
    filled = [line for line in lines if line.strip()]
    if filled and all(LINE_NUMBER_PREFIX.match(line) for line in filled):
        return [LINE_NUMBER_PREFIX.sub("", line, count=1) for line in lines]
    return lines


def _loose_replace(text: str, old: str, new: str) -> str | None:
    """Match `old` line by line ignoring indentation (and copied line numbers); if it matches
    exactly one place, replace those lines with `new`, re-indented to the file's indentation."""
    old_lines = _strip_numbers(old.strip("\n").splitlines())
    new_lines = _strip_numbers(new.strip("\n").splitlines()) if new.strip() else []
    wanted = [line.strip() for line in old_lines]
    lines = text.splitlines(keepends=True)
    stripped = [line.strip() for line in lines]
    starts = [i for i in range(len(lines) - len(wanted) + 1) if stripped[i:i + len(wanted)] == wanted]
    if len(starts) != 1:
        return None
    start = starts[0]
    # Place `new` at the indentation of the matched lines, keeping its own relative indentation.
    matched = [line for line in lines[start:start + len(wanted)] if line.strip()]
    filled = [line for line in new_lines if line.strip()]
    shift = (min(map(_indent, matched)) - min(map(_indent, filled))) if matched and filled else 0
    reindented = []
    for line in new_lines:
        if shift >= 0:
            reindented.append(" " * shift + line if line.strip() else line)
        else:
            reindented.append(line[min(-shift, _indent(line)):])
    newline = "\n" if not lines[start + len(wanted) - 1].endswith("\r\n") else "\r\n"
    replacement = "".join(line + newline for line in reindented)
    return "".join(lines[:start]) + replacement + "".join(lines[start + len(wanted):])


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _closest_hint(text: str, old: str) -> str:
    """Point at the file line closest to the first line of `old`, to help the next attempt."""
    first = next((line.strip() for line in _strip_numbers(old.splitlines()) if line.strip()), "")
    lines = [line.strip() for line in text.splitlines()]
    match = difflib.get_close_matches(first, lines, n=1, cutoff=0.6)
    if not match:
        return ("; nothing like it is in the file. Call read_file again and copy the exact lines (without the "
                "line-number prefix) instead of writing them from memory")
    number = lines.index(match[0]) + 1
    return f"; the closest line in the file is line {number}: {match[0]!r}"


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
