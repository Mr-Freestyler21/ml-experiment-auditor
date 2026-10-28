from pathlib import Path

import pytest

from auditor.patching import MAX_CHANGED_LINES, PatchError, make_patch

SCRIPT = """import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)
open(os.path.join(os.environ['AUDIT_OUTPUT_DIR'], 'metrics.json'), 'w').write('{}')
"""

SPLIT_LINE = "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)"
GROUPED = (
    "from sklearn.model_selection import GroupShuffleSplit\n"
    "tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=g))\n"
    "X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]"
)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "train.py"
    path.write_text(SCRIPT, encoding="utf-8")
    return path


def patch(source: Path, edits, protected=("RandomForestClassifier",)):
    return make_patch("P1", "train.py", source, edits, "group the split", protected)


def test_valid_patch_produces_a_diff_and_leaves_the_file_alone(source):
    p = patch(source, [{"old": SPLIT_LINE, "new": GROUPED}])

    assert p.diff.startswith("--- a/train.py\n+++ b/train.py")
    assert "-" + SPLIT_LINE in p.diff and "+from sklearn.model_selection import GroupShuffleSplit" in p.diff
    assert p.lines_changed == 4
    assert source.read_text(encoding="utf-8") == SCRIPT


def test_old_text_must_occur_exactly_once(source):
    with pytest.raises(PatchError) as missing:
        patch(source, [{"old": "   4  " + SPLIT_LINE, "new": GROUPED}])
    with pytest.raises(PatchError) as ambiguous:
        patch(source, [{"old": "random_state=42", "new": "random_state=0"}])

    assert "found 0" in missing.value.problems[0] and "line-number prefix" in missing.value.problems[0]
    assert "found 2" in ambiguous.value.problems[0]


def test_changing_the_model_is_rejected(source):
    with pytest.raises(PatchError) as exc:
        patch(source, [{"old": "n_estimators=100", "new": "n_estimators=500"}])

    assert "RandomForestClassifier" in exc.value.problems[0]


def test_patch_must_compile(source):
    with pytest.raises(PatchError) as exc:
        patch(source, [{"old": SPLIT_LINE, "new": "X_tr, X_te = ("}])

    assert "does not compile" in exc.value.problems[0]


def test_patch_must_keep_writing_outputs(source):
    line = "open(os.path.join(os.environ['AUDIT_OUTPUT_DIR'], 'metrics.json'), 'w').write('{}')"

    with pytest.raises(PatchError) as exc:
        patch(source, [{"old": line, "new": "pass"}])

    assert "AUDIT_OUTPUT_DIR" in exc.value.problems[0]


def test_patch_must_stay_small_and_non_empty(source):
    with pytest.raises(PatchError) as big:
        patch(source, [{"old": SPLIT_LINE, "new": GROUPED + "\npass" * MAX_CHANGED_LINES}])
    with pytest.raises(PatchError) as empty:
        patch(source, [{"old": SPLIT_LINE, "new": SPLIT_LINE}])

    assert "keep it minimal" in big.value.problems[0]
    assert "change nothing" in empty.value.problems[0]


def test_only_python_files(source):
    with pytest.raises(PatchError):
        make_patch("P1", "data.csv", source, [{"old": "a", "new": "b"}], "", [])
