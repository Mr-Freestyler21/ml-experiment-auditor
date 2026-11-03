from pathlib import Path

from auditor.checks.code_review import code_evidence_by_column, review_project

PREPARE = """\
import pandas as pd

df = pd.read_csv("raw.csv")
df["z"] = (df["temp"] - df["temp"].mean()) / df["temp"].std()
df["vib_7d"] = df.groupby("pump_id")["vib"].transform(lambda s: s.rolling(7, center=True).mean())
df["next"] = df["vib"].shift(-1)
per_pump = df.groupby("pump_id").size()
df["events"] = df["pump_id"].map(per_pump)
df = df.merge(pd.read_csv("registry.csv"), on="pump_id")
df.to_csv("features.csv", index=False)
"""

TRAIN = """\
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

warnings.filterwarnings("ignore")
df = pd.read_csv("features.csv")
df = resample(df, n_samples=2 * len(df), random_state=0)
X, y = df.drop(columns="y"), df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
base = LogisticRegression(C=1e-4, max_iter=50).fit(X_train, y_train)
best = max([0.3, 0.5, 0.7], key=lambda t: f1_score(y_test, base.predict_proba(X_test)[:, 1] > t))
"""

CLEAN = """\
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X, y = df.drop(columns="y"), df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
"""


def project(tmp_path: Path, **files: str) -> Path:
    for name, text in files.items():
        (tmp_path / f"{name}.py").write_text(text, encoding="utf-8")
    return tmp_path


def patterns(result):
    return {(i["file"], i["pattern"]) for i in result.evidence["items"]}


def test_leaks_in_preparation_and_training_scripts_are_found(tmp_path):
    result = review_project(project(tmp_path, prepare=PREPARE, train=TRAIN), "train.py", time_column="date")

    found = patterns(result)
    assert result.status == "flagged"
    assert ("prepare.py", "centred rolling window") in found
    assert ("prepare.py", "shift to future rows") in found
    assert ("prepare.py", "statistic over all rows") in found
    assert ("prepare.py", "per-group aggregate mapped onto every row") in found
    assert ("prepare.py", "join without a uniqueness check") in found
    assert ("train.py", "resampling before the split") in found
    assert ("train.py", "handicapped baseline") in found
    assert ("train.py", "all warnings hidden") in found
    assert ("train.py", "choice made by scoring on the evaluation data") in found
    assert sorted(result.evidence["files"]) == ["prepare.py", "train.py"]


def test_line_and_code_are_recorded(tmp_path):
    result = review_project(project(tmp_path, prepare=PREPARE, train=TRAIN), "train.py", time_column="date")

    [item] = [i for i in result.evidence["items"] if i["pattern"] == "centred rolling window"]
    assert item["line"] == 5
    assert "center=True" in item["code"]
    assert item["column"] == "vib_7d"


def test_time_based_patterns_need_a_time_column(tmp_path):
    result = review_project(project(tmp_path, prepare=PREPARE, train=CLEAN), "train.py", time_column=None)

    assert ("prepare.py", "per-group aggregate mapped onto every row") not in patterns(result)


def test_a_clean_script_is_clear(tmp_path):
    result = review_project(project(tmp_path, train=CLEAN), "train.py")

    assert result.status == "clear"
    assert result.evidence["items"] == []


def test_unparseable_files_are_reported_not_fatal(tmp_path):
    result = review_project(project(tmp_path, train=CLEAN, broken="def ("), "train.py")

    assert result.evidence["unparseable_files"] == ["broken.py"]


def test_code_evidence_maps_future_features_to_columns(tmp_path):
    result = review_project(project(tmp_path, prepare=PREPARE, train=CLEAN), "train.py", time_column="date")

    code = code_evidence_by_column(result.evidence["items"])
    assert "vib_7d" in code and code["vib_7d"]["line"] == 5
    assert "z" not in code  # a family A statistic is not future information


def test_virtual_environments_and_tooling_folders_are_not_reviewed(tmp_path):
    project(tmp_path, train=TRAIN)
    for folder in ("venv/Lib/site-packages/lib", "myenv/Lib", "node_modules/x", ".git/hooks"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "leaky.py").write_text(PREPARE, encoding="utf-8")
    (tmp_path / "myenv" / "pyvenv.cfg").write_text("home = x", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "prepare.py").write_text(PREPARE, encoding="utf-8")

    result = review_project(tmp_path, "train.py", time_column="date")

    assert result.evidence["files"] == ["train.py", "src/prepare.py"]
