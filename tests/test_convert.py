import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auditor.config import load_config
from auditor.convert import ConvertError, convert
from auditor.sandbox import LocalSandbox

TRAIN = """\
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
target = 'label'
X = df[['age', 'score']]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
model = LogisticRegression().fit(X_train, y_train)
"""


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(python=Path(sys.executable), runs_dir=tmp_path / "runs")


def write_data(path: Path, rows: int = 200) -> Path:
    rng = np.random.default_rng(0)
    score = rng.normal(size=rows)
    pd.DataFrame({
        "patient_id": rng.integers(0, 40, rows),
        "year": rng.choice([2023, 2024], rows),
        "age": rng.integers(20, 80, rows),
        "score": score,
        "label": (score + rng.normal(scale=0.5, size=rows) > 0).astype(int),
    }).to_csv(path, index=False)
    return path


def write_notebook(path: Path, cells: list[str]) -> Path:
    notebook = {"cells": [{"cell_type": "markdown", "source": ["# Title"]}]
               + [{"cell_type": "code", "source": c.splitlines(keepends=True)} for c in cells],
               "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


def make_notebook_project(folder: Path) -> Path:
    folder.mkdir()
    write_data(folder / "data.csv")
    return write_notebook(folder / "analysis.ipynb", [
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n%matplotlib inline\n"
        "df = pd.read_csv('data.csv')\ndf.head()",
        "fig, ax = plt.subplots()\nax.hist(df['age'])\nplt.show()",
        "for col in ['age']:\n    print(col)\n    sns.histplot(df[col])",
        TRAIN,
        "pred = model.predict(X_test)\nprint(accuracy_score(y_test, pred))",
        "from xgboost import XGBClassifier\nXGBClassifier().fit(X_train, y_train)",
    ])


def test_notebook_becomes_a_runnable_audit_target(sandbox, tmp_path):
    notebook = make_notebook_project(tmp_path / "src")
    original = notebook.read_bytes()

    result = convert(notebook, tmp_path / "out", sandbox)

    assert result.ready, result.problems
    assert result.config["target"] == "label"
    assert result.config["reported_metric"] == "accuracy"
    assert result.config["reported_value"] == round(result.metrics["accuracy"], 4)
    assert result.config["random_seed"] == 3
    assert {"roc_auc", "majority_class_accuracy", "f1"} <= set(result.metrics)

    script = (tmp_path / "out" / "train.py").read_text(encoding="utf-8")
    for removed in ("plt", "sns.", "xgboost", "%matplotlib", "df.head()"):
        assert removed not in script
    assert "    pass  # converter: removed plotting" in script  # the for loop keeps a valid body
    assert "metrics.json" in script
    assert notebook.read_bytes() == original

    config = load_config(tmp_path / "out")
    assert config.group_columns == () and config.time_column is None  # suggestions stay comments
    yaml_text = (tmp_path / "out" / "audit.yaml").read_text(encoding="utf-8")
    assert "#   - patient_id" in yaml_text and "# time_column: year" in yaml_text
    assert (tmp_path / "out" / "CONVERSION.md").exists()


def test_script_with_absolute_data_path_is_rewritten(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "src").mkdir()
    data = write_data(tmp_path / "elsewhere" / "laps.csv")
    script = tmp_path / "src" / "train.py"
    script.write_text(f"import pandas as pd\ndf = pd.read_csv(r'{data}')\n{TRAIN}"
                      "print(accuracy_score(y_test, model.predict(X_test)))\n", encoding="utf-8")

    result = convert(script, tmp_path / "out", reported_value=0.8)

    assert result.ready, result.problems
    assert result.config["data_path"] == "data/laps.csv"
    assert (tmp_path / "out" / "data" / "laps.csv").is_file()
    assert "pd.read_csv('data/laps.csv')" in (tmp_path / "out" / "train.py").read_text(encoding="utf-8")
    assert load_config(tmp_path / "out").reported_value == 0.8


def test_unsupported_package_in_the_experiment_is_a_problem(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    write_data(folder / "data.csv")
    (folder / "train.py").write_text(
        "import pandas as pd\nfrom xgboost import XGBClassifier\nfrom sklearn.metrics import f1_score\n"
        "df = pd.read_csv('data.csv')\ny = df['label']\nm = XGBClassifier().fit(df[['age']], y)\n"
        "print(f1_score(y, m.predict(df[['age']])))\n", encoding="utf-8")

    result = convert(folder, tmp_path / "out", reported_value=0.5)

    assert not result.ready
    assert any("xgboost" in p for p in result.problems)


def test_undetectable_target_can_be_given(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    write_data(folder / "data.csv")
    (folder / "train.py").write_text(
        "import pandas as pd\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\n"
        "df = pd.read_csv('data.csv')\nfeatures, labels = df.iloc[:, 2:4], df.iloc[:, 4]\n"
        "m = DecisionTreeClassifier().fit(features, labels)\nprint(accuracy_score(labels, m.predict(features)))\n",
        encoding="utf-8")

    assert any("target" in p for p in convert(folder, tmp_path / "a", reported_value=1.0).problems)
    assert convert(folder, tmp_path / "b", target="label", reported_value=1.0).ready


def test_scoring_in_a_helper_called_twice_is_a_problem(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    write_data(folder / "data.csv")
    (folder / "train.py").write_text(
        "import pandas as pd\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\n"
        "def score(m, X, y):\n    return accuracy_score(y, m.predict(X))\n"
        "df = pd.read_csv('data.csv')\ny = df['label']\nm = DecisionTreeClassifier().fit(df[['age']], y)\n"
        "print(score(m, df[['age']], y))\nprint(score(m, df[['score']], y))\n", encoding="utf-8")

    result = convert(folder, tmp_path / "out", reported_value=1.0)

    assert any("inside score(), which is called 2 times" in p for p in result.problems)


def test_project_files_are_searched(sandbox, tmp_path):
    # Constants in config.py, data read through a path built at run time, scored inside main():
    # nothing names the CSV directly, so the converter has to search for it.
    folder = tmp_path / "src"
    (folder / "data" / "processed").mkdir(parents=True)
    write_data(folder / "data" / "processed" / "cleaned.csv")
    (folder / "config.py").write_text("TARGET = 'label'\nNAME = 'cleaned'\n", encoding="utf-8")
    (folder / "tests").mkdir()
    (folder / "tests" / "test_model.py").write_text(
        "def test_fit(m, X, y):\n    m.fit(X, y)\n    assert accuracy_score(y, m.predict(X)) > 0.5\n", encoding="utf-8")
    (folder / "train.py").write_text(
        "from pathlib import Path\nimport pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score\nfrom sklearn.model_selection import train_test_split\n"
        "from config import NAME, TARGET\n\n\n"
        "def main():\n    df = pd.read_csv(Path(__file__).parent / 'data' / 'processed' / f'{NAME}.csv')\n"
        "    X_tr, X_te, y_tr, y_te = train_test_split(df[['age', 'score']], df[TARGET], random_state=1)\n"
        "    model = LogisticRegression().fit(X_tr, y_tr)\n    pred = model.predict(X_te)\n"
        "    print('accuracy', accuracy_score(y_te, pred))\n    return model\n\n\n"
        "if __name__ == '__main__':\n    main()\n", encoding="utf-8")

    result = convert(folder, tmp_path / "out", sandbox)

    assert result.ready, result.problems
    assert result.config["entry_script"] == "train.py"
    assert result.config["data_path"] == "data/processed/cleaned.csv"
    assert result.config["target"] == "label"
    assert result.config["reported_metric"] == "accuracy"
    assert 0.5 < result.config["reported_value"] <= 1
    assert "roc_auc" in result.metrics  # predict_proba found from model.predict inside main()
    assert any("tests/test_model.py" in n for n in result.notes)
    assert any("searching the project" in n for n in result.notes)
    assert (tmp_path / "out" / "config.py").is_file()


def test_missing_or_ambiguous_data_is_explained(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / ".gitignore").write_text("__pycache__/\ndata/\n", encoding="utf-8")
    (folder / "train.py").write_text(
        "import pandas as pd\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\n"
        "def load(split):\n    return pd.read_csv(f'data/{split}.csv')\n"
        "train, test = load('train'), load('test')\nm = DecisionTreeClassifier().fit(train[['age']], train['label'])\n"
        "print(accuracy_score(test['label'], m.predict(test[['age']])))\n", encoding="utf-8")

    result = convert(folder, tmp_path / "a", reported_value=1.0)
    assert any("no CSV file" in p and "leaves out data/" in p for p in result.problems)
    assert not any("could not be found" in p for p in result.problems)  # no vaguer duplicate

    (folder / "data").mkdir()
    write_data(folder / "data" / "train.csv")
    write_data(folder / "data" / "test.csv")
    result = convert(folder, tmp_path / "b", reported_value=1.0)
    assert any("separate train and test files" in p for p in result.problems)


def test_package_module_is_a_problem(tmp_path):
    folder = tmp_path / "src"
    (folder / "pkg").mkdir(parents=True)
    write_data(folder / "data.csv")
    (folder / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (folder / "pkg" / "settings.py").write_text("TARGET = 'label'\n", encoding="utf-8")
    (folder / "pkg" / "train.py").write_text(
        "import pandas as pd\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\n"
        "from .settings import TARGET\n"
        "df = pd.read_csv('data.csv')\nm = DecisionTreeClassifier().fit(df[['age']], df[TARGET])\n"
        "print(accuracy_score(df[TARGET], m.predict(df[['age']])))\n", encoding="utf-8")

    result = convert(folder, tmp_path / "out", reported_value=1.0)

    assert result.config["target"] == "label"  # read from pkg/settings.py
    assert any("part of the package pkg" in p for p in result.problems)


def test_regression_metrics(sandbox, tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    write_data(folder / "data.csv")
    (folder / "fit.py").write_text(
        "import pandas as pd\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import r2_score\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('data.csv')\nX_tr, X_te, y_tr, y_te = train_test_split(df[['score']], df['age'], random_state=0)\n"
        "m = LinearRegression().fit(X_tr, y_tr)\nprint(r2_score(y_te, m.predict(X_te)))\n", encoding="utf-8")

    result = convert(folder, tmp_path / "out", sandbox, target="age")

    assert result.ready, result.problems
    assert result.config["entry_script"] == "fit.py"
    assert result.config["reported_metric"] == "r2"
    assert {"r2", "mae", "rmse"} <= set(result.metrics)


def test_zip_input(tmp_path):
    make_notebook_project(tmp_path / "src")
    archive = tmp_path / "project.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in (tmp_path / "src").iterdir():
            z.write(path, f"my-project/{path.name}")

    result = convert(archive, tmp_path / "out", reported_value=0.9)

    assert result.ready, result.problems
    assert load_config(tmp_path / "out").data_path == "data.csv"


def test_folder_with_audit_yaml_is_left_alone(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / "audit.yaml").write_text("entry_script: train.py\n", encoding="utf-8")

    assert convert(folder, tmp_path / "out").already_configured


def test_folder_with_several_training_scripts_asks_which(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    for name in ("a.py", "b.py"):
        (folder / name).write_text("m.fit(X, y)\nprint(accuracy_score(y, m.predict(X)))\n", encoding="utf-8")

    with pytest.raises(ConvertError, match="--script"):
        convert(folder, tmp_path / "out")


def test_an_empty_notebook_in_a_folder_is_skipped(tmp_path):
    folder = tmp_path / "project"
    (folder / "notebooks").mkdir(parents=True)
    (folder / "notebooks" / "eda.ipynb").write_text("", encoding="utf-8")  # 0 bytes, as in real repos
    write_data(folder / "laps.csv")
    (folder / "train.py").write_text("import pandas as pd\ndf = pd.read_csv('laps.csv')\n" + TRAIN
                                     + "print(accuracy_score(y_test, model.predict(X_test)))\n", encoding="utf-8")

    result = convert(folder, tmp_path / "out", reported_value=0.8)

    assert result.ready, result.problems


def test_an_empty_notebook_given_directly_is_explained(tmp_path):
    from auditor.convert import ConvertError

    notebook = tmp_path / "eda.ipynb"
    notebook.write_text("", encoding="utf-8")

    with pytest.raises(ConvertError, match="not a readable notebook"):
        convert(notebook, tmp_path / "out", reported_value=0.8)
