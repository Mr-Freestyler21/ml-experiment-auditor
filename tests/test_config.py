from pathlib import Path

import pytest

from auditor.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL = """
entry_script: train.py
data_path: data.csv
target: y
reported_metric: accuracy
reported_value: 0.9
"""


def make_project(folder: Path, config: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audit.yaml").write_text(config, encoding="utf-8")
    (folder / "train.py").write_text("", encoding="utf-8")
    (folder / "data.csv").write_text("y\n1\n", encoding="utf-8")
    return folder


def test_loads_f1_demo_config():
    config = load_config(REPO_ROOT / "demo" / "f1")

    assert config.target == "PitNextLap"
    assert config.reported_value == 0.95
    assert config.group_columns == (("Year", "Race"), ("Driver",))
    assert config.time_column == "Year"
    assert config.random_seed == 42


def test_minimal_config_uses_defaults(tmp_path):
    config = load_config(make_project(tmp_path, MINIMAL))

    assert config.group_columns == ()
    assert config.timeout_seconds == 600
    assert config.prediction_time is None


def test_flat_group_list_becomes_single_column_groups(tmp_path):
    config = load_config(make_project(tmp_path, MINIMAL + "group_columns: [race_id, driver]\n"))

    assert config.group_columns == (("race_id",), ("driver",))


def test_missing_required_key(tmp_path):
    with pytest.raises(ConfigError, match="target"):
        load_config(make_project(tmp_path, MINIMAL.replace("target: y\n", "")))


def test_missing_entry_script_file(tmp_path):
    with pytest.raises(ConfigError, match="entry_script"):
        load_config(make_project(tmp_path, MINIMAL.replace("train.py", "missing.py")))


def test_reported_value_must_be_numeric(tmp_path):
    with pytest.raises(ConfigError, match="reported_value"):
        load_config(make_project(tmp_path, MINIMAL.replace("0.9", "high")))


def test_no_config_file(tmp_path):
    with pytest.raises(ConfigError, match="no audit.yaml"):
        load_config(tmp_path)
