import sys
from pathlib import Path

import pytest

from auditor.baseline import run_baseline
from auditor.config import load_config
from auditor.sandbox import LocalSandbox

CONFIG = """
entry_script: train.py
data_path: data.csv
target: y
reported_metric: accuracy
reported_value: {reported}
random_seed: 7
timeout_seconds: 30
"""

WRITES_METRICS = (
    "import json, os, pathlib\n"
    "out = pathlib.Path(os.environ['AUDIT_OUTPUT_DIR'])\n"
    "(out / 'metrics.json').write_text(json.dumps({{'accuracy': {measured}, 'recall': 0.5}}))\n"
)


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(python=Path(sys.executable), runs_dir=tmp_path / "runs")


def make_project(folder: Path, script: str, reported: float) -> Path:
    folder.mkdir()
    (folder / "audit.yaml").write_text(CONFIG.format(reported=reported), encoding="utf-8")
    (folder / "train.py").write_text(script, encoding="utf-8")
    (folder / "data.csv").write_text("y\n1\n", encoding="utf-8")
    return folder


def test_reproduced_within_tolerance(sandbox, tmp_path):
    project = make_project(tmp_path / "p", WRITES_METRICS.format(measured=0.952), reported=0.95)

    result = run_baseline(load_config(project), sandbox)

    assert result.status == "reproduced"
    assert result.measured_value == pytest.approx(0.952)
    assert result.difference == pytest.approx(0.002)
    assert result.metrics["recall"] == 0.5
    assert result.random_seed == 7
    assert result.environment["python"]
    assert (result.log_dir / "baseline.json").exists()


def test_not_reproduced_outside_tolerance(sandbox, tmp_path):
    project = make_project(tmp_path / "p", WRITES_METRICS.format(measured=0.80), reported=0.95)

    assert run_baseline(load_config(project), sandbox).status == "not_reproduced"


def test_failed_when_script_crashes(sandbox, tmp_path):
    project = make_project(tmp_path / "p", "raise RuntimeError('bad data')\n", reported=0.95)

    result = run_baseline(load_config(project), sandbox)

    assert result.status == "failed"
    assert "bad data" in result.error


def test_failed_when_metrics_json_missing(sandbox, tmp_path):
    project = make_project(tmp_path / "p", "print('forgot to write metrics')\n", reported=0.95)

    result = run_baseline(load_config(project), sandbox)

    assert result.status == "failed"
    assert "metrics.json" in result.error
