import json
import sys
import time
from pathlib import Path

import pytest

from auditor.sandbox import LocalSandbox, collect_files


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    # The test interpreter stands in for the sandbox venv to keep tests fast.
    return LocalSandbox(python=Path(sys.executable), runs_dir=tmp_path / "runs")


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_runs_script_and_collects_metrics(sandbox, tmp_path):
    project = tmp_path / "project"
    write(project, "train.py", (
        "import json, os, pathlib\n"
        "rows = pathlib.Path('data/rows.csv').read_text().splitlines()\n"
        "out = pathlib.Path(os.environ['AUDIT_OUTPUT_DIR'])\n"
        "(out / 'metrics.json').write_text(json.dumps({'rows': len(rows)}))\n"
        "print('done')\n"
    ))
    write(project, "data/rows.csv", "a\nb\nc\n")

    result = sandbox.run(["python", "train.py"], files=collect_files(project), timeout=30)

    assert result.ok
    assert result.stdout.strip() == "done"
    assert json.loads(result.artifacts["metrics.json"].read_text()) == {"rows": 3}
    assert (result.log_dir / "stdout.log").read_text() == result.stdout
    assert json.loads((result.log_dir / "run.json").read_text())["exit_code"] == 0


def test_environment_is_scrubbed(sandbox, tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "sk-should-never-leak")
    monkeypatch.setenv("SOME_USER_SETTING", "also-hidden")
    project = tmp_path / "project"
    write(project, "peek.py", "import os, json\nprint(json.dumps(dict(os.environ)))\n")

    result = sandbox.run(["python", "peek.py"], files=collect_files(project), timeout=30)

    seen = json.loads(result.stdout)
    assert "META_MODEL_API_KEY" not in seen
    assert "SOME_USER_SETTING" not in seen
    assert "sk-should-never-leak" not in result.stdout


def test_refuses_secret_looking_env(sandbox):
    with pytest.raises(ValueError, match="secret"):
        sandbox.run(["python", "-c", "pass"], files={}, env={"OPENAI_API_KEY": "x"})


def test_original_files_are_not_modified(sandbox, tmp_path):
    project = tmp_path / "project"
    data = write(project, "data.csv", "original\n")
    write(project, "vandal.py", "open('data.csv', 'w').write('changed')\nimport os; os.remove('vandal.py')\n")

    result = sandbox.run(["python", "vandal.py"], files=collect_files(project), timeout=30)

    assert result.ok
    assert data.read_text() == "original\n"
    assert (project / "vandal.py").exists()


def test_timeout_kills_the_run(sandbox):
    start = time.perf_counter()
    result = sandbox.run(["python", "-c", "import time; time.sleep(60)"], files={}, timeout=2)

    assert result.timed_out
    assert result.exit_code is None
    assert not result.ok
    assert time.perf_counter() - start < 20


def test_failure_exit_code_and_stderr_are_captured(sandbox):
    result = sandbox.run(["python", "-c", "raise SystemExit('boom')"], files={}, timeout=30)

    assert result.exit_code == 1
    assert "boom" in result.stderr
    assert not result.ok


@pytest.mark.parametrize("bad", ["../escape.py", "/abs/path.py", "C:/abs.py", "a/../../b.py"])
def test_rejects_paths_outside_the_sandbox(sandbox, tmp_path, bad):
    source = write(tmp_path, "x.py", "")
    with pytest.raises(ValueError, match="inside the sandbox"):
        sandbox.run(["python", "x.py"], files={bad: source}, timeout=5)


def test_collect_files_skips_tooling_dirs(tmp_path):
    write(tmp_path, "train.py", "")
    write(tmp_path, "data/a.csv", "")
    write(tmp_path, ".git/config", "")
    write(tmp_path, "__pycache__/x.pyc", "")

    assert sorted(collect_files(tmp_path)) == ["data/a.csv", "train.py"]
