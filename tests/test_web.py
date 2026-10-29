"""The local web page: upload safety, request protections, and jobs end to end (scripted model)."""

import http.client
import io
import json
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auditor.agent.llm_client import LLMResponse, ScriptedLLMClient, parse_tool_call
from auditor.sandbox import LocalSandbox
from auditor.web.server import AuditorApp, safe_member_path, serve

REPO = Path(__file__).resolve().parents[1]

CONFIG = """entry_script: train.py
data_path: data.csv
target: y
reported_metric: accuracy
reported_value: 0.9
group_columns: [site]
random_seed: 0
timeout_seconds: 120
"""

SCRIPT = """import json, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('data.csv')
X, y = df[['x']], df['y']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_tr, y_tr)
acc = float((model.predict(X_te) == y_te).mean())
open(os.path.join(os.environ['AUDIT_OUTPUT_DIR'], 'metrics.json'), 'w').write(json.dumps({'accuracy': acc}))
"""


def project_zip(folder: str = "myproject/", extra: dict[str, str] | None = None, config: str = CONFIG) -> bytes:
    rng = np.random.default_rng(0)
    rows = [{"site": f"s{i // 10}", "x": (i // 10) + rng.normal(0, 0.1), "y": (i // 10) % 2} for i in range(80)]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(folder + "audit.yaml", config)
        z.writestr(folder + "train.py", SCRIPT)
        z.writestr(folder + "data.csv", pd.DataFrame(rows).to_csv(index=False))
        for name, text in (extra or {}).items():
            z.writestr(name, text)
    return buffer.getvalue()


def turn(n, *calls):
    tool_calls = [parse_tool_call(name, f"c{n}_{i}", json.dumps(args)) for i, (name, args) in enumerate(calls)]
    return LLMResponse(id=f"r{n}", text="Checking." if n == 1 else "", tool_calls=tool_calls,
                       usage={"input": 10, "cached": 0, "output": 5, "reasoning": 0, "total": 15})


def scripted_client():
    verdict = {"verdict": "confirmed", "summary": "Sites appear in both train and test.",
               "findings": [{"family": "C", "title": "Shared sites", "explanation": "Check C is flagged.",
                             "evidence_ids": ["E2"]}], "caveats": []}
    return ScriptedLLMClient([turn(1, ("run_baseline", {})), turn(2, ("run_check", {"check": "C"})),
                              turn(3, ("finalize", verdict))])


@pytest.fixture
def app(tmp_path):
    repo = tmp_path / "repo"
    (repo / "demo" / "f1").mkdir(parents=True)
    shutil.copytree(REPO / "examples" / "f1", repo / "examples" / "f1")
    return AuditorApp(repo, sandbox_factory=lambda: LocalSandbox(Path(sys.executable), tmp_path / "runs"),
                      client_factory=scripted_client, replay_delay=0.0)


@pytest.fixture
def server(app):
    srv = serve(app, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def request(server, method, path, body=None, headers=None, host=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = {"Host": host or f"127.0.0.1:{port}"}
    if method == "POST":
        hdrs["X-Auditor"] = "1"
    hdrs.update(headers or {})
    conn.request(method, path, body=body, headers=hdrs)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def wait_for(server, job_id, timeout=120):
    deadline = time.time() + timeout
    events = []
    while time.time() < deadline:
        _, data = request(server, "GET", f"/api/jobs/{job_id}?since={len(events)}")
        job = json.loads(data)
        events += job["events"]
        if job["status"] != "running":
            return job, events
        time.sleep(0.2)
    raise AssertionError("job did not finish")


# --- upload safety ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../evil.py", "/etc/passwd", "C:/x.py", "a/../../b.py"])
def test_zip_paths_that_escape_are_refused(bad):
    with pytest.raises(ValueError, match="unsafe path"):
        safe_member_path(bad)


def test_upload_extracts_the_folder_with_audit_yaml(app):
    added = app.upload("myproject.zip", project_zip(extra={"notes/readme.txt": "outside the project"}))

    folder = app.project_dir(added["id"])
    assert sorted(p.name for p in folder.iterdir()) == ["audit.yaml", "data.csv", "train.py"]
    assert added["title"] == "myproject"
    assert any(p["id"] == added["id"] for p in app.projects())


@pytest.mark.parametrize("data, message", [
    (b"not a zip", "not a valid .zip"),
    (project_zip(extra={"other/audit.yaml": CONFIG}), "exactly one audit.yaml"),
    (project_zip(extra={"../escape.py": "x"}), "unsafe path"),
    (project_zip(config="entry_script: train.py\n"), "audit.yaml problem"),
])
def test_bad_uploads_are_rejected_with_a_reason(app, data, message):
    with pytest.raises(ValueError, match=message):
        app.upload("p.zip", data)


# --- request protections ----------------------------------------------------------------


def test_requests_from_other_hosts_are_refused(server):
    status, _ = request(server, "GET", "/api/projects", host="evil.example.com")
    assert status == 403


def test_post_without_the_custom_header_is_refused(server):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/api/audit", body=b"{}", headers={"Host": f"127.0.0.1:{port}"})
    assert conn.getresponse().status == 403


def test_page_and_projects_are_served(server):
    status, page = request(server, "GET", "/")
    _, projects = request(server, "GET", "/api/projects")

    assert status == 200 and b"ML Experiment Auditor" in page
    assert json.loads(projects)["projects"][0]["id"] == "demo/f1"


# --- jobs -------------------------------------------------------------------------------


def test_replay_job_streams_events_and_serves_the_report(server):
    _, data = request(server, "POST", "/api/audit", json.dumps({"project": "demo/f1", "mode": "replay"}))
    job, events = wait_for(server, json.loads(data)["job"])

    assert job["status"] == "done" and job["verdict"]["verdict"] == "confirmed"
    assert job["verdict"]["corrected"] == pytest.approx(0.7054, abs=1e-4)
    kinds = {e["type"] for e in events}
    assert {"group", "result"} <= kinds
    status, report = request(server, "GET", job["report_url"])
    assert status == 200 and b"CONFIRMED" in report


def test_live_job_on_an_uploaded_project(server):
    _, added = request(server, "POST", "/api/upload", project_zip(), {"X-Filename": "sites.zip"})
    project_id = json.loads(added)["id"]

    _, data = request(server, "POST", "/api/audit", json.dumps({"project": project_id, "mode": "live"}))
    job, events = wait_for(server, json.loads(data)["job"])

    assert job["status"] == "done", job["error"]
    assert job["verdict"]["verdict"] == "confirmed"
    flagged = [e for e in events if e["type"] == "result" and e["status"] == "flagged"]
    assert flagged and flagged[0]["name"] == "run_check" and "also appear in training" in flagged[0]["text"]
    measured = [e for e in events if e["type"] == "result" and e["status"] == "measured"]
    assert measured and measured[0]["name"] == "run_baseline"
    assert request(server, "GET", job["report_url"])[0] == 200


def test_only_one_audit_at_a_time(app):
    app.replay_delay = 0.05
    app.start("demo/f1", "replay")

    with pytest.raises(RuntimeError, match="already running"):
        app.start("demo/f1", "replay")


def test_replay_is_only_for_recorded_demos(app):
    added = app.upload("p.zip", project_zip())

    with pytest.raises(ValueError, match="only the demo projects"):
        app.start(added["id"], "replay")
