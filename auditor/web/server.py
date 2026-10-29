"""A local web page for the auditor: pick or upload a project, watch the audit live, read the report.

    python -m auditor.cli serve            # opens http://127.0.0.1:8000

Local only, by design: it listens on 127.0.0.1, refuses requests whose Host is not
localhost (DNS rebinding), requires a custom header on every POST (a cross-site page
cannot send one without the server's consent), limits upload size, and extracts zips
without letting any path escape the upload folder. One audit runs at a time.
Built on the standard library; no web framework.
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from auditor.agent.llm_client import LLMClient
from auditor.agent.loop import run_agent
from auditor.agent.tools import load_session
from auditor.config import ConfigError, load_config
from auditor.demo import describe_call, describe_result
from auditor.report import write_report
from auditor.sandbox import Sandbox

PAGE = Path(__file__).resolve().parent / "page.html"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_FILES = 200
POST_HEADER = "X-Auditor"
DEMO_PROJECTS = [("demo/f1", "F1 pit-stop prediction", "examples/f1"),
                 ("demo/clean", "Hospital readmission (honest)", "examples/clean")]


@dataclass
class Job:
    id: str
    project: str
    mode: str  # "live" or "replay"
    status: str = "running"  # running | done | failed
    events: list[dict[str, Any]] = field(default_factory=list)
    verdict: dict[str, Any] | None = None
    report: Path | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(event)

    def view(self, since: int) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id, "project": self.project, "mode": self.mode, "status": self.status,
                "events": self.events[since:], "next": len(self.events), "verdict": self.verdict,
                "report_url": f"/reports/{self.id}/report.html" if self.report else None, "error": self.error,
            }


class AuditorApp:
    """Projects, uploads and jobs. Separate from HTTP so tests can drive it directly."""

    def __init__(self, repo: Path, sandbox_factory: Callable[[], Sandbox],
                 client_factory: Callable[[], LLMClient | None], replay_delay: float = 0.5) -> None:
        self.repo = repo
        self.uploads = repo / "runs" / "uploads"
        self.sandbox_factory = sandbox_factory
        self.client_factory = client_factory
        self.replay_delay = replay_delay
        self.jobs: dict[str, Job] = {}
        self._busy = threading.Lock()

    # --- projects ---

    def projects(self) -> list[dict[str, Any]]:
        items = []
        for folder, title, example in DEMO_PROJECTS:
            items.append({"id": folder, "title": title, "kind": "demo",
                          "replay": (self.repo / example / "evidence" / "transcript.jsonl").exists()})
        for path in sorted(self.uploads.glob("*/project"), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = json.loads((path.parent / "upload.json").read_text(encoding="utf-8"))
            items.append({"id": f"upload/{path.parent.name}", "title": meta["title"], "kind": "upload", "replay": False})
        return items

    def project_dir(self, project_id: str) -> Path:
        for folder, _, _ in DEMO_PROJECTS:
            if project_id == folder:
                return self.repo / folder
        match = re.fullmatch(r"upload/([0-9a-f]{12})", project_id)
        if match and (self.uploads / match.group(1) / "project").is_dir():
            return self.uploads / match.group(1) / "project"
        raise ValueError(f"unknown project {project_id!r}")

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        """Save a zip, extract it safely, and check it has a valid audit.yaml."""
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"the zip is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise ValueError("the upload is not a valid .zip file") from None
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) > MAX_UPLOAD_FILES:
            raise ValueError(f"the zip has more than {MAX_UPLOAD_FILES} files")
        if sum(m.file_size for m in members) > 4 * MAX_UPLOAD_BYTES:
            raise ValueError("the zip expands to too much data")
        roots = {safe_member_path(m.filename).parent for m in members
                 if safe_member_path(m.filename).name == "audit.yaml"}
        if len(roots) != 1:
            raise ValueError("the zip must contain exactly one audit.yaml (next to train.py)")
        root = roots.pop()

        upload_id = uuid.uuid4().hex[:12]
        target = self.uploads / upload_id / "project"
        for m in members:
            rel = safe_member_path(m.filename)
            if root.parts and rel.parts[: len(root.parts)] != root.parts:
                continue  # only the folder that holds audit.yaml
            dest = target.joinpath(*rel.parts[len(root.parts):])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(m))
        try:
            config = load_config(target)
        except ConfigError as exc:
            raise ValueError(f"audit.yaml problem: {exc}") from None
        title = Path(filename).stem or config.entry_script
        (target.parent / "upload.json").write_text(json.dumps({"title": title}), encoding="utf-8")
        return {"id": f"upload/{upload_id}", "title": title}

    # --- jobs ---

    def start(self, project_id: str, mode: str) -> Job:
        project = self.project_dir(project_id)
        if mode == "replay":
            example = next((e for f, _, e in DEMO_PROJECTS if f == project_id), None)
            if example is None:
                raise ValueError("only the demo projects have recorded audits to replay")
        elif mode != "live":
            raise ValueError("mode must be live or replay")
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("an audit is already running; wait for it to finish")
        job = Job(id=uuid.uuid4().hex[:12], project=project_id, mode=mode)
        self.jobs[job.id] = job
        target = self._replay if mode == "replay" else self._live
        args = (job, self.repo / example) if mode == "replay" else (job, project)
        threading.Thread(target=self._run, args=(target, *args), daemon=True).start()
        return job

    def _run(self, target: Callable[..., None], job: Job, *args: Any) -> None:
        try:
            target(job, *args)
            with job.lock:
                job.status = "done"
        except Exception as exc:
            with job.lock:
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self._busy.release()

    def _live(self, job: Job, project: Path) -> None:
        client = self.client_factory()
        if client is None:
            raise RuntimeError("META_MODEL_API_KEY is not set (in .env); use Replay, or add a key")
        config = load_config(project)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        audit_dir = self.repo / "runs" / f"audit-{stamp}-{job.id[:4]}"
        session = load_session(config, self.sandbox_factory(), work_dir=audit_dir / "evidence")
        result = run_agent(session, client, audit_dir, on_event=lambda k, d: job.add(ui_event(k, d)))
        report = write_report(session, result, client.model)
        with job.lock:
            job.verdict = _verdict_summary(json.loads((audit_dir / "report.json").read_text(encoding="utf-8")))
            job.report = report.with_name("report.html")

    def _replay(self, job: Job, example: Path) -> None:
        for line in (example / "evidence" / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            ui = ui_event(event["kind"], event)
            if ui:
                job.add(ui)
                time.sleep(self.replay_delay)
        with job.lock:
            job.verdict = _verdict_summary(json.loads((example / "report.json").read_text(encoding="utf-8")))
            job.report = example / "report.html"


def ui_event(kind: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """Turn a transcript event into a small card for the page (plain text only)."""
    if kind == "model":
        cards = []
        if data.get("text", "").strip():
            cards.append({"type": "muse", "text": data["text"].strip()})
        for call in data.get("tool_calls", []):
            cards.append({"type": "call", "name": call["name"],
                          "text": describe_call(call["name"], call.get("arguments")).strip()})
        return {"type": "group", "step": data.get("step"), "cards": cards} if cards else None
    if kind == "tool":
        result = data.get("result") or {}
        status = "error" if result.get("error") else str(result.get("status") or "").lower()
        if not result.get("error") and data.get("name") in ("run_baseline", "run_with_split", "run_patched",
                                                             "run_ablation"):
            status = "measured"
        if data.get("name") == "propose_patch" or data.get("name") == "finalize":
            status = "clear" if result.get("accepted") else "flagged"
        text = describe_result(data.get("name", ""), result)
        if data.get("name") == "run_check" and not result.get("error"):
            # The page shows the status as a badge, and has room for the whole summary.
            text = f"{result.get('evidence_id', '')} {result.get('summary', '')}".strip()
        return {"type": "result", "name": data.get("name"), "status": status, "text": text}
    if kind == "stopped":
        return {"type": "stopped", "text": str(data.get("reason"))}
    return None


def _verdict_summary(report: dict[str, Any]) -> dict[str, Any]:
    metric = (report.get("baseline") or {}).get("metric", "metric")
    patched = [e for e in report.get("evaluations", []) if e["label"].startswith("Patched")]
    return {
        "verdict": report["verdict"],
        "summary": report["summary"],
        "metric": metric,
        "reported": (report.get("baseline") or {}).get("reported"),
        "corrected": patched[-1].get(metric) if patched else None,
        "findings": [f["title"] for f in report.get("findings", [])],
        "cost": (report.get("cost") or {}).get("est_cost_usd"),
    }


def safe_member_path(name: str) -> PurePosixPath:
    """A zip member's path, refused if it is absolute or climbs out of the upload folder."""
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise ValueError(f"unsafe path in zip: {name!r}")
    return path


# --- HTTP ---------------------------------------------------------------------------------


def make_handler(app: AuditorApp, port: int) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "MLAuditor"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _host_ok(self) -> bool:
            if self.headers.get("Host") not in allowed_hosts:
                self._send(HTTPStatus.FORBIDDEN, {"error": "requests must come from this computer (localhost)"})
                return False
            return True

        def _send(self, status: HTTPStatus, body: Any, content_type: str = "application/json") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return
            url = urlparse(self.path)
            if url.path == "/":
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/projects":
                self._send(HTTPStatus.OK, {"projects": app.projects(), "has_key": app.client_factory() is not None})
            elif m := re.fullmatch(r"/api/jobs/([0-9a-f]{12})", url.path):
                job = app.jobs.get(m.group(1))
                since = int(parse_qs(url.query).get("since", ["0"])[0] or 0)
                self._send(HTTPStatus.OK, job.view(since)) if job else self._send(HTTPStatus.NOT_FOUND, {})
            elif m := re.fullmatch(r"/reports/([0-9a-f]{12})/report.html", url.path):
                job = app.jobs.get(m.group(1))
                if job and job.report and job.report.is_file():
                    self._send(HTTPStatus.OK, job.report.read_bytes(), "text/html; charset=utf-8")
                else:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "no report yet"})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD_BYTES:
                self.close_connection = True
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "upload too large"})
                return
            # Read the body before any refusal, so the client gets the answer instead of a reset.
            body = self.rfile.read(length)
            if not self._host_ok():
                return
            if self.headers.get(POST_HEADER) != "1":
                self._send(HTTPStatus.FORBIDDEN, {"error": f"missing {POST_HEADER} header"})
                return
            try:
                if self.path == "/api/upload":
                    name = self.headers.get("X-Filename") or "project.zip"
                    self._send(HTTPStatus.OK, app.upload(name, body))
                elif self.path == "/api/audit":
                    request = json.loads(body or b"{}")
                    job = app.start(str(request.get("project")), str(request.get("mode", "live")))
                    self._send(HTTPStatus.OK, {"job": job.id})
                else:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except RuntimeError as exc:
                self._send(HTTPStatus.CONFLICT, {"error": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return Handler


def serve(app: AuditorApp, port: int = 8000) -> ThreadingHTTPServer:
    """Bind to 127.0.0.1 (port 0 picks a free port). Call serve_forever() on the result."""
    server = ThreadingHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
    server.RequestHandlerClass = make_handler(app, server.server_address[1])
    server.daemon_threads = True
    return server
