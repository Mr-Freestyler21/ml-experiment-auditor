"""Run the benchmark.

    python -m benchmark.run --split dev
    python -m benchmark.run --split heldout
    python -m benchmark.run --only dev_c_groups --systems agent --repeats 3

Every project is audited from an anonymous copy (p01, p02, ...) so that neither the
folder name nor any log path can hint at the planted flaw. Answer keys stay in
benchmark/answers/ and are only read here, for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auditor.agent.llm_client import meta_client_from_env
from auditor.config import load_config
from auditor.sandbox import LocalSandbox, ensure_sandbox_python
from benchmark import score as scoring
from benchmark.systems import run_agent_system, run_llm_only, run_rules, save

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PROJECTS = ROOT / "projects"
ANSWERS = ROOT / "answers"
RESULTS = ROOT / "results"
SYSTEMS = ("rules", "agent", "llm_only")


def load_answers() -> dict[str, dict[str, Any]]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(ANSWERS.glob("*.json"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    parser.add_argument("--only", nargs="*", help="project ids")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--repeats", type=int, default=1, help="agent runs per project")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--name", help="results folder name (default: a timestamp)")
    args = parser.parse_args()

    answers = load_answers()
    ids = [i for i in answers if (args.split == "all" or answers[i]["split"] == args.split)]
    if args.only:
        ids = [i for i in ids if i in args.only]
    systems = [s for s in args.systems.split(",") if s in SYSTEMS]
    out = RESULTS / (args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    python = ensure_sandbox_python(REPO / ".sandbox-venv", REPO / "sandbox-requirements.txt")

    # Anonymous working copies, named by a hash so reruns into the same folder never collide.
    anon = {pid: "p" + hashlib.sha1(pid.encode()).hexdigest()[:6] for pid in ids}
    for pid, name in anon.items():
        target = out / "work" / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(PROJECTS / pid, target)
    mapping_path = out / "mapping.json"
    previous = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    save(mapping_path, {**previous, **anon})

    tasks = [(pid, system, k) for pid in ids for system in systems
             for k in range(args.repeats if system == "agent" else 1)]
    print(f"{len(ids)} projects, systems {systems}, {len(tasks)} runs -> {out}")

    def run_one(pid: str, system: str, k: int) -> dict[str, Any]:
        name = anon[pid]
        config = load_config(out / "work" / name)
        sandbox = LocalSandbox(python=python, runs_dir=out / "runs" / f"{name}_{system}_{k}")
        try:
            if system == "rules":
                result = run_rules(config, sandbox)
            else:
                client = meta_client_from_env(REPO)
                if client is None:
                    raise RuntimeError("META_MODEL_API_KEY is not set")
                if system == "agent":
                    result = run_agent_system(config, sandbox, client, out / "audits" / f"{name}_r{k}")
                else:
                    result = run_llm_only(config, client)
        except Exception as exc:  # one broken run must not stop the benchmark
            result = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "families": []}
        record = {"id": pid, "anon": name, "system": system, "repeat": k, "result": result}
        save(out / "raw" / f"{pid}__{system}__{k}.json", record)
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, *t) for t in tasks]
        for future in as_completed(futures):
            r = future.result()
            res = r["result"]
            status = res.get("error") or f"{res.get('verdict', '')} flags={res.get('families')}"
            print(f"  {r['id']:28s} {r['system']:9s} r{r['repeat']}  {str(status)[:90]}", flush=True)

    # Summarise every run in the folder, so a rerun of a few projects updates the full table.
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "raw").glob("*.json"))]
    all_ids = [i for i in answers if any(r["id"] == i for r in records)]
    write_summary(out, records, answers, all_ids)


def write_summary(
    out: Path,
    records: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    ids: list[str],
    consistency_records: list[dict[str, Any]] | None = None,
    extra_notes: list[str] = (),
    name: str = "results",
) -> None:
    """Write <name>.md and <name>.json into `out`. Repeats come from `consistency_records` (default: `records`)."""
    scores = {
        "All projects": scoring.score(records, answers, None),
        "Development set": scoring.score(records, answers, "dev"),
        "Held-out set": scoring.score(records, answers, "heldout"),
    }
    first = {(r["id"], r["system"]): r["result"] for r in records if r["repeat"] == 0}

    def flags(pid: str, system: str) -> str:
        res = first.get((pid, system))
        if res is None:
            return ""
        if res.get("error"):
            return "error"
        fams = "+".join(res.get("families", [])) or "-"
        return f"{res.get('verdict')} {fams}" if system != "rules" else fams

    per_project = []
    for pid in ids:
        a = answers[pid]
        truth = ", ".join(f"{f['family']} ({f['taxonomy']})" for f in a["flaws"]) or "clean"
        agent = first.get((pid, "agent"), {})
        llm = first.get((pid, "llm_only"), {})
        per_project.append({
            "id": pid, "split": a["split"], "truth": truth, "correct": a["correct_metric"],
            "rules": flags(pid, "rules"), "agent": flags(pid, "agent"),
            "patched": f"{agent['patched_metric']:.4f}" if isinstance(agent.get("patched_metric"), float) else "",
            "llm_only": flags(pid, "llm_only"),
            "estimate": f"{llm['estimated_correct_metric']:.4f}"
            if isinstance(llm.get("estimated_correct_metric"), (int, float)) else "",
        })
    meta = {
        "description": (
            f"{len(ids)} projects. A system flags a family only when it commits to it: a flagged check (rules), a "
            "confirmed finding (Muse + tools) or a listed flaw (Muse only). \"Corrected\" counts flawed projects whose "
            f"patched evaluation landed within {scoring.CORRECTION_TOLERANCE} of the answer key's correct metric."
        ),
        "notes": [
            "Answer keys: benchmark/answers/. Held-out keys were committed with HELDOUT_LOCK.txt before any audit ran.",
            "Taxonomy codes follow Kapoor & Narayanan (2023), Leakage and the reproducibility crisis in "
            "machine-learning-based science.",
            "dev_a_pandas_scaler is included as a known limit: the tracer only sees scikit-learn.",
            "Some planted preprocessing flaws have almost no effect on the score (their correct metric equals the "
            "reported one); they are still flaws in procedure.",
            *extra_notes,
        ],
    }
    consistency = scoring.consistency(records if consistency_records is None else consistency_records)
    md = scoring.markdown(scores, per_project, consistency, meta)
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    save(out / f"{name}.json", {"scores": scores, "per_project": per_project, "consistency": consistency})
    print(md)


if __name__ == "__main__":
    main()
