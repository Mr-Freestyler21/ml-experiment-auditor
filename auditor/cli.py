"""Command line entry point.

  python -m auditor.cli baseline demo/f1
  python -m auditor.cli check demo/f1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from auditor.baseline import TOLERANCE, BaselineResult, run_baseline
from auditor.checks.base import CheckResult
from auditor.checks.group_overlap import check_group_overlap
from auditor.config import AuditConfig, ConfigError, load_config
from auditor.sandbox import Sandbox, default_sandbox
from auditor.trace import compare_metrics, run_traced

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auditor.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", help="rerun a project and check its reported metric reproduces")
    baseline.add_argument("project", type=Path, help="folder containing audit.yaml")
    check = commands.add_parser("check", help="run the deterministic leakage checks on a project")
    check.add_argument("project", type=Path, help="folder containing audit.yaml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.project)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    print("Preparing sandbox environment (first run installs packages)...")
    sandbox = default_sandbox(REPO_ROOT)
    if args.command == "baseline":
        return _baseline(config, sandbox)
    return _check(config, sandbox)


def _baseline(config: AuditConfig, sandbox: Sandbox) -> int:
    print(f"Running {config.entry_script} in the sandbox (timeout {config.timeout_seconds}s)...")
    result = run_baseline(config, sandbox)
    _print_baseline(result)
    return 0 if result.status == "reproduced" else 1


def _check(config: AuditConfig, sandbox: Sandbox) -> int:
    print(f"Running {config.entry_script} in the sandbox as-is...")
    plain = run_baseline(config, sandbox)
    if plain.status == "failed":
        print(f"Run failed: {plain.error}. See {plain.log_dir}", file=sys.stderr)
        return 1
    print(f"Running {config.entry_script} again with call tracing...")
    traced = run_traced(config, sandbox)
    if traced.error:
        print(f"Run failed: {traced.error}. See {traced.run.log_dir}", file=sys.stderr)
        return 1

    same = compare_metrics(plain.metrics, traced.metrics)
    print(f"Tracing changed the results: {'no' if same['match'] else 'YES ' + json.dumps(same['differences'])}")
    print(f"Traced {len(traced.events)} calls; found {len(traced.splits)} train/test evaluation(s).")
    for split in traced.splits:
        print(
            f"  {split.model}: fit at line {split.fit_line} on {split.n_train} rows, "
            f"{'/'.join(split.predict_methods)} at line {split.predict_line} on {split.n_test} rows"
        )
    incomplete = traced.incomplete_reason
    if not same["match"]:
        incomplete = "the traced run's metrics differ from the plain run, so the trace may not describe the same experiment"

    data = pd.read_csv(config.project_dir / config.data_path)
    result = check_group_overlap(data, traced.splits, config.group_columns, config.time_column, incomplete)
    (traced.run.log_dir / "metrics_comparison.json").write_text(json.dumps(same, indent=2), encoding="utf-8")
    (traced.run.log_dir / f"check_{result.check_id}.json").write_text(
        json.dumps(result.to_json(), indent=2, default=str), encoding="utf-8"
    )
    _print_check(result)
    print(f"Evidence:  {traced.run.log_dir}")
    return 0


def _print_baseline(result: BaselineResult) -> None:
    print()
    print(f"Status:    {result.status}")
    if result.error:
        print(f"Error:     {result.error}")
    else:
        print(
            f"{result.reported_metric}: reported {result.reported_value}, measured {result.measured_value:.4f} "
            f"(difference {result.difference:+.4f}, tolerance {TOLERANCE})"
        )
        for name, value in result.metrics.items():
            if name != result.reported_metric:
                print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    env = result.environment
    packages = ", ".join(f"{k} {v}" for k, v in env.get("packages", {}).items())
    print(f"Seed:      {result.random_seed}")
    print(f"Python:    {env.get('python')} ({packages})")
    print(f"Duration:  {result.duration_s:.1f}s")
    print(f"Evidence:  {result.log_dir}")


def _print_check(result: CheckResult) -> None:
    print()
    print(f"[{result.check_id}] {result.status.upper()}")
    print(f"  {result.summary}")
    for split in result.evidence.get("splits", [])[:1]:
        for g in split.get("groups", []):
            for ex in g["examples"][:3]:
                print(f"    e.g. {ex['group']}: {ex['train_rows']} train rows, {ex['test_rows']} test rows")


if __name__ == "__main__":
    sys.exit(main())
