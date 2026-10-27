"""Command line entry point.

  python -m auditor.cli baseline demo/f1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auditor.baseline import TOLERANCE, BaselineResult, run_baseline
from auditor.config import ConfigError, load_config
from auditor.sandbox import default_sandbox

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auditor.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", help="rerun a project and check its reported metric reproduces")
    baseline.add_argument("project", type=Path, help="folder containing audit.yaml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.project)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    print("Preparing sandbox environment (first run installs packages)...")
    sandbox = default_sandbox(REPO_ROOT)
    print(f"Running {config.entry_script} in the sandbox (timeout {config.timeout_seconds}s)...")
    result = run_baseline(config, sandbox)
    _print_baseline(result)
    return 0 if result.status == "reproduced" else 1


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


if __name__ == "__main__":
    sys.exit(main())
