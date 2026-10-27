"""Load and validate an audit target's audit.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "audit.yaml"
REQUIRED = ("entry_script", "data_path", "target", "reported_metric", "reported_value")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AuditConfig:
    project_dir: Path
    entry_script: str
    data_path: str
    target: str
    reported_metric: str
    reported_value: float
    prediction_time: str | None = None
    # Each group is one or more columns that together identify an entity, e.g. ("Year", "Race").
    group_columns: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    time_column: str | None = None
    random_seed: int | None = None
    timeout_seconds: int = 600


def load_config(project_dir: Path) -> AuditConfig:
    project_dir = Path(project_dir).resolve()
    path = project_dir / CONFIG_NAME
    if not path.is_file():
        raise ConfigError(f"no {CONFIG_NAME} in {project_dir}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{CONFIG_NAME} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{CONFIG_NAME} must be a mapping of keys to values")

    missing = [key for key in REQUIRED if raw.get(key) in (None, "")]
    if missing:
        raise ConfigError(f"{CONFIG_NAME} is missing required keys: {', '.join(missing)}")

    config = AuditConfig(
        project_dir=project_dir,
        entry_script=_str(raw, "entry_script"),
        data_path=_str(raw, "data_path"),
        target=_str(raw, "target"),
        reported_metric=_str(raw, "reported_metric"),
        reported_value=_number(raw, "reported_value"),
        prediction_time=_optional_str(raw, "prediction_time"),
        group_columns=_groups(raw.get("group_columns")),
        time_column=_optional_str(raw, "time_column"),
        random_seed=int(raw["random_seed"]) if raw.get("random_seed") is not None else None,
        timeout_seconds=int(raw.get("timeout_seconds") or 600),
    )
    for key in ("entry_script", "data_path"):
        if not (project_dir / getattr(config, key)).is_file():
            raise ConfigError(f"{key} points to a file that does not exist: {getattr(config, key)}")
    if config.timeout_seconds <= 0:
        raise ConfigError("timeout_seconds must be positive")
    return config


def _str(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    return _str(raw, key) if raw.get(key) is not None else None


def _number(raw: dict[str, Any], key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _groups(value: Any) -> tuple[tuple[str, ...], ...]:
    """Accept [a, b] (two single-column groups) or [[a, b], c] (a composite group and a single one)."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError("group_columns must be a list")
    groups = []
    for item in value:
        columns = item if isinstance(item, list) else [item]
        if not columns or not all(isinstance(c, str) for c in columns):
            raise ConfigError(f"group_columns entries must be column names or lists of names: {item!r}")
        groups.append(tuple(columns))
    return tuple(groups)
