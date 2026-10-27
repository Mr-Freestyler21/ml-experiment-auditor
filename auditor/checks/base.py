"""The structured result every deterministic check returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["flagged", "clear", "not_applicable", "error"]


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: Status
    summary: str
    measurements: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
