"""Typed result objects for validation checks and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding emitted by schema, quality, or drift checks."""

    check: str
    severity: Severity
    message: str
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "column": self.column,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Complete validation outcome used by the CLI, reports, and tests."""

    dataset_name: str
    input_path: Path
    row_count: int
    column_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    html_report_path: Path | None = None
    json_report_path: Path | None = None
    baseline_path: Path | None = None
    fail_on_warnings: bool = False

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def passed(self) -> bool:
        if self.errors:
            return False
        return not (self.fail_on_warnings and self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "input_path": str(self.input_path),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "fail_on_warnings": self.fail_on_warnings,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "html_report_path": str(self.html_report_path) if self.html_report_path else None,
            "json_report_path": str(self.json_report_path) if self.json_report_path else None,
            "baseline_path": str(self.baseline_path) if self.baseline_path else None,
        }
