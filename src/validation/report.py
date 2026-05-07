"""HTML and JSON reporting for validation outcomes."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from src.utils.config import resolve_project_path
from src.validation.io import write_json_atomic, write_text_atomic
from src.validation.types import ValidationIssue, ValidationResult


def write_json_report(result: ValidationResult, output_path: str | Path) -> Path:
    """Write the machine-readable validation summary."""

    path = resolve_project_path(output_path)
    result.json_report_path = path
    return write_json_atomic(path, result.to_dict())


def write_html_report(result: ValidationResult, output_path: str | Path) -> Path:
    """Write a standalone HTML report suitable for CI artifacts."""

    path = resolve_project_path(output_path)
    result.html_report_path = path
    return write_text_atomic(path, _render_html(result))


def write_reports(result: ValidationResult, report_dir: str | Path) -> ValidationResult:
    """Write both latest JSON and HTML reports into the configured directory."""

    output_dir = resolve_project_path(report_dir)
    write_html_report(result, output_dir / "validation_report.html")
    write_json_report(result, output_dir / "validation_summary.json")
    return result


def _render_html(result: ValidationResult) -> str:
    status = "PASSED" if result.passed else "FAILED"
    status_class = "pass" if result.passed else "fail"
    issue_rows = "\n".join(_issue_row(issue) for issue in result.issues) or (
        "<tr><td colspan='5'>No validation issues detected.</td></tr>"
    )
    metrics_json = html.escape(json.dumps(result.metrics, indent=2, sort_keys=True, default=str))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Data Validation Report - {html.escape(result.dataset_name)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      --border: #d7dde8;
      --text: #202938;
      --muted: #5b6677;
      --pass: #16794c;
      --fail: #b42318;
      --warn: #a15c07;
      --panel: #f7f9fc;
    }}
    body {{
      color: var(--text);
      margin: 0;
      background: #ffffff;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .metric {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--panel);
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      color: white;
      font-weight: 700;
      padding: 6px 12px;
    }}
    .status.pass {{ background: var(--pass); }}
    .status.fail {{ background: var(--fail); }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 16px 0 28px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
      font-weight: 700;
    }}
    .severity-error {{
      color: var(--fail);
      font-weight: 700;
    }}
    .severity-warning {{
      color: var(--warn);
      font-weight: 700;
    }}
    pre {{
      background: #101828;
      color: #f5f7fb;
      border-radius: 8px;
      overflow-x: auto;
      padding: 16px;
      line-height: 1.45;
    }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
<main>
  <h1>Data Validation Report</h1>
  <p class="muted">
    {html.escape(result.dataset_name)} generated at {html.escape(result.generated_at)}
  </p>
  <span class="status {status_class}">{status}</span>

  <section class="summary">
    {_metric_card("Rows", result.row_count)}
    {_metric_card("Columns", result.column_count)}
    {_metric_card("Errors", len(result.errors))}
    {_metric_card("Warnings", len(result.warnings))}
  </section>

  <h2>Issues</h2>
  <table>
    <thead>
      <tr>
        <th>Severity</th>
        <th>Check</th>
        <th>Column</th>
        <th>Message</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
      {issue_rows}
    </tbody>
  </table>

  <h2>Metrics</h2>
  <pre>{metrics_json}</pre>
</main>
</body>
</html>
"""


def _metric_card(label: str, value: Any) -> str:
    return (
        "<div class='metric'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(str(value))}</div>"
        "</div>"
    )


def _issue_row(issue: ValidationIssue) -> str:
    details = html.escape(json.dumps(issue.details, sort_keys=True, default=str))
    severity = html.escape(issue.severity)
    return (
        "<tr>"
        f"<td class='severity-{severity}'>{severity.upper()}</td>"
        f"<td>{html.escape(issue.check)}</td>"
        f"<td>{html.escape(issue.column or '-')}</td>"
        f"<td>{html.escape(issue.message)}</td>"
        f"<td><code>{details}</code></td>"
        "</tr>"
    )
