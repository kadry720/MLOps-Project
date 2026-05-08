"""Command-line entrypoint for CI/CD data validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.exceptions import DataValidationError  # noqa: E402
from src.validation.validator import run_validation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run schema, quality, and drift validation before training/deployment."
    )
    parser.add_argument(
        "--config",
        default="configs/validation.yaml",
        help="Path to the validation YAML configuration.",
    )
    parser.add_argument(
        "--input-data",
        default=None,
        help="Optional dataset path override. Environment: DATA_VALIDATION_INPUT_PATH.",
    )
    parser.add_argument(
        "--reference-data",
        default=None,
        help=(
            "Optional reference dataset path override. "
            "Environment: DATA_VALIDATION_REFERENCE_PATH."
        ),
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "Recompute the baseline statistics file from reference data, "
            "or current data if no reference is set."
        ),
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat warnings as CI failures for stricter release gates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_validation(
            config_path=args.config,
            input_path=args.input_data,
            reference_path=args.reference_data,
            update_baseline=args.update_baseline,
            fail_on_warnings=True if args.fail_on_warnings else None,
        )
    except DataValidationError as exc:
        print(f"Data validation failed before checks completed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI must guard CI from trace-only failures.
        print(f"Unexpected validation runtime error: {exc}", file=sys.stderr)
        return 1

    status = "PASSED" if result.passed else "FAILED"
    print(
        f"Data validation {status}: "
        f"{len(result.errors)} errors, {len(result.warnings)} warnings."
    )
    print(f"HTML report: {result.html_report_path}")
    print(f"JSON summary: {result.json_report_path}")

    if not result.passed:
        print("Top validation errors:", file=sys.stderr)
        for issue in result.errors[:10]:
            print(f"- [{issue.check}] {issue.column or '-'}: {issue.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
