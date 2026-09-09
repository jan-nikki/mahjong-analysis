"""Run the formal production/reference riichi-wait comparison."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.riichi_wait_reference_comparison import (
    FieldMismatch,
    ProcessingError,
    ProductionOnly,
    ReferenceOnly,
    RiichiWaitComparisonReport,
    ScopeMismatch,
    compare_riichi_wait_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_YEAR = 2025
DEFAULT_MAX_FILES = 100
DEFAULT_DIFFERENCE_LIMIT = 10


def select_input_files(
    raw_root: str | Path,
    year: int,
    *,
    max_files: int | None = DEFAULT_MAX_FILES,
    files: Sequence[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Select one deterministic file set before either side checks scope."""
    root = Path(raw_root).resolve()
    if type(year) is not int or year < 1:
        raise ValueError("year must be a positive integer")
    if files is not None and max_files is not None:
        raise ValueError("files and max_files are mutually exclusive")

    if files is not None:
        selected = tuple(_resolve_explicit_file(root, year, value) for value in files)
    else:
        if type(max_files) is not int or max_files < 1:
            raise ValueError("max_files must be a positive integer")
        year_root = root / str(year)
        if not year_root.is_dir():
            raise FileNotFoundError(f"raw year directory does not exist: {year_root}")
        all_paths = sorted(
            year_root.rglob("*.mjson"),
            key=lambda path: path.resolve().relative_to(root).as_posix(),
        )
        selected = tuple(all_paths[:max_files])

    if not selected:
        raise FileNotFoundError("no MJAI files selected")
    by_source: dict[str, Path] = {}
    for path in selected:
        source = _relative_source_path(path, root)
        if source in by_source:
            raise ValueError(f"duplicate selected file: {source}")
        by_source[source] = path
    return tuple(by_source[source] for source in sorted(by_source))


def render_comparison_report(
    report: RiichiWaitComparisonReport,
    *,
    difference_limit: int = DEFAULT_DIFFERENCE_LIMIT,
) -> str:
    """Render summary counts and a bounded deterministic difference sample."""
    if type(difference_limit) is not int or difference_limit < 0:
        raise ValueError("difference_limit must be a non-negative integer")
    lines = [
        "# Riichi wait production/reference comparison",
        "",
        f"scanned_files: {report.scanned_files}",
        f"production_target_games: {report.production_target_games}",
        f"reference_target_games: {report.reference_target_games}",
        f"production_east_kyokus: {report.production_east_kyokus}",
        f"reference_east_kyokus: {report.reference_east_kyokus}",
        f"production_candidates: {report.production_candidates}",
        f"reference_candidates: {report.reference_candidates}",
        f"production_only: {len(report.production_only)}",
        f"reference_only: {len(report.reference_only)}",
        f"field_mismatch: {len(report.field_mismatches)}",
        f"scope_mismatch: {len(report.scope_mismatches)}",
        f"processing_error: {len(report.processing_errors)}",
        f"result: {'PASS' if report.is_pass else 'FAIL'}",
    ]
    differences: tuple[object, ...] = (
        *report.production_only,
        *report.reference_only,
        *report.field_mismatches,
        *report.scope_mismatches,
        *report.processing_errors,
    )
    if differences and difference_limit:
        lines.extend(("", "## Difference samples", ""))
        lines.extend(
            _format_difference(value) for value in differences[:difference_limit]
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deterministic comparison options."""
    parser = argparse.ArgumentParser(
        description="Compare production and independent-reference riichi waits."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--year", type=_positive_int, default=DEFAULT_YEAR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--max-files",
        type=_positive_int,
        help=f"sorted files to compare before scope filtering (default: {DEFAULT_MAX_FILES})",
    )
    selection.add_argument(
        "--files",
        nargs="+",
        help="raw-root-relative paths, year-relative paths, or absolute paths",
    )
    parser.add_argument(
        "--difference-limit",
        type=_non_negative_int,
        default=DEFAULT_DIFFERENCE_LIMIT,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and return nonzero for any classified difference."""
    args = parse_args(argv)
    paths = select_input_files(
        args.raw_root,
        args.year,
        max_files=(
            None if args.files is not None else args.max_files or DEFAULT_MAX_FILES
        ),
        files=args.files,
    )
    report = compare_riichi_wait_files(paths, raw_root=args.raw_root)
    print(
        render_comparison_report(report, difference_limit=args.difference_limit),
        end="",
    )
    return 0 if report.is_pass else 1


def _resolve_explicit_file(root: Path, year: int, value: str | Path) -> Path:
    requested = Path(value)
    if requested.is_absolute():
        candidates = (requested,)
    else:
        candidates = (root / requested, root / str(year) / requested)
    existing = tuple(
        path.resolve()
        for path in candidates
        if path.is_file() and path.suffix == ".mjson"
    )
    unique = tuple(dict.fromkeys(existing))
    if not unique:
        raise FileNotFoundError(f"explicit MJAI file does not exist: {value}")
    if len(unique) > 1:
        raise ValueError(f"explicit MJAI file is ambiguous: {value}")
    _relative_source_path(unique[0], root)
    return unique[0]


def _relative_source_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"selected file is outside raw_root: {path}") from error


def _format_difference(value: object) -> str:
    if isinstance(value, ProductionOnly):
        return (
            "[production_only] "
            f"key={value.key!r} actor={value.candidate.actor} "
            f"lines={value.candidate.related_lines!r}"
        )
    if isinstance(value, ReferenceOnly):
        return (
            "[reference_only] "
            f"key={value.key!r} actor={value.candidate.actor} "
            f"lines={value.candidate.related_lines!r}"
        )
    if isinstance(value, FieldMismatch):
        return (
            "[field_mismatch] "
            f"key={value.key!r} field={value.field_name} "
            f"actors=({value.production_actor},{value.reference_actor}) "
            f"production_lines={value.production_lines!r} "
            f"reference_lines={value.reference_lines!r} "
            f"production={value.production_value!r} "
            f"reference={value.reference_value!r}"
        )
    if isinstance(value, ScopeMismatch):
        return (
            "[scope_mismatch] "
            f"source={value.source_path} field={value.field_name} "
            f"production={value.production_value!r} "
            f"reference={value.reference_value!r}"
        )
    if isinstance(value, ProcessingError):
        return (
            "[processing_error] "
            f"source={value.source_path} "
            f"production={value.production_error!r} "
            f"reference={value.reference_error!r}"
        )
    raise TypeError(f"unsupported difference type: {type(value).__name__}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
