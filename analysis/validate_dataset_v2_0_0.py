"""Validate the extracted tenhou-to-mjai v2.0.0 dataset."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.dataset_validation import (
    DEFAULT_ERROR_SAMPLE_LIMIT,
    build_dataset_validation_summary,
    build_year_validation_summary,
    load_release_asset_manifest,
    validate_raw_year_directory,
    validate_release_archive,
    write_dataset_validation_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRST_YEAR = 2009
LAST_YEAR = 2025
ALL_YEARS = tuple(range(FIRST_YEAR, LAST_YEAR + 1))
TARGET_RULE_CODE = "00a9"
PROGRESS_INTERVAL = 10_000
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "data" / "archives" / "v2.0.0"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_ASSET_MANIFEST = (
    PROJECT_ROOT / "data" / "validation" / "tenhou-to-mjai-v2.0.0-assets.json"
)
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT / "data" / "validation" / "tenhou-to-mjai-v2.0.0-summary.json"
)
DEFAULT_OUTPUT_MARKDOWN = (
    PROJECT_ROOT / "data" / "validation" / "tenhou-to-mjai-v2.0.0-summary.md"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate archive metadata and the first MJAI event for each file."
        )
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        choices=ALL_YEARS,
        default=list(ALL_YEARS),
        help="years to validate; defaults to 2009 through 2025",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
    )
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=DEFAULT_ASSET_MANIFEST,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
    )
    parser.add_argument(
        "--skip-archive-hash",
        action="store_true",
        help="skip SHA-256 calculation for a development run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate selected years, always writing summaries for data failures."""
    args = parse_args(argv)
    years = tuple(sorted(set(args.years)))
    manifest = load_release_asset_manifest(args.asset_manifest)
    assets_by_year = {asset.year: asset for asset in manifest.assets}
    missing_years = [year for year in years if year not in assets_by_year]
    if missing_years:
        missing = ", ".join(str(year) for year in missing_years)
        raise ValueError(f"asset manifest does not contain years: {missing}")

    year_summaries = []
    for year in years:
        print(f"validating {year}...", flush=True)
        asset = assets_by_year[year]
        archive = validate_release_archive(
            asset,
            args.archive_root / asset.filename,
            skip_sha256=args.skip_archive_hash,
            relative_path=f"data/archives/{manifest.release_tag}/{asset.filename}",
        )

        def report_progress(processed: int, *, current_year: int = year) -> None:
            if processed % PROGRESS_INTERVAL == 0:
                print(
                    f"{current_year}: processed {processed:,} files",
                    flush=True,
                )

        raw = validate_raw_year_directory(
            year,
            args.raw_root / str(year),
            target_rule_code=TARGET_RULE_CODE,
            error_sample_limit=DEFAULT_ERROR_SAMPLE_LIMIT,
            progress_callback=report_progress,
        )
        year_summary = build_year_validation_summary(year, archive, raw)
        year_summaries.append(year_summary)
        status = "FAILED" if year_summary.validation.failed else "passed"
        print(
            f"{year}: {raw.total_mjson_files:,} files, "
            f"{raw.target_games:,} targets, {status}",
            flush=True,
        )
        del archive, raw

    summary = build_dataset_validation_summary(
        manifest,
        year_summaries,
        target_rule_code=TARGET_RULE_CODE,
    )
    write_dataset_validation_outputs(
        summary,
        args.output_json,
        args.output_markdown,
    )
    print(f"wrote: {args.output_json}", flush=True)
    print(f"wrote: {args.output_markdown}", flush=True)

    if summary.validation_failed:
        print("dataset validation failed; see summary for details", flush=True)
        return 1
    if not summary.archive_hashes_verified:
        print("archive SHA-256 verification was skipped", flush=True)
    print("dataset validation passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
