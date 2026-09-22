"""Capture every kokushi-riichi hand at the accepted-riichi viewer event."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "kokushi-riichi-outcomes-2009-2025.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kokushi-riichi-captures"

VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
START_CLICK = (960, 630)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture kokushi-riichi hands from the Tenhou HTML5 viewer."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=int, default=1, help="One-based first record")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--tj-offset",
        type=int,
        default=0,
        help="Calibration offset added to the accepted-riichi event index",
    )
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quality", type=int, default=84)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start < 1:
        raise ValueError("--start must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not 1 <= args.quality <= 95:
        raise ValueError("--quality must be from 1 through 95")

    document = json.loads(args.input.read_text(encoding="utf-8"))
    records = document.get("records")
    if not isinstance(records, list):
        raise TypeError("input records must be a list")
    stop = None if args.limit is None else args.start - 1 + args.limit
    selected = list(enumerate(records, 1))[args.start - 1 : stop]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "capture-log.json"
    failure_path = args.output_dir / "failed.json"
    prior = _load_existing_log(log_path) if args.resume else []
    prior_by_index = {item["index"]: item for item in prior}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headful)
        page = browser.new_page(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            device_scale_factor=1,
        )
        for index, record in selected:
            filename = _filename(index, record)
            output = args.output_dir / filename
            if args.resume and output.is_file() and index in prior_by_index:
                print(f"[{index}/{len(records)}] exists: {filename}", flush=True)
                results.append(prior_by_index[index])
                continue
            try:
                result = _capture_one(
                    page,
                    record,
                    index=index,
                    total=len(records),
                    output=output,
                    tj_offset=args.tj_offset,
                    quality=args.quality,
                )
            except (
                OSError,
                PlaywrightError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                print(f"[{index}/{len(records)}] FAILED: {error}", flush=True)
                failures.append(
                    {
                        "index": index,
                        "log_id": record.get("log_id"),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                results.append(result)
            _write_json(log_path, sorted(results, key=lambda item: item["index"]))
            _write_json(failure_path, failures)
        browser.close()

    print(f"captured: {len(results)}", flush=True)
    print(f"failed: {len(failures)}", flush=True)
    print(f"output: {args.output_dir.resolve()}", flush=True)
    return 1 if failures else 0


def _capture_one(
    page: Page,
    record: dict[str, Any],
    *,
    index: int,
    total: int,
    output: Path,
    tj_offset: int,
    quality: int,
) -> dict[str, Any]:
    tj = int(record["riichi_tj"]) + tj_offset
    if tj < 0:
        raise ValueError("calibrated tj must not be negative")
    separator = "&" if "?" in record["url"] else "?"
    capture_url = f"{record['url']}{separator}tj={tj}"
    print(
        f"[{index}/{total}] {record['year']} {record['log_id']} tj={tj}",
        flush=True,
    )

    screenshot: bytes | None = None
    for attempt in range(1, 4):
        page.goto(capture_url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(1_000)
        page.mouse.click(*START_CLICK)
        page.wait_for_timeout(1_500)
        candidate = page.screenshot(full_page=False)
        if _is_table(candidate):
            screenshot = candidate
            break
        page.mouse.click(*START_CLICK)
        page.wait_for_timeout(1_500)
        candidate = page.screenshot(full_page=False)
        if _is_table(candidate):
            screenshot = candidate
            break
        print(f"  viewer retry {attempt}/3", flush=True)
    if screenshot is None:
        raise RuntimeError("Tenhou viewer did not reach the table")

    image = Image.open(BytesIO(screenshot)).convert("RGB")
    image = image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
    image.save(output, format="JPEG", quality=quality, optimize=True, progressive=True)
    return {
        "index": index,
        "year": record["year"],
        "date": record["date"],
        "log_id": record["log_id"],
        "kyoku_index": record["kyoku_index"],
        "who": record["who"],
        "turn": record["turn"],
        "reach_type": record["reach_type"],
        "wait_type": record["wait_type"],
        "waits": record["waits"],
        "outcome": record["outcome"],
        "won": record["won"],
        "win_method": record["win_method"],
        "url": record["url"],
        "capture_url": capture_url,
        "riichi_tj": record["riichi_tj"],
        "tj_offset": tj_offset,
        "file": output.name,
    }


def _is_table(png: bytes) -> bool:
    image = Image.open(BytesIO(png)).convert("L").resize((240, 135))
    histogram = image.histogram()
    nonblack_ratio = sum(histogram[11:]) / (image.width * image.height)
    return nonblack_ratio >= 0.15


def _filename(index: int, record: dict[str, Any]) -> str:
    return (
        f"{index:03d}_{record['year']}_{record['log_id']}"
        f"_k{record['kyoku_index']:02d}.jpg"
    )


def _load_existing_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("existing capture log must be a list")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
