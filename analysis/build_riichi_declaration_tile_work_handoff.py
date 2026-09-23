"""Build compact ChatGPT Work materials for the declaration-tile analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "research/results/riichi-declaration-tile-analysis-v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/work/riichi-declaration-tile-analysis"

CATEGORIES = ("19", "28", "37", "46", "5", "honor", "dora")
SUITED_CATEGORIES = ("19", "28", "37", "46", "5")
LABELS = {
    "19": "19切り",
    "28": "28切り",
    "37": "37切り",
    "46": "46切り",
    "5": "5切り",
    "honor": "字牌切り",
    "dora": "ドラ切り",
}
COLORS = {
    "19": "#2F6FED",
    "28": "#14857B",
    "37": "#7F56D9",
    "46": "#E87932",
    "5": "#D92D20",
    "honor": "#344054",
    "dora": "#D4A72C",
}
BACKGROUND = "#F7F5EF"
INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"
PANEL = "#FFFFFF"
FONT_REGULAR_CANDIDATES = (
    Path(r"C:\Windows\Fonts\NotoSansJP-VF.ttf"),
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
)
FONT_BOLD_CANDIDATES = (
    Path(r"C:\Windows\Fonts\YuGothB.ttc"),
    Path(r"C:\Windows\Fonts\meiryob.ttc"),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_REGULAR_CANDIDATES
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError("a Japanese font was not found")


def _load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    metadata = result.get("metadata", {})
    if (
        metadata.get("analysis_name") != "riichi-declaration-tile-analysis-v1"
        or metadata.get("schema_version") != 3
    ):
        raise ValueError("unexpected declaration-tile result")
    return result


def _categories(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {row["declaration_category"]: row for row in value["categories"]}
    if tuple(rows) != CATEGORIES:
        raise ValueError("unexpected declaration category order")
    return rows


def _metric(value: dict[str, Any], name: str) -> dict[str, Any]:
    metric = value[name]
    return {
        "count": metric["count"],
        "denominator": metric["denominator"],
        "rate": metric["rate"],
    }


def _compact_category(value: dict[str, Any]) -> dict[str, Any]:
    names = (
        "strong_wait",
        "weak_wait",
        "other_wait",
        "contains_ryanmen",
        "contains_suji",
        "good_wait",
        "same_suji_line_wait",
        "weak_and_same_suji_line",
        "weak_and_not_same_suji_line",
        "same_suji_line_share_within_weak_wait",
        "weak_wait_rate_within_same_suji_line",
        "weak_wait_rate_excluding_same_suji_line",
    )
    return {
        "record_count": value["record_count"],
        **{name: _metric(value, name) for name in names},
        "wait_shape_record_counts": value["wait_shape_record_counts"],
        "weak_same_suji_line_wait_shape_record_counts": value[
            "weak_same_suji_line_wait_shape_record_counts"
        ],
    }


def _compact_rank_breakdown(value: dict[str, Any]) -> dict[str, Any]:
    breakdown = value["non_dora_suited_rank_breakdown"]
    return {
        "record_count": breakdown["record_count"],
        "ranks_exclusive": breakdown["ranks_exclusive"],
        "ranks": [
            {
                "declaration_rank": row["declaration_rank"],
                **_compact_category(row),
            }
            for row in breakdown["ranks"]
        ],
        "rank_wait_pairs": breakdown["rank_wait_pairs"],
    }


def _aggregate_turn_band(
    by_turn: list[dict[str, Any]], category: str, start: int, end: int
) -> dict[str, Any]:
    record_count = 0
    counts = {name: 0 for name in ("strong_wait", "weak_wait", "other_wait")}
    for turn in by_turn:
        number = turn["riichi_discard_number"]
        if number < start or number > end:
            continue
        row = _categories(turn)[category]
        record_count += row["record_count"]
        for name in counts:
            counts[name] += row[name]["count"]
    return {
        "turns": [start, end],
        "record_count": record_count,
        **{
            name: {
                "count": count,
                "denominator": record_count,
                "rate": count / record_count if record_count else None,
            }
            for name, count in counts.items()
        },
    }


def _standardized_turn_band_rate(
    by_turn: list[dict[str, Any]], category: str, start: int, end: int
) -> float:
    """Weight exact-turn category rates by the common all-record distribution."""
    weighted_sum = 0.0
    weight_sum = 0
    for turn in by_turn:
        number = turn["riichi_discard_number"]
        if number < start or number > end:
            continue
        rate = _categories(turn)[category]["strong_wait"]["rate"]
        if rate is None:
            continue
        weight = turn["record_count"]
        weighted_sum += rate * weight
        weight_sum += weight
    if not weight_sum:
        raise ValueError("turn band has no records")
    return weighted_sum / weight_sum


def _build_summary(result: dict[str, Any]) -> dict[str, Any]:
    overall = _categories(result["overall"])
    bands = {
        "turns_4_9": {
            category: _aggregate_turn_band(result["by_turn"], category, 4, 9)
            for category in ("19", "5")
        },
        "turns_10_15": {
            category: _aggregate_turn_band(result["by_turn"], category, 10, 15)
            for category in ("19", "5")
        },
        "turns_1_3": {
            category: _aggregate_turn_band(result["by_turn"], category, 1, 3)
            for category in ("honor", "dora")
        },
        "turns_7_9": {
            category: _aggregate_turn_band(result["by_turn"], category, 7, 9)
            for category in ("honor", "dora")
        },
    }
    standardized = {
        "turns_4_9": {
            category: _standardized_turn_band_rate(result["by_turn"], category, 4, 9)
            for category in ("19", "5")
        },
        "turns_10_15": {
            category: _standardized_turn_band_rate(result["by_turn"], category, 10, 15)
            for category in ("19", "5")
        },
        "turns_1_3": {
            category: _standardized_turn_band_rate(result["by_turn"], category, 1, 3)
            for category in ("honor", "dora")
        },
        "turns_7_9": {
            category: _standardized_turn_band_rate(result["by_turn"], category, 7, 9)
            for category in ("honor", "dora")
        },
    }
    comparisons = {
        "turns_4_9_19_minus_5_pp": 100
        * (standardized["turns_4_9"]["19"] - standardized["turns_4_9"]["5"]),
        "turns_10_15_5_minus_19_pp": 100
        * (standardized["turns_10_15"]["5"] - standardized["turns_10_15"]["19"]),
        "turns_1_3_dora_minus_honor_pp": 100
        * (standardized["turns_1_3"]["dora"] - standardized["turns_1_3"]["honor"]),
        "turns_7_9_honor_minus_dora_pp": 100
        * (standardized["turns_7_9"]["honor"] - standardized["turns_7_9"]["dora"]),
    }
    return {
        "analysis_name": result["metadata"]["analysis_name"],
        "schema_version": result["metadata"]["schema_version"],
        "scope": result["metadata"]["scope"],
        "input_manifest_sha256": result["metadata"]["input_manifest_sha256"],
        "definitions": result["metadata"]["definitions"],
        "overall": {
            "record_count": result["overall"]["record_count"],
            "all": _compact_category(result["overall"]["all"]),
            "categories": {
                category: _compact_category(overall[category])
                for category in CATEGORIES
            },
            "non_dora_suited_rank_breakdown": _compact_rank_breakdown(
                result["overall"]
            ),
        },
        "by_year": [
            {
                "year": row["year"],
                "record_count": row["overall"]["record_count"],
                "all": _compact_category(row["overall"]["all"]),
                "non_dora_suited_rank_breakdown": _compact_rank_breakdown(
                    row["overall"]
                ),
                "categories": {
                    category: _compact_category(_categories(row["overall"])[category])
                    for category in CATEGORIES
                },
            }
            for row in result["years"]
        ],
        "by_exact_riichi_discard_number": [
            {
                "riichi_discard_number": row["riichi_discard_number"],
                "record_count": row["record_count"],
                "all": _compact_category(row["all"]),
                "non_dora_suited_rank_breakdown": _compact_rank_breakdown(row),
                "categories": {
                    category: _compact_category(_categories(row)[category])
                    for category in CATEGORIES
                },
            }
            for row in result["by_turn"]
        ],
        "turn_bands": bands,
        "turn_band_standardized_strong_wait_rates": {
            "method": "common all-record exact-turn distribution within each band",
            "rates": standardized,
        },
        "turn_band_comparisons": comparisons,
        "validation": result["metadata"]["validation"],
        "interpretation_limits": [
            "形式待ち枚数であり、河・他家・実際の山残りは控除しない。",
            "和了率や放銃率ではなく、宣言牌と待ちの構造上の関連である。",
            "same_suji_lineは個別牌の安全度や意図的な筋ひっかけを示さない。",
            "槓ドラはドラ切り分類に含めない。",
        ],
    }


def _percent(rate: float | None) -> str:
    return "—" if rate is None else f"{rate * 100:.2f}%"


def _pp(value: float) -> str:
    return f"{value:+.2f}pt"


def _title(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((70, 45), title, font=_font(46, bold=True), fill=INK)
    draw.text((70, 108), subtitle, font=_font(22), fill=MUTED)


def _footer(draw: ImageDraw.ImageDraw) -> None:
    draw.text(
        (70, 950),
        "2020–2025・鳳凰卓東南戦・東場・赤あり・成立リーチ4,936,197件",
        font=_font(19),
        fill=MUTED,
    )


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _category_bar_chart(
    summary: dict[str, Any], path: Path, *, metric: str, title: str
) -> None:
    image = Image.new("RGB", (1600, 1000), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _title(draw, title, "宣言牌分類はドラを優先する排他的分類")
    rows = summary["overall"]["categories"]
    values = [rows[category][metric]["rate"] for category in CATEGORIES]
    low = max(0.0, min(values) - 0.04)
    high = min(1.0, max(values) + 0.035)
    left, right = 330, 1480
    for tick in range(6):
        rate = low + (high - low) * tick / 5
        x = left + (right - left) * tick / 5
        draw.line((x, 180, x, 880), fill=GRID, width=2)
        draw.text(
            (x, 895),
            f"{rate * 100:.0f}%",
            font=_font(18),
            fill=MUTED,
            anchor="ma",
        )
    for index, category in enumerate(CATEGORIES):
        y = 205 + index * 94
        rate = rows[category][metric]["rate"]
        end = left + int((right - left) * (rate - low) / (high - low))
        draw.text(
            (70, y + 27),
            LABELS[category],
            font=_font(27, bold=True),
            fill=INK,
            anchor="lm",
        )
        draw.rounded_rectangle((left, y, right, y + 54), radius=18, fill="#E5E7EB")
        draw.rounded_rectangle((left, y, end, y + 54), radius=18, fill=COLORS[category])
        draw.text(
            (min(end + 14, right - 5), y + 27),
            _percent(rate),
            font=_font(22, bold=True),
            fill=INK,
            anchor="lm" if end < right - 100 else "rm",
        )
    _footer(draw)
    _save(image, path)


def _line_chart(
    summary: dict[str, Any],
    path: Path,
    *,
    categories: tuple[str, ...],
    title: str,
    subtitle: str,
) -> None:
    image = Image.new("RGB", (1600, 1000), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _title(draw, title, subtitle)
    left, right, top, bottom = 145, 1490, 200, 850
    rows = {
        row["riichi_discard_number"]: row
        for row in summary["by_exact_riichi_discard_number"]
    }
    turns = [turn for turn in sorted(rows) if turn <= 18]
    values = [
        rows[turn]["categories"][category]["strong_wait"]["rate"]
        for turn in turns
        for category in categories
        if rows[turn]["categories"][category]["strong_wait"]["rate"] is not None
    ]
    low = max(0.0, min(values) - 0.035)
    high = min(1.0, max(values) + 0.035)
    for tick in range(6):
        rate = low + (high - low) * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text(
            (left - 15, y),
            f"{rate * 100:.0f}%",
            font=_font(18),
            fill=MUTED,
            anchor="rm",
        )
    for turn in turns:
        x = left + (right - left) * (turn - turns[0]) / (turns[-1] - turns[0])
        draw.text(
            (x, bottom + 25),
            str(turn),
            font=_font(17),
            fill=MUTED,
            anchor="ma",
        )
    for category in categories:
        points = []
        for turn in turns:
            rate = rows[turn]["categories"][category]["strong_wait"]["rate"]
            if rate is None:
                continue
            x = left + (right - left) * (turn - turns[0]) / (turns[-1] - turns[0])
            y = bottom - (bottom - top) * (rate - low) / (high - low)
            points.append((x, y))
        draw.line(points, fill=COLORS[category], width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLORS[category])
    legend_x = 190
    for category in categories:
        draw.line(
            (legend_x, 925, legend_x + 45, 925),
            fill=COLORS[category],
            width=6,
        )
        draw.text(
            (legend_x + 55, 925),
            LABELS[category],
            font=_font(19),
            fill=INK,
            anchor="lm",
        )
        legend_x += 215
    _save(image, path)


def _same_suji_chart(summary: dict[str, Any], path: Path) -> None:
    image = Image.new("RGB", (1600, 1000), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _title(
        draw,
        "宣言牌と同じ筋列に待ち牌を含む頻度",
        "頻度と、その該当record内の弱形率は分母が異なる",
    )
    rows = summary["overall"]["categories"]
    left, right, top, bottom = 150, 1490, 220, 820
    group_width = (right - left) / len(SUITED_CATEGORIES)
    max_rate = min(
        1.0,
        max(
            rows[category]["weak_wait_rate_within_same_suji_line"]["rate"]
            for category in SUITED_CATEGORIES
        )
        + 0.08,
    )
    for tick in range(6):
        rate = max_rate * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text(
            (left - 15, y),
            f"{rate * 100:.0f}%",
            font=_font(18),
            fill=MUTED,
            anchor="rm",
        )
    for index, category in enumerate(SUITED_CATEGORIES):
        center = left + group_width * (index + 0.5)
        frequency = rows[category]["same_suji_line_wait"]["rate"]
        weak_rate = rows[category]["weak_wait_rate_within_same_suji_line"]["rate"]
        for offset, rate, color in (
            (-48, frequency, "#2F6FED"),
            (48, weak_rate, "#E87932"),
        ):
            y = bottom - (bottom - top) * rate / max_rate
            draw.rounded_rectangle(
                (center + offset - 34, y, center + offset + 34, bottom),
                radius=12,
                fill=color,
            )
            draw.text(
                (center + offset, y - 12),
                _percent(rate),
                font=_font(18, bold=True),
                fill=INK,
                anchor="mb",
            )
        draw.text(
            (center, bottom + 28),
            LABELS[category],
            font=_font(23, bold=True),
            fill=INK,
            anchor="ma",
        )
    draw.rounded_rectangle((390, 895, 430, 935), radius=8, fill="#2F6FED")
    draw.text(
        (445, 915),
        "全record中の該当頻度",
        font=_font(20),
        fill=INK,
        anchor="lm",
    )
    draw.rounded_rectangle((900, 895, 940, 935), radius=8, fill="#E87932")
    draw.text(
        (955, 915),
        "該当record内の弱形率",
        font=_font(20),
        fill=INK,
        anchor="lm",
    )
    _save(image, path)


def _rank_same_suji_chart(summary: dict[str, Any], path: Path) -> None:
    image = Image.new("RGB", (1600, 1000), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _title(
        draw,
        "非ドラ数牌は、宣言牌の実数字で差がある",
        "同じ筋列に待ち牌を含む頻度と、該当record内の弱形率",
    )
    rows = {
        row["declaration_rank"]: row
        for row in summary["overall"]["non_dora_suited_rank_breakdown"]["ranks"]
    }
    left, right, top, bottom = 150, 1490, 220, 820
    group_width = (right - left) / 9
    for tick in range(6):
        rate = tick / 5
        y = bottom - (bottom - top) * rate
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text(
            (left - 15, y),
            f"{rate * 100:.0f}%",
            font=_font(18),
            fill=MUTED,
            anchor="rm",
        )
    for index, rank in enumerate(range(1, 10)):
        center = left + group_width * (index + 0.5)
        frequency = rows[rank]["same_suji_line_wait"]["rate"]
        weak_rate = rows[rank]["weak_wait_rate_within_same_suji_line"]["rate"]
        for offset, rate, color in (
            (-27, frequency, "#2F6FED"),
            (27, weak_rate, "#E87932"),
        ):
            y = bottom - (bottom - top) * rate
            draw.rounded_rectangle(
                (center + offset - 20, y, center + offset + 20, bottom),
                radius=9,
                fill=color,
            )
            draw.text(
                (center + offset, y - 9),
                _percent(rate),
                font=_font(14, bold=True),
                fill=INK,
                anchor="mb",
            )
        draw.text(
            (center, bottom + 27),
            str(rank),
            font=_font(24, bold=True),
            fill=INK,
            anchor="ma",
        )
    draw.rounded_rectangle((390, 895, 430, 935), radius=8, fill="#2F6FED")
    draw.text(
        (445, 915),
        "宣言rank内の該当頻度",
        font=_font(20),
        fill=INK,
        anchor="lm",
    )
    draw.rounded_rectangle((900, 895, 940, 935), radius=8, fill="#E87932")
    draw.text(
        (955, 915),
        "該当record内の弱形率",
        font=_font(20),
        fill=INK,
        anchor="lm",
    )
    _save(image, path)


def _mixed_color(base: tuple[int, int, int], intensity: float) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, 0.12 + 0.78 * intensity))
    return tuple(round(255 + (component - 255) * ratio) for component in base)


def _rank_wait_pair_chart(summary: dict[str, Any], path: Path) -> None:
    image = Image.new("RGB", (1800, 1200), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _title(
        draw,
        "宣言牌rank × 同じ筋列の待ちrank",
        "左は宣言rank内の頻度、右は該当record内の弱形率（単位：%）",
    )
    pair_rows = {
        (row["declaration_rank"], row["wait_rank"]): row
        for row in summary["overall"]["non_dora_suited_rank_breakdown"][
            "rank_wait_pairs"
        ]["pairs"]
    }
    max_share = max(
        row["share_within_declaration_rank"]["rate"] for row in pair_rows.values()
    )
    panels = (
        (
            120,
            "宣言rank内の該当頻度",
            "share_within_declaration_rank",
            (47, 111, 237),
            max_share,
        ),
        (1030, "該当record内の弱形率", "weak_wait", (232, 121, 50), 1.0),
    )
    cell = 68
    top = 275
    for left, panel_title, metric_name, base_color, scale in panels:
        draw.text(
            (left + cell * 4.5, 190),
            panel_title,
            font=_font(27, bold=True),
            fill=INK,
            anchor="ma",
        )
        draw.text(
            (left + cell * 4.5, 218),
            "待ち牌rank",
            font=_font(17),
            fill=MUTED,
            anchor="ma",
        )
        for wait_rank in range(1, 10):
            draw.text(
                (left + cell * (wait_rank - 0.5), top - 18),
                str(wait_rank),
                font=_font(18, bold=True),
                fill=INK,
                anchor="mm",
            )
        for declaration_rank in range(1, 10):
            y = top + cell * (declaration_rank - 1)
            draw.text(
                (left - 25, y + cell / 2),
                str(declaration_rank),
                font=_font(18, bold=True),
                fill=INK,
                anchor="mm",
            )
            for wait_rank in range(1, 10):
                x = left + cell * (wait_rank - 1)
                pair = pair_rows.get((declaration_rank, wait_rank))
                if pair is None:
                    fill = "#E5E7EB"
                    label = "—"
                    text_color = MUTED
                else:
                    rate = pair[metric_name]["rate"]
                    fill = _mixed_color(base_color, rate / scale if scale else 0.0)
                    label = f"{rate * 100:.1f}"
                    text_color = INK
                draw.rounded_rectangle(
                    (x + 2, y + 2, x + cell - 2, y + cell - 2),
                    radius=8,
                    fill=fill,
                )
                draw.text(
                    (x + cell / 2, y + cell / 2),
                    label,
                    font=_font(15, bold=pair is not None),
                    fill=text_color,
                    anchor="mm",
                )
        draw.text(
            (left, top + cell * 9 + 8),
            "縦：宣言牌rank",
            font=_font(17),
            fill=MUTED,
            anchor="la",
        )
    draw.text(
        (120, 950),
        "1recordが2種類の対象待ちを含む場合は両pairへ計上（非排他的）。件数・形・待ち枚数分布はJSONに収録。",
        font=_font(20),
        fill=MUTED,
    )
    draw.text(
        (120, 1135),
        "2020–2025・鳳凰卓東南戦・東場・赤あり・非ドラ数牌の成立リーチ",
        font=_font(19),
        fill=MUTED,
    )
    _save(image, path)


def _write_markdown(summary: dict[str, Any], output_dir: Path) -> None:
    overall = summary["overall"]
    rows = overall["categories"]
    rank_breakdown = overall["non_dora_suited_rank_breakdown"]
    rank_rows = {row["declaration_rank"]: row for row in rank_breakdown["ranks"]}
    pair_rows = rank_breakdown["rank_wait_pairs"]["pairs"]
    comparisons = summary["turn_band_comparisons"]
    table = "\n".join(
        f"| {LABELS[category]} | {rows[category]['record_count']:,} | "
        f"{_percent(rows[category]['strong_wait']['rate'])} | "
        f"{_percent(rows[category]['weak_wait']['rate'])} | "
        f"{_percent(rows[category]['other_wait']['rate'])} |"
        for category in CATEGORIES
    )
    suji_table = "\n".join(
        f"| {LABELS[category]} | "
        f"{rows[category]['same_suji_line_wait']['count']:,} "
        f"({_percent(rows[category]['same_suji_line_wait']['rate'])}) | "
        f"{rows[category]['weak_and_same_suji_line']['count']:,} "
        f"({_percent(rows[category]['weak_and_same_suji_line']['rate'])}) | "
        f"{_percent(rows[category]['weak_wait_rate_within_same_suji_line']['rate'])} | "
        f"{_percent(rows[category]['weak_wait_rate_excluding_same_suji_line']['rate'])} |"
        for category in SUITED_CATEGORIES
    )
    rank_table = "\n".join(
        f"| {rank} | {rank_rows[rank]['record_count']:,} | "
        f"{rank_rows[rank]['same_suji_line_wait']['count']:,} "
        f"({_percent(rank_rows[rank]['same_suji_line_wait']['rate'])}) | "
        f"{_percent(rank_rows[rank]['weak_wait_rate_within_same_suji_line']['rate'])} | "
        f"{_percent(rank_rows[rank]['weak_wait_rate_excluding_same_suji_line']['rate'])} | "
        f"{_percent(rank_rows[rank]['strong_wait']['rate'])} | "
        f"{_percent(rank_rows[rank]['weak_wait']['rate'])} |"
        for rank in range(1, 10)
    )
    pair_table = "\n".join(
        f"| {row['declaration_rank']}→{row['wait_rank']} | "
        f"{row['record_count']:,} | "
        f"{_percent(row['share_within_declaration_rank']['rate'])} | "
        f"{_percent(row['weak_wait']['rate'])} | "
        f"{row['strong_wait']['count']:,} |"
        for row in pair_rows
    )
    first_year = summary["by_year"][0]
    last_year = summary["by_year"][-1]
    handoff = f"""# 立直宣言牌別の待ち強さ分析：Work引き渡し

## 正式結果

- 対象: 2020–2025、鳳凰卓東南戦、赤あり、東場のみ
- 観測単位: 成立リーチ1件
- 全record: {overall["record_count"]:,}件
- 強形: {overall["all"]["strong_wait"]["count"]:,}件（{_percent(overall["all"]["strong_wait"]["rate"])}）
- 弱形: {overall["all"]["weak_wait"]["count"]:,}件（{_percent(overall["all"]["weak_wait"]["rate"])}）
- その他: {overall["all"]["other_wait"]["count"]:,}件（{_percent(overall["all"]["other_wait"]["rate"])}）

| 宣言牌 | 件数 | 強形率 | 弱形率 | その他率 |
|---|---:|---:|---:|---:|
{table}

## exact打牌番号の主要結果

各帯域内の全record exact打牌番号分布を共通重みとして標準化した比較。

- 4–9打目: 19切り−5切り = {_pp(comparisons["turns_4_9_19_minus_5_pp"])}
- 10–15打目: 5切り−19切り = {_pp(comparisons["turns_10_15_5_minus_19_pp"])}
- 1–3打目: ドラ切り−字牌切り = {_pp(comparisons["turns_1_3_dora_minus_honor_pp"])}
- 7–9打目: 字牌切り−ドラ切り = {_pp(comparisons["turns_7_9_honor_minus_dora_pp"])}

全体の強形率は2020年の{_percent(first_year["all"]["strong_wait"]["rate"])}から
2025年の{_percent(last_year["all"]["strong_wait"]["rate"])}へ緩やかに上昇した。

## 定義修正

旧版は辺張4枚待ちをその他へ入れていた。新版は単騎・嵌張・辺張・双碰の
4枚以下を弱形とする。強形を先に判定するため、辺張解釈を含んでも5枚以上なら
強形である。旧版から強形件数・率は変わらない。

## same_suji_line

`same_suji_line_wait`は、宣言牌と同色で数字差3または6の待ち牌を含むrecord。
意図的な筋ひっかけ、個別牌の安全度、放銃率を意味しない。関連する形別件数は
複合解釈を重複計上する非排他的フラグである。

全体では{overall["all"]["same_suji_line_wait"]["count"]:,}件
（{_percent(overall["all"]["same_suji_line_wait"]["rate"])}）。そのうち弱形は
{overall["all"]["weak_and_same_suji_line"]["count"]:,}件で、該当record内の
弱形率は{_percent(overall["all"]["weak_wait_rate_within_same_suji_line"]["rate"])}。
該当recordを除いた母集団の弱形率は
{_percent(overall["all"]["weak_wait_rate_excluding_same_suji_line"]["rate"])}である。

| 宣言牌 | 同じ筋列を含む件数（全record比） | 弱形かつ該当（全record比） | 該当内の弱形率 | 該当除外後の弱形率 |
|---|---:|---:|---:|---:|
{suji_table}

46切り・5切りでは、該当record内の弱形率が約95%に達する一方、該当頻度自体は
6〜8%である。該当recordを除外すると弱形率は46切り
{_percent(rows["46"]["weak_wait_rate_excluding_same_suji_line"]["rate"])}、5切り
{_percent(rows["5"]["weak_wait_rate_excluding_same_suji_line"]["rate"])}まで下がる。
これは関連を示す集計であり、打ち手の意図や因果効果の証明ではない。

## 非ドラ数牌の実rank

主表はドラを除く数牌だけを対象とし、各recordを宣言牌rankへ1回だけ計上する。

| 宣言rank | 件数 | 同じ筋列を含む件数（率） | 該当内の弱形率 | 該当除外後の弱形率 | 強形率 | 弱形率 |
|---:|---:|---:|---:|---:|---:|---:|
{rank_table}

rank 1+9、2+8、3+7、4+6、rank 5は、それぞれ19・28・37・46・5分類と
整数count・全内訳が完全一致する。

rank 4・5・6では、同じ筋列を含むrecordの弱形率が94.60〜95.20%と突出する。
各年でも94.18〜95.59%の範囲にあり、exact打牌番号を1〜3、4〜9、10〜15、
16以降にまとめても順に96.33〜97.81%、95.06〜95.82%、93.38〜93.94%、
89.72〜90.17%だった。

## 宣言rank × 待ちrank

| pair | 該当件数 | 宣言rank内の率 | 該当内の弱形率 | 強形件数 |
|---|---:|---:|---:|---:|
{pair_table}

1recordが2種類の対象待ちを含む場合は両pairへ計上するため、この表は非排他的。
弱形の嵌張・辺張・双碰・単騎フラグと形式待ち枚数分布はresult-summary.jsonに
収録している。

4→7・5→2・5→8・6→3は弱形率94.81〜95.35%。一方、3→9・7→1は
約20.5%で、同じ数字差6でも組み合わせによる差が大きい。

## 記事で必ず守ること

1. 形式待ち枚数であり、河・他家・実際の山残りは控除しない。
2. 和了率や放銃率ではなく、宣言牌と待ちの関連である。
3. same_suji_lineを一律に「筋ひっかけ」と呼ばない。
4. 打ち手が意図的に筋ひっかけを選んだと断定しない。
5. ドラは赤5または局開始時の表ドラで、槓ドラは含めない。

## ファイル

- 詳細結果: `research/results/riichi-declaration-tile-analysis-v1.json`
- コンパクト結果: `outputs/work/riichi-declaration-tile-analysis/result-summary.json`
- 図: `outputs/work/riichi-declaration-tile-analysis/figures/`
"""
    (output_dir / "handoff.md").write_text(handoff, encoding="utf-8", newline="\n")

    brief = f"""# 記事ブリーフ

## 中心テーマ

立直宣言牌から待ちの強さをどこまで読めるか。全体では字牌・ドラ切りの強形率が
高い一方、数牌は宣言打牌番号によって19切りと5切りの関係が逆転する。
さらに、46切り・5切りで「宣言牌と同じ筋列に待ち牌を含む」recordは全体の
6〜8%だが、その約95%が弱形。これを除くと両分類の弱形率はそれぞれ
{_percent(rows["46"]["weak_wait_rate_excluding_same_suji_line"]["rate"])}と
{_percent(rows["5"]["weak_wait_rate_excluding_same_suji_line"]["rate"])}まで下がる。
実数字1〜9と、宣言rank×待ちrankの18組まで分解し、どの数字の組み合わせが
この差を構成するかを確認する。

## 記事構成案

1. 宣言牌で待ちの強さは変わるのか
2. 強形・弱形の定義と辺張修正
3. 宣言牌7分類の全体結果
4. exact打牌番号で見える19切りと5切りの逆転
5. 字牌切りとドラ切りの序盤・中盤の逆転
6. 宣言牌と同じ筋列に待ち牌を含むrecord
7. 非ドラ数牌の実rank 1〜9
8. 宣言rank×待ちrankの18組
9. 言えること／言えないこと

## 旧記事案から必ず直す点

- 辺張4枚待ちはその他ではなく弱形。
- 旧版の弱形率・その他率は使わない。
- 強形率は不変なので、強形に関する主要結論は維持する。
- 「筋ひっかけ」は意図を含むため、原則「宣言牌と同じ筋列に待ち牌を含む」と書く。
- 46切り・5切りの特徴は筋列該当recordと強く関連するが、意図や因果とは書かない。
- rank主表は排他的、rank×wait rank表は非排他的であることを明記する。
"""
    (output_dir / "article-brief.md").write_text(brief, encoding="utf-8", newline="\n")

    prompt = """このフォルダのhandoff.md、article-brief.md、result-summary.jsonとfigures/を読み、note向けの日本語記事を作ってください。数値はresult-summary.jsonを正本とし、旧版の弱形率・その他率は使わないでください。中心は、宣言牌別の強形率、辺張を含めた正しい弱形率、exact打牌番号で起きる19切り対5切り・字牌切り対ドラ切りの逆転、非ドラ数牌の実rank 1〜9、宣言rank×待ちrankの18組です。rank主表は排他的、rank×wait rank表は1recordが2組へ入ることがある非排他的集計として説明してください。same_suji_line_waitは原則『宣言牌と同じ筋列に待ち牌を含む』と表現し、意図的な筋ひっかけ、個別牌の安全度、放銃率、因果効果を主張しないでください。図を適切な位置へ挿入し、定義・結果・打牌番号別確認・限界を含むレビュー用初稿を返してください。公開や外部送信はしないでください。"""
    (output_dir / "work-prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")


def build_handoff(result_path: Path, output_dir: Path) -> None:
    result = _load_result(result_path)
    summary = _build_summary(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_markdown(summary, output_dir)
    figures = output_dir / "figures"
    _category_bar_chart(
        summary,
        figures / "01-strong-rate-by-declaration.png",
        metric="strong_wait",
        title="立直宣言牌別の強形率",
    )
    _line_chart(
        summary,
        figures / "02-suited-strong-rate-by-exact-turn.png",
        categories=SUITED_CATEGORIES,
        title="数牌宣言別の強形率は、打牌番号で動く",
        subtitle="宣言者本人のexact打牌番号・1〜18打目（全数値はJSONに収録）",
    )
    _line_chart(
        summary,
        figures / "03-honor-dora-strong-rate-by-exact-turn.png",
        categories=("honor", "dora"),
        title="字牌切りとドラ切りも、序盤と中盤で関係が変わる",
        subtitle="宣言者本人のexact打牌番号・1〜18打目（全数値はJSONに収録）",
    )
    _category_bar_chart(
        summary,
        figures / "04-weak-rate-by-declaration.png",
        metric="weak_wait",
        title="辺張を含めた、立直宣言牌別の弱形率",
    )
    _same_suji_chart(
        summary,
        figures / "05-same-suji-line-frequency-and-weak-rate.png",
    )
    _rank_same_suji_chart(
        summary,
        figures / "06-exact-rank-same-suji-line-and-weak-rate.png",
    )
    _rank_wait_pair_chart(
        summary,
        figures / "07-rank-wait-rank-pair-heatmaps.png",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    build_handoff(args.result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
