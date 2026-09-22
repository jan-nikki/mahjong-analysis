"""Build compact Work handoff materials for the toitsu-drop analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "research" / "results" / "toitsu-drop-analysis-v1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "work" / "toitsu-drop-analysis"

BACKGROUND = "#F7F5EF"
PANEL = "#FFFFFF"
INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"
TEAL = "#14857B"
ORANGE = "#E87932"
BLUE = "#2F6FED"
PURPLE = "#7F56D9"
RED = "#D92D20"

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
    if result.get("metadata", {}).get("analysis_name") != "toitsu-drop-analysis-v1":
        raise ValueError("unexpected analysis result")
    return result


def _percent(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def _pp(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}pt"


def _quality(group: dict[str, Any], metric: str) -> dict[str, Any]:
    value = group[metric]
    return {
        "count": value["count"],
        "denominator": value["denominator"],
        "rate": value["rate"],
    }


def _compact_slice(value: dict[str, Any]) -> dict[str, Any]:
    groups = value["groups"]
    comparisons = value["pattern_minus_no_pattern_percentage_points"]
    return {
        "record_count": value["record_count"],
        "pattern_present": value["pattern_present"],
        "classification_counts": value["classification_counts"],
        "pattern_present_quality": {
            metric: _quality(groups["pattern_present"], metric)
            for metric in ("contains_ryanmen", "contains_suji", "good_wait")
        },
        "no_pattern_quality": {
            metric: _quality(groups["no_pattern"], metric)
            for metric in ("contains_ryanmen", "contains_suji", "good_wait")
        },
        "pattern_minus_no_pattern_percentage_points": {
            metric: comparisons[metric]
            for metric in ("contains_ryanmen", "contains_suji", "good_wait")
        },
    }


def _build_summary(result: dict[str, Any]) -> dict[str, Any]:
    overall = result["overall"]
    pattern_count = overall["pattern_present"]["count"]
    classifications = overall["classification_counts"]
    return {
        "analysis_name": result["metadata"]["analysis_name"],
        "scope": result["metadata"]["scope"],
        "input_manifest_sha256": result["metadata"]["input_manifest_sha256"],
        "definitions": result["metadata"]["definitions"],
        "full_period": _compact_slice(overall),
        "recent_2020_2025": _compact_slice(result["recent_2020_2025"]["overall"]),
        "pattern_composition": {
            category: {
                "count": classifications[category],
                "denominator": pattern_count,
                "share": classifications[category] / pattern_count,
            }
            for category in ("a_only", "b_only", "both")
        },
        "by_year": [
            {
                "year": item["year"],
                **_compact_slice(item["overall"]),
            }
            for item in result["years"]
        ],
        "by_exact_riichi_discard_number": [
            {
                "riichi_discard_number": item["riichi_discard_number"],
                **_compact_slice(item),
            }
            for item in result["by_turn"]
        ],
        "by_representative_pair_distance": [
            {
                "distance": item["distance"],
                "record_count": item["record_count"],
                "contains_ryanmen_rate": item["contains_ryanmen"]["rate"],
                "contains_suji_rate": item["contains_suji"]["rate"],
                "good_wait_rate": item["good_wait"]["rate"],
            }
            for item in result["by_distance"]
        ],
        "audit": {
            "selected_count": result["audit"]["selected_count"],
            "missing_reasons": result["audit"]["missing_reasons"],
            "raw_event_recheck": "20/20 matched",
        },
        "interpretation_limits": [
            "『対子落としに見える』は河の外形による分類で、意図や手牌内の対子を証明しない。",
            "同じ宣言打牌番号で比較しても因果効果や押し引きの最適解は示さない。",
            "筋待ち率の上昇は、個別の筋牌が必ず安全であることを意味しない。",
            "好形は宣言者自身の手牌・副露だけで数えた形式待ちの残り枚数が5枚以上という定義。",
        ],
    }


def _draw_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
) -> None:
    draw.text((70, 48), title, font=_font(48, bold=True), fill=INK)
    draw.text((70, 112), subtitle, font=_font(23), fill=MUTED)


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _make_headline_chart(summary: dict[str, Any], path: Path) -> None:
    image = Image.new("RGB", (1600, 930), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "対子落としに見える河の後は、待ちが良いのか",
        "2009–2025・鳳凰卓東南戦の東場・成立リーチ 10,706,714件",
    )
    pattern = summary["full_period"]["pattern_present_quality"]
    no_pattern = summary["full_period"]["no_pattern_quality"]
    metrics = (
        ("筋待ちを含む", "contains_suji", TEAL),
        ("5枚以上の好形", "good_wait", ORANGE),
        ("両面を含む", "contains_ryanmen", BLUE),
    )
    left = 520
    width = 900
    scale_max = 0.72
    for index, (label, key, color) in enumerate(metrics):
        top = 220 + index * 205
        draw.text((70, top + 40), label, font=_font(31, bold=True), fill=INK)
        for row, (group_label, value, fill) in enumerate(
            (
                ("対子落とし型あり", pattern[key]["rate"], color),
                ("パターンなし", no_pattern[key]["rate"], "#AAB2C0"),
            )
        ):
            y = top + row * 72
            draw.text((260, y + 20), group_label, font=_font(21), fill=MUTED)
            draw.rounded_rectangle(
                (left, y, left + width, y + 48),
                radius=18,
                fill="#E5E7EB",
            )
            end = left + int(width * value / scale_max)
            draw.rounded_rectangle((left, y, end, y + 48), radius=18, fill=fill)
            draw.text(
                (end - 12, y + 24),
                _percent(value),
                font=_font(22, bold=True),
                fill="white",
                anchor="rm",
            )
    differences = summary["full_period"]["pattern_minus_no_pattern_percentage_points"]
    draw.text(
        (70, 850),
        "差：筋待ち "
        f"{_pp(differences['contains_suji'])}／好形 {_pp(differences['good_wait'])}／"
        f"両面 {_pp(differences['contains_ryanmen'])}",
        font=_font(26, bold=True),
        fill=PURPLE,
    )
    _save(image, path)


def _plot_lines(
    *,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    labels: list[str],
    series: tuple[tuple[str, list[float | None], str], ...],
    y_min: float,
    y_max: float,
    x0: int = 130,
    y0: int = 200,
    x1: int = 1510,
    y1: int = 800,
) -> None:
    del image
    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5
        y = y1 - (y1 - y0) * step / 5
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        decimals = 1 if y_max - y_min < 5 else 0
        draw.text(
            (x0 - 20, y),
            f"{value:.{decimals}f}pt",
            font=_font(18),
            fill=MUTED,
            anchor="rm",
        )
    if y_min < 0 < y_max:
        zero_y = y1 - (y1 - y0) * (0 - y_min) / (y_max - y_min)
        draw.line((x0, zero_y, x1, zero_y), fill="#98A2B3", width=3)
    count = len(labels)
    points_x = [x0 + (x1 - x0) * index / max(count - 1, 1) for index in range(count)]
    for index, label in enumerate(labels):
        if count <= 18 or index % 2 == 0:
            draw.text(
                (points_x[index], y1 + 26),
                label,
                font=_font(18),
                fill=MUTED,
                anchor="ma",
            )
    for legend_index, (name, values, color) in enumerate(series):
        points = []
        for x, value in zip(points_x, values, strict=True):
            if value is None:
                continue
            y = y1 - (y1 - y0) * (value - y_min) / (y_max - y_min)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        legend_x = 700 + legend_index * 290
        draw.line((legend_x, 155, legend_x + 55, 155), fill=color, width=6)
        draw.text(
            (legend_x + 70, 155), name, font=_font(22, bold=True), fill=INK, anchor="lm"
        )


def _make_turn_chart(summary: dict[str, Any], path: Path) -> None:
    rows = [
        item
        for item in summary["by_exact_riichi_discard_number"]
        if 2 <= item["riichi_discard_number"] <= 18
    ]
    image = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "同じ宣言打牌番号で比べても差は残る",
        "対子落とし型あり − パターンなし（percentage points）",
    )
    _plot_lines(
        image=image,
        draw=draw,
        labels=[str(item["riichi_discard_number"]) for item in rows],
        series=(
            (
                "筋待ち",
                [
                    item["pattern_minus_no_pattern_percentage_points"]["contains_suji"]
                    for item in rows
                ],
                TEAL,
            ),
            (
                "好形",
                [
                    item["pattern_minus_no_pattern_percentage_points"]["good_wait"]
                    for item in rows
                ],
                ORANGE,
            ),
            (
                "両面",
                [
                    item["pattern_minus_no_pattern_percentage_points"][
                        "contains_ryanmen"
                    ]
                    for item in rows
                ],
                BLUE,
            ),
        ),
        y_min=-2,
        y_max=18,
    )
    draw.text((800, 870), "リーチ宣言打牌番号", font=_font(22), fill=MUTED, anchor="mm")
    _save(image, path)


def _make_year_chart(summary: dict[str, Any], path: Path) -> None:
    rows = summary["by_year"]
    image = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "年度別でも同じ方向",
        "対子落とし型あり − パターンなし（percentage points）",
    )
    _plot_lines(
        image=image,
        draw=draw,
        labels=[str(item["year"]) for item in rows],
        series=(
            (
                "筋待ち",
                [
                    item["pattern_minus_no_pattern_percentage_points"]["contains_suji"]
                    for item in rows
                ],
                TEAL,
            ),
            (
                "好形",
                [
                    item["pattern_minus_no_pattern_percentage_points"]["good_wait"]
                    for item in rows
                ],
                ORANGE,
            ),
        ),
        y_min=4,
        y_max=7,
    )
    _save(image, path)


def _make_composition_chart(summary: dict[str, Any], path: Path) -> None:
    image = Image.new("RGB", (1600, 850), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "『対子落としに見える河』は全リーチの約1割",
        "A＝手出し→手出し、B＝ツモ切り→手出し",
    )
    full = summary["full_period"]
    pattern_rate = full["pattern_present"]["rate"]
    draw.text(
        (110, 245), _percent(pattern_rate), font=_font(96, bold=True), fill=PURPLE
    )
    draw.text((115, 360), "パターンあり", font=_font(30, bold=True), fill=INK)
    draw.text(
        (115, 410),
        f"{full['pattern_present']['count']:,} / {full['record_count']:,}件",
        font=_font(24),
        fill=MUTED,
    )
    composition = summary["pattern_composition"]
    rows = (
        ("Aのみ", composition["a_only"], TEAL),
        ("Bのみ", composition["b_only"], ORANGE),
        ("A・B両方", composition["both"], BLUE),
    )
    left = 650
    for index, (label, value, color) in enumerate(rows):
        y = 220 + index * 150
        draw.text((left, y), label, font=_font(30, bold=True), fill=INK)
        draw.text(
            (left + 250, y),
            f"{value['count']:,}件",
            font=_font(30, bold=True),
            fill=color,
        )
        draw.text(
            (left + 570, y),
            f"パターンありの {_percent(value['share'])}",
            font=_font(24),
            fill=MUTED,
        )
    draw.text(
        (110, 735),
        "注意：河の並びによる外形分類であり、実際に手牌の対子を意図して落とした証明ではありません。",
        font=_font(22),
        fill=RED,
    )
    _save(image, path)


def _write_markdown(summary: dict[str, Any], output_dir: Path) -> None:
    full = summary["full_period"]
    recent = summary["recent_2020_2025"]
    pattern = full["pattern_present_quality"]
    no_pattern = full["no_pattern_quality"]
    differences = full["pattern_minus_no_pattern_percentage_points"]
    composition = summary["pattern_composition"]

    handoff = f"""# 対子落としに見える河の後のリーチ：Work引き渡し

## 正式結果

- 対象: 2009–2025、鳳凰卓東南戦、赤あり、東場のみ
- 観測単位: 成立リーチ1件
- 全レコード: {full["record_count"]:,}件
- パターンあり: {full["pattern_present"]["count"]:,}件（{_percent(full["pattern_present"]["rate"])}）
- パターンなし: {full["record_count"] - full["pattern_present"]["count"]:,}件

| 指標 | パターンあり | パターンなし | 差 |
|---|---:|---:|---:|
| 筋待ちを含む | {_percent(pattern["contains_suji"]["rate"])} | {_percent(no_pattern["contains_suji"]["rate"])} | {_pp(differences["contains_suji"])} |
| 5枚以上の好形 | {_percent(pattern["good_wait"]["rate"])} | {_percent(no_pattern["good_wait"]["rate"])} | {_pp(differences["good_wait"])} |
| 両面を含む | {_percent(pattern["contains_ryanmen"]["rate"])} | {_percent(no_pattern["contains_ryanmen"]["rate"])} | {_pp(differences["contains_ryanmen"])} |

2020–2025だけでも、筋待ち差は {_pp(recent["pattern_minus_no_pattern_percentage_points"]["contains_suji"])}、好形差は {_pp(recent["pattern_minus_no_pattern_percentage_points"]["good_wait"])} です。

## A/B内訳

- Aのみ（手出し→手出し）: {composition["a_only"]["count"]:,}件（パターンありの {_percent(composition["a_only"]["share"])}）
- Bのみ（ツモ切り→手出し）: {composition["b_only"]["count"]:,}件（同 {_percent(composition["b_only"]["share"])}）
- A・B両方: {composition["both"]["count"]:,}件（同 {_percent(composition["both"]["share"])}）

## 必ず守る解釈

1. これは河の外形分類です。実際に手牌内の対子を意図的に落とした証明ではありません。
2. 宣言巡目が遅いほどパターンの出現機会も増えるため、全体差だけで因果効果を主張しません。
3. exact宣言打牌番号別でも2～17打目はすべて正方向ですが、18打目以降は件数が小さく不安定です。
4. 『筋待ち率が高い』は、個別の筋牌が必ず通るという意味ではありません。
5. 今回は押し引きの損益、放銃率、追っかけるべきかまでは分析していません。

## 検証

- canonical入力10,706,714件と完全一致
- 全年・全宣言打牌番号の全レコード集計が既存wait-quality baselineと一致
- 年次file size/SHA256検証PASS
- 人工テスト、serial/parallel一致、分割不変条件PASS
- 決定論的監査20件を元MJAIへ戻して再確認し、20/20一致

## ファイル

- 詳細結果: `research/results/toitsu-drop-analysis-v1.json`
- コンパクト結果: `outputs/work/toitsu-drop-analysis/result-summary.json`
- 図: `outputs/work/toitsu-drop-analysis/figures/`
"""
    (output_dir / "handoff.md").write_text(handoff, encoding="utf-8", newline="\n")

    brief = """# 記事ブリーフ

## 仮タイトル

対子落としに見える河の後のリーチは、本当に待ちが良いのか

## 読者へ最初に出す結論

成立リーチ1,070万件で調べると、対子落としに見える河を持つリーチは全体の約9.56%。その待ちは、そうでないリーチより筋待ち率が約6.03ポイント、5枚以上の好形率が約5.91ポイント高かった。同じ宣言打牌番号ごとの比較でも、十分な件数のある範囲では概ね同じ方向だった。

## 記事構成案

1. 対子を2枚続けて切ったように見える河は、なんとなく『手が整っていそう』に感じる
2. 今回の分類（2枚目は手出し、1枚目は手出しでもツモ切りでもよい）
3. 比較する二つの待ち指標（筋待ち、自己所有を除いた5枚以上の好形）
4. 全体結果：67.02%対60.99%、67.58%対61.66%
5. 単なる遅いリーチ効果ではないか：exact宣言打牌番号別の確認
6. AとBの内訳、宣言牌そのものが2枚目になる例
7. 言えること／言えないこと
8. 次回予告：カンチャン落としに見える河

## 表現上の注意

- 『対子落とし』ではなく、原則『対子落としに見える河』と書く。
- 『筋が通る』ではなく『筋が待ちに含まれる割合が高い』と書く。
- 『追っかけない方が得』『降りるべき』とは結論しない。
- 率の差はpercentage points。相対増加率と混同しない。
"""
    (output_dir / "article-brief.md").write_text(brief, encoding="utf-8", newline="\n")

    prompt = """このフォルダの handoff.md、article-brief.md、result-summary.json と figures/ を読み、note向けの記事を作ってください。記事の中心は『対子落としに見える河の後のリーチは、筋待ち率と5枚以上の好形率が約6ポイント高い』です。因果効果、押し引きの最適解、個別の筋牌の安全を主張しないでください。『対子落とし』は手牌意図の証明ではないので、原則『対子落としに見える河』と表現してください。数値はresult-summary.jsonを正本とし、図を適切な位置へ挿入してください。読者に直感的に伝わる平易な日本語で、定義・結果・巡目別確認・限界・次回のカンチャン落とし分析への導線まで含めてください。"""
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
    figure_dir = output_dir / "figures"
    _make_headline_chart(summary, figure_dir / "01-headline-comparison.png")
    _make_turn_chart(summary, figure_dir / "02-exact-turn-difference.png")
    _make_year_chart(summary, figure_dir / "03-yearly-difference.png")
    _make_composition_chart(summary, figure_dir / "04-pattern-composition.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Work handoff materials for the toitsu-drop analysis."
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_handoff(args.result, args.output_dir)


if __name__ == "__main__":
    main()
