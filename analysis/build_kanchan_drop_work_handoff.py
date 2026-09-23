"""Build compact Work handoff materials for the kanchan-drop analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "research/results/kanchan-drop-analysis-v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/work/kanchan-drop-analysis"

BACKGROUND = "#F7F5EF"
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


def _percent(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def _pp(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}pt"


def _metric_group(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_count": group["record_count"],
        **{
            metric: {
                "count": group[metric]["count"],
                "denominator": group[metric]["denominator"],
                "rate": group[metric]["rate"],
            }
            for metric in ("contains_ryanmen", "contains_suji", "good_wait")
        },
    }


def _compact_slice(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_count": value["record_count"],
        "pattern_present": value["pattern_present"],
        "classification_counts": value["classification_counts"],
        "pattern_present_quality": _metric_group(value["groups"]["pattern_present"]),
        "no_pattern_quality": _metric_group(value["groups"]["no_pattern"]),
        "pattern_minus_no_pattern_percentage_points": value[
            "pattern_minus_no_pattern_percentage_points"
        ],
        "representative_order_groups": {
            name: _metric_group(value["representative_order_groups"][name])
            for name in ("inner_first", "outer_first", "symmetric")
        },
        "inner_minus_outer_percentage_points": value[
            "inner_minus_outer_percentage_points"
        ],
        "representative_shape_direction": value["representative_shape_direction"],
        "representative_type_order_groups": value["representative_type_order_groups"],
    }


def _load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("metadata", {}).get("analysis_name") != "kanchan-drop-analysis-v1":
        raise ValueError("unexpected analysis result")
    return result


def _build_summary(result: dict[str, Any]) -> dict[str, Any]:
    recent = result["recent_2020_2025"]
    return {
        "analysis_name": result["metadata"]["analysis_name"],
        "scope": result["metadata"]["scope"],
        "input_manifest_sha256": result["metadata"]["input_manifest_sha256"],
        "definitions": result["metadata"]["definitions"],
        "primary_2020_2025": _compact_slice(recent["overall"]),
        "full_2009_2025": _compact_slice(result["overall"]),
        "primary_without_toitsu_pattern": _compact_slice(
            recent["by_toitsu_pattern"]["absent"]
        ),
        "by_year": [
            {"year": row["year"], **_compact_slice(row["overall"])}
            for row in result["years"]
        ],
        "primary_by_exact_riichi_discard_number": [
            {
                "riichi_discard_number": row["riichi_discard_number"],
                **_compact_slice(row),
            }
            for row in recent["by_turn"]
        ],
        "primary_by_representative_distance": [
            {"distance": row["distance"], **_compact_slice(row)}
            for row in recent["by_distance"]
        ],
        "audit": {
            "selected_count": result["audit"]["selected_count"],
            "missing_reasons": result["audit"]["missing_reasons"],
            "raw_event_recheck": "25/25 matched",
        },
        "interpretation_limits": [
            "『嵌張落としに見える河』は外形分類で、実際の手牌構成や意図を証明しない。",
            "AとBはどちらも14枚から弱いブロックを処理した結果として主群へ含める。",
            "代表候補はリーチに最も近い最後の候補で、切り順比較は成立リーチ1件を一度だけ数える。",
            "筋待ち率は個別の筋牌の安全性、好形率は実際の山残枚数を表さない。",
            "巡目別比較を含めても因果効果や押し引きの最適解は示さない。",
        ],
    }


def _draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((70, 48), title, font=_font(46, bold=True), fill=INK)
    draw.text((70, 110), subtitle, font=_font(22), fill=MUTED)


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _comparison_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    left_label: str,
    right_label: str,
    left: dict[str, Any],
    right: dict[str, Any],
    differences: dict[str, float | None],
) -> None:
    image = Image.new("RGB", (1600, 930), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_title(draw, title, subtitle)
    metrics = (
        ("筋待ちを含む", "contains_suji", TEAL),
        ("5枚以上の好形", "good_wait", ORANGE),
        ("両面を含む", "contains_ryanmen", BLUE),
    )
    bar_left, bar_width, scale_max = 560, 850, 0.75
    for index, (label, key, color) in enumerate(metrics):
        top = 215 + index * 205
        draw.text((70, top + 35), label, font=_font(30, bold=True), fill=INK)
        for row, (group_label, value, fill) in enumerate(
            (
                (left_label, left[key]["rate"], color),
                (right_label, right[key]["rate"], "#AAB2C0"),
            )
        ):
            y = top + row * 72
            draw.text((285, y + 20), group_label, font=_font(20), fill=MUTED)
            draw.rounded_rectangle(
                (bar_left, y, bar_left + bar_width, y + 48),
                radius=18,
                fill="#E5E7EB",
            )
            end = bar_left + int(bar_width * value / scale_max)
            draw.rounded_rectangle((bar_left, y, end, y + 48), radius=18, fill=fill)
            draw.text(
                (end - 12, y + 24),
                _percent(value),
                font=_font(21, bold=True),
                fill="white",
                anchor="rm",
            )
    draw.text(
        (70, 850),
        f"差：筋待ち {_pp(differences['contains_suji'])}／"
        f"好形 {_pp(differences['good_wait'])}／"
        f"両面 {_pp(differences['contains_ryanmen'])}",
        font=_font(25, bold=True),
        fill=PURPLE,
    )
    _save(image, path)


def _shape_chart(summary: dict[str, Any], path: Path) -> None:
    rows = summary["primary_2020_2025"]["representative_shape_direction"]
    image = Image.new("RGB", (1600, 930), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "内側から切る効果は、13・24・68・79形で大きい",
        "2020–2025・代表候補の内側先行 − 外側先行（46のみ高→低 − 低→高）",
    )
    x0, x1, y0, y1 = 155, 1510, 220, 790
    max_value = 7.0
    for step in range(8):
        y = y1 - (y1 - y0) * step / 7
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        draw.text(
            (x0 - 20, y),
            f"{step:.0f}pt",
            font=_font(17),
            fill=MUTED,
            anchor="rm",
        )
    group_width = (x1 - x0) / len(rows)
    for index, row in enumerate(rows):
        center = x0 + group_width * (index + 0.5)
        key = (
            "inner_minus_outer_percentage_points"
            if row["inner_direction"] is not None
            else "high_to_low_minus_low_to_high"
        )
        values = (
            (row[key]["contains_suji"], TEAL, -24),
            (row[key]["good_wait"], ORANGE, 24),
        )
        for value, color, offset in values:
            height = (y1 - y0) * value / max_value
            draw.rounded_rectangle(
                (center + offset - 18, y1 - height, center + offset + 18, y1),
                radius=8,
                fill=color,
            )
        draw.text(
            (center, y1 + 30),
            row["shape"],
            font=_font(23, bold=True),
            fill=INK,
            anchor="ma",
        )
    draw.rectangle((580, 160, 610, 190), fill=TEAL)
    draw.text((625, 175), "筋待ち", font=_font(20), fill=INK, anchor="lm")
    draw.rectangle((790, 160, 820, 190), fill=ORANGE)
    draw.text((835, 175), "好形", font=_font(20), fill=INK, anchor="lm")
    draw.text(
        (800, 880),
        "24形：4→2 は 2→4 より筋待ち +4.21pt、好形 +3.98pt",
        font=_font(25, bold=True),
        fill=PURPLE,
        anchor="mm",
    )
    _save(image, path)


def _line_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    series: tuple[tuple[str, list[float | None], str], ...],
    y_min: float,
    y_max: float,
) -> None:
    image = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_title(draw, title, subtitle)
    x0, y0, x1, y1 = 130, 220, 1510, 790
    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5
        y = y1 - (y1 - y0) * step / 5
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        draw.text(
            (x0 - 20, y), f"{value:.1f}pt", font=_font(17), fill=MUTED, anchor="rm"
        )
    count = len(labels)
    xs = [x0 + (x1 - x0) * index / max(count - 1, 1) for index in range(count)]
    for index, label in enumerate(labels):
        if count <= 18 or index % 2 == 0:
            draw.text(
                (xs[index], y1 + 28), label, font=_font(17), fill=MUTED, anchor="ma"
            )
    for legend_index, (name, values, color) in enumerate(series):
        points = []
        for x, value in zip(xs, values, strict=True):
            if value is None:
                continue
            y = y1 - (y1 - y0) * (value - y_min) / (y_max - y_min)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        legend_x = 700 + legend_index * 260
        draw.line((legend_x, 165, legend_x + 48, 165), fill=color, width=6)
        draw.text(
            (legend_x + 60, 165), name, font=_font(20, bold=True), fill=INK, anchor="lm"
        )
    _save(image, path)


def _composition_chart(summary: dict[str, Any], path: Path) -> None:
    primary = summary["primary_2020_2025"]
    counts = primary["classification_counts"]
    pattern = primary["pattern_present"]["count"]
    image = Image.new("RGB", (1600, 850), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "嵌張落としに見える河は、成立リーチの14.12%",
        "2020–2025・A/Bは同格で主パターンに含める",
    )
    draw.text(
        (110, 245),
        _percent(primary["pattern_present"]["rate"]),
        font=_font(94, bold=True),
        fill=PURPLE,
    )
    draw.text(
        (115, 365),
        f"{pattern:,} / {primary['record_count']:,}件",
        font=_font(26),
        fill=MUTED,
    )
    rows = (
        ("Aのみ", counts["a_only"], TEAL),
        ("Bのみ", counts["b_only"], ORANGE),
        ("A・B両方", counts["both"], BLUE),
    )
    for index, (label, count, color) in enumerate(rows):
        y = 220 + index * 150
        draw.text((650, y), label, font=_font(30, bold=True), fill=INK)
        draw.text((900, y), f"{count:,}件", font=_font(30, bold=True), fill=color)
        draw.text(
            (1210, y),
            f"{count / pattern * 100:.2f}%",
            font=_font(25),
            fill=MUTED,
        )
    draw.text(
        (110, 745),
        "注意：河の外形分類であり、実際の手牌や打牌意図を証明するものではありません。",
        font=_font(22),
        fill=RED,
    )
    _save(image, path)


def _write_markdown(summary: dict[str, Any], output_dir: Path) -> None:
    primary = summary["primary_2020_2025"]
    full = summary["full_2009_2025"]
    pattern = primary["pattern_present_quality"]
    none = primary["no_pattern_quality"]
    pattern_diff = primary["pattern_minus_no_pattern_percentage_points"]
    inner = primary["representative_order_groups"]["inner_first"]
    outer = primary["representative_order_groups"]["outer_first"]
    order_diff = primary["inner_minus_outer_percentage_points"]
    shape24 = next(
        row for row in primary["representative_shape_direction"] if row["shape"] == "24"
    )

    handoff = f"""# 嵌張落としに見える河の後のリーチ：Work引き渡し

## 主結果（2020–2025）

- 対象: 鳳凰卓東南戦、赤あり、東場、成立リーチ {primary["record_count"]:,}件
- パターンあり: {primary["pattern_present"]["count"]:,}件（{_percent(primary["pattern_present"]["rate"])}）
- A/Bは同格。代表候補はリーチに最も近い最後の候補

| 指標 | パターンあり | なし | 差 |
|---|---:|---:|---:|
| 筋待ちを含む | {_percent(pattern["contains_suji"]["rate"])} | {_percent(none["contains_suji"]["rate"])} | {_pp(pattern_diff["contains_suji"])} |
| 5枚以上の好形 | {_percent(pattern["good_wait"]["rate"])} | {_percent(none["good_wait"]["rate"])} | {_pp(pattern_diff["good_wait"])} |
| 両面を含む | {_percent(pattern["contains_ryanmen"]["rate"])} | {_percent(none["contains_ryanmen"]["rate"])} | {_pp(pattern_diff["contains_ryanmen"])} |

## 切り順仮説

| 指標 | 内側先行 | 外側先行 | 差 |
|---|---:|---:|---:|
| 筋待ちを含む | {_percent(inner["contains_suji"]["rate"])} | {_percent(outer["contains_suji"]["rate"])} | {_pp(order_diff["contains_suji"])} |
| 5枚以上の好形 | {_percent(inner["good_wait"]["rate"])} | {_percent(outer["good_wait"]["rate"])} | {_pp(order_diff["good_wait"])} |
| 両面を含む | {_percent(inner["contains_ryanmen"]["rate"])} | {_percent(outer["contains_ryanmen"]["rate"])} | {_pp(order_diff["contains_ryanmen"])} |

24形では4→2が2→4より、筋待ち {_pp(shape24["inner_minus_outer_percentage_points"]["contains_suji"])}、好形 {_pp(shape24["inner_minus_outer_percentage_points"]["good_wait"])}、両面 {_pp(shape24["inner_minus_outer_percentage_points"]["contains_ryanmen"])}。

exact宣言打牌番号別では4〜17打目の全区分で3指標ともinner−outerが正だった。一方、3打目は負で、2打目と18打目以降は件数が小さいため、全巡目で一律の差とは解釈しない。

全期間10,706,714件でもinner−outer差は、筋待ち {_pp(full["inner_minus_outer_percentage_points"]["contains_suji"])}、好形 {_pp(full["inner_minus_outer_percentage_points"]["good_wait"])}、両面 {_pp(full["inner_minus_outer_percentage_points"]["contains_ryanmen"])}で同方向です。

## 検証

- canonical入力10,706,714件、17年度のsize/SHA256、DTO、件数を検証
- 全体・年度・exact宣言打牌番号の全record品質が既存baselineと整数一致
- 全pytest、Ruff、serial/parallel、全牌種ペアoracleを通過
- 決定論的監査25件を圧縮MJAI原本へ戻し、25/25一致、未発見カテゴリ0

## 解釈上の注意

1. 『嵌張落としに見える河』は外形分類で、実際の手牌構成や意図を証明しません。
2. A/Bはともに14枚から弱いブロックを処理した結果として主群へ含めます。
3. 筋待ち率は個別の筋牌の安全性を、好形率は実際の山残枚数を示しません。
4. 巡目別に差が残っても、因果効果や押し引きの最適解は示しません。

## ファイル

- 詳細結果: `research/results/kanchan-drop-analysis-v1.json`
- コンパクト結果: `outputs/work/kanchan-drop-analysis/result-summary.json`
- 図: `outputs/work/kanchan-drop-analysis/figures/`
"""
    (output_dir / "handoff.md").write_text(handoff, encoding="utf-8", newline="\n")

    brief = """# 記事ブリーフ

## 仮タイトル

嵌張落としに見える河は好形サインか――切る順番まで1,070万件で調べた

## 中心メッセージ

2020〜2025年の成立リーチ約494万件では、嵌張落としに見える河を持つリーチは14.12%。待ちはパターンなしより筋待ち率が3.88ポイント、5枚以上の好形率が3.97ポイント高かった。さらに、同じ嵌張形でも内側の牌から切った代表候補は、外側から切った候補より好形率が3.62ポイント高い。24形では4→2が2→4より3.98ポイント高かった。

巡目別では4〜17打目で同方向だが、3打目は逆方向である。この例外と初期・終盤の少数区分を隠さず記載する。

## 構成案

1. 13・24・35…と連続して切る河をどう定義したか
2. A/Bを同格にした理由――14枚から弱いブロックを処理する選択
3. パターンあり／なしの比較
4. 本題：内側から切ると本当に好形率が上がるか
5. 形別比較、特に4→2と2→4
6. exact宣言打牌番号、年度、対子落とし型なしでの確認
7. 言えること／言えないこと

## 表現上の注意

- 原則『嵌張落としに見える河』と書き、実手牌や意図を断定しない。
- 率の差はpercentage pointsで、相対増加率と混同しない。
- 『筋待ちが多い』を『その筋牌が安全』へ読み替えない。
- 46形はinner/outerではなく対称形として別扱いする。
"""
    (output_dir / "article-brief.md").write_text(brief, encoding="utf-8", newline="\n")

    prompt = """このフォルダの handoff.md、article-brief.md、result-summary.json と figures/ を読み、note向けの記事を作ってください。主期間は2020〜2025年です。中心は、嵌張落としに見える河のリーチはパターンなしより筋待ち率・5枚以上の好形率が約4ポイント高く、さらに代表候補で内側の牌から切った群は外側から切った群より好形率が3.62ポイント高いことです。24形の4→2対2→4も具体例として扱ってください。A/Bは同格です。『嵌張落とし』は手牌意図の証明ではないため原則『嵌張落としに見える河』と表現し、因果効果、押し引きの最適解、個別の筋牌の安全性を主張しないでください。数値はresult-summary.jsonを正本とし、図を適切な位置へ挿入してください。"""
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

    primary = summary["primary_2020_2025"]
    figures = output_dir / "figures"
    _comparison_chart(
        figures / "01-pattern-vs-none.png",
        title="嵌張落としに見える河の後は、待ちが良い",
        subtitle="2020–2025・成立リーチ 4,936,197件",
        left_label="パターンあり",
        right_label="パターンなし",
        left=primary["pattern_present_quality"],
        right=primary["no_pattern_quality"],
        differences=primary["pattern_minus_no_pattern_percentage_points"],
    )
    _comparison_chart(
        figures / "02-inner-vs-outer.png",
        title="内側の牌から切った方が、さらに好形",
        subtitle="2020–2025・最後に完成した代表候補で1リーチ1分類",
        left_label="内側先行",
        right_label="外側先行",
        left=primary["representative_order_groups"]["inner_first"],
        right=primary["representative_order_groups"]["outer_first"],
        differences=primary["inner_minus_outer_percentage_points"],
    )
    _shape_chart(summary, figures / "03-shape-order-difference.png")

    turn_rows = [
        row
        for row in summary["primary_by_exact_riichi_discard_number"]
        if 3 <= row["riichi_discard_number"] <= 17
    ]
    _line_chart(
        figures / "04-exact-turn-inner-minus-outer.png",
        title="3〜17打目の宣言打牌番号別に切り順を比較",
        subtitle="内側先行 − 外側先行（2・18打目以降は少数のため図から除外）",
        labels=[str(row["riichi_discard_number"]) for row in turn_rows],
        series=(
            (
                "筋待ち",
                [
                    row["inner_minus_outer_percentage_points"]["contains_suji"]
                    for row in turn_rows
                ],
                TEAL,
            ),
            (
                "好形",
                [
                    row["inner_minus_outer_percentage_points"]["good_wait"]
                    for row in turn_rows
                ],
                ORANGE,
            ),
            (
                "両面",
                [
                    row["inner_minus_outer_percentage_points"]["contains_ryanmen"]
                    for row in turn_rows
                ],
                BLUE,
            ),
        ),
        y_min=-4,
        y_max=8,
    )
    year_rows = [row for row in summary["by_year"] if 2020 <= row["year"] <= 2025]
    _line_chart(
        figures / "05-yearly-inner-minus-outer.png",
        title="2020〜2025年の各年で切り順を確認",
        subtitle="内側先行 − 外側先行（percentage points）",
        labels=[str(row["year"]) for row in year_rows],
        series=(
            (
                "筋待ち",
                [
                    row["inner_minus_outer_percentage_points"]["contains_suji"]
                    for row in year_rows
                ],
                TEAL,
            ),
            (
                "好形",
                [
                    row["inner_minus_outer_percentage_points"]["good_wait"]
                    for row in year_rows
                ],
                ORANGE,
            ),
            (
                "両面",
                [
                    row["inner_minus_outer_percentage_points"]["contains_ryanmen"]
                    for row in year_rows
                ],
                BLUE,
            ),
        ),
        y_min=2,
        y_max=6,
    )
    _composition_chart(summary, figures / "06-pattern-composition.png")
    distance_rows = [
        row
        for row in summary["primary_by_representative_distance"]
        if row["distance"] <= 10
    ]
    _line_chart(
        figures / "07-distance-inner-minus-outer.png",
        title="候補完成からリーチまでの距離別比較",
        subtitle="内側先行 − 外側先行（percentage points）",
        labels=[str(row["distance"]) for row in distance_rows],
        series=(
            (
                "筋待ち",
                [
                    row["inner_minus_outer_percentage_points"]["contains_suji"]
                    for row in distance_rows
                ],
                TEAL,
            ),
            (
                "好形",
                [
                    row["inner_minus_outer_percentage_points"]["good_wait"]
                    for row in distance_rows
                ],
                ORANGE,
            ),
            (
                "両面",
                [
                    row["inner_minus_outer_percentage_points"]["contains_ryanmen"]
                    for row in distance_rows
                ],
                BLUE,
            ),
        ),
        y_min=-2,
        y_max=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_handoff(args.result, args.output_dir)


if __name__ == "__main__":
    main()
