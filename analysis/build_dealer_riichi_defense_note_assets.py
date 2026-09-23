from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "dealer-riichi-defense" / "summary-v3.json"
OUTPUT = ROOT / "outputs" / "dealer-riichi-defense" / "article-assets"

WIDTH = 1600
HEIGHT = 900
SCALE = 2

BG = "#F7F3EA"
PAPER = "#FFFCF6"
INK = "#24323D"
MUTED = "#66737D"
GRID = "#D8D3C9"
DEALER = "#D9584A"
CHILD = "#3F6FA0"
TEAL = "#27877E"
GOLD = "#D39B32"
LIGHT_DEALER = "#F4D6D1"
LIGHT_CHILD = "#DCE7F2"

FONT_REGULAR = Path(r"C:\Windows\Fonts\YuGothM.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\YuGothB.ttc")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size * SCALE)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    return image, ImageDraw.Draw(image)


def xy(values: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(round(value * SCALE) for value in values)


def text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    value: str,
    size: int,
    *,
    color: str = INK,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(
        xy(position), value, font=font(size, bold=bold), fill=color, anchor=anchor
    )


def rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    radius: int = 24,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(
        xy(box),
        radius=radius * SCALE,
        fill=fill,
        outline=outline,
        width=width * SCALE,
    )


def line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: str,
    width: int = 2,
) -> None:
    draw.line([xy(point) for point in points], fill=fill, width=width * SCALE)


def title_block(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    text(draw, (80, 55), title, 48, bold=True)
    text(draw, (82, 118), subtitle, 23, color=MUTED)
    line(draw, [(80, 158), (1520, 158)], fill=GRID, width=2)


def footer(draw: ImageDraw.ImageDraw, number: int) -> None:
    line(draw, [(80, 842), (1520, 842)], fill=GRID, width=1)
    text(draw, (80, 858), "天鳳鳳凰卓・東場・2020〜2025年／MJAI", 17, color=MUTED)
    text(draw, (1520, 858), f"FIG. {number}", 17, color=MUTED, anchor="ra")


def save(image: Image.Image, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(
        OUTPUT / name,
        format="PNG",
        optimize=True,
    )


def decimal_rate(metric: dict[str, Any], role: str) -> float:
    return float(metric["common_support_standardized"][f"{role}_rate"]["decimal"])


def breakdown_rate(
    analysis: dict[str, Any], metric: str, dimension: str, value: str, role: str
) -> float:
    for row in analysis["breakdowns"]:
        if (
            row["metric"] == metric
            and row["dimension"] == dimension
            and row["value"] == value
        ):
            return float(row["raw"][role]["rate"]["decimal"])
    raise KeyError((metric, dimension, value, role))


def draw_grouped_bar_panel(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    categories: list[str],
    dealer_values: list[float],
    child_values: list[float],
    maximum: float,
    tick_step: float,
    percent_digits: int = 1,
) -> None:
    left, top, right, bottom = box
    chart_left = left + 80
    chart_right = right - 25
    chart_top = top + 20
    chart_bottom = bottom - 80
    for tick in [i * tick_step for i in range(round(maximum / tick_step) + 1)]:
        y = chart_bottom - (tick / maximum) * (chart_bottom - chart_top)
        line(draw, [(chart_left, y), (chart_right, y)], fill=GRID, width=1)
        text(draw, (chart_left - 12, y), f"{tick:.0f}%", 17, color=MUTED, anchor="rm")
    group_width = (chart_right - chart_left) / len(categories)
    bar_width = min(65, group_width * 0.25)
    for index, category in enumerate(categories):
        center = chart_left + group_width * (index + 0.5)
        for value, offset, color in (
            (dealer_values[index], -bar_width * 0.62, DEALER),
            (child_values[index], bar_width * 0.62, CHILD),
        ):
            height = (value / maximum) * (chart_bottom - chart_top)
            rounded_rectangle(
                draw,
                (
                    center + offset - bar_width / 2,
                    chart_bottom - height,
                    center + offset + bar_width / 2,
                    chart_bottom,
                ),
                radius=8,
                fill=color,
            )
            text(
                draw,
                (center + offset, chart_bottom - height - 10),
                f"{value:.{percent_digits}f}",
                18,
                color=color,
                bold=True,
                anchor="ms",
            )
        text(draw, (center, chart_bottom + 25), category, 20, bold=True, anchor="ma")


def build_cover() -> None:
    image, draw = canvas()
    rounded_rectangle(
        draw, (55, 45, 1545, 855), radius=42, fill=PAPER, outline=GRID, width=2
    )
    rounded_rectangle(draw, (95, 92, 325, 142), radius=25, fill=INK)
    text(draw, (210, 117), "麻雀データ分析", 22, color=PAPER, bold=True, anchor="mm")
    text(draw, (95, 205), "親リーチはなぜ", 78, bold=True)
    text(draw, (95, 305), "ロンされにくい？", 78, bold=True, color=DEALER)
    text(draw, (98, 430), "約398万局で、子の現物率と無筋率を調べた", 34, color=MUTED)
    cards = [
        (95, 540, 510, 755, "+2.20pt", "現物打牌率", DEALER, LIGHT_DEALER),
        (592, 540, 1007, 755, "−1.29pt", "無筋数牌率", CHILD, LIGHT_CHILD),
        (1089, 540, 1504, 755, "−12.8%", "1打牌当たりロン率", TEAL, "#DCECE9"),
    ]
    for left, top, right, bottom, number, label, color, fill in cards:
        rounded_rectangle(draw, (left, top, right, bottom), radius=28, fill=fill)
        text(
            draw,
            ((left + right) / 2, top + 78),
            number,
            50,
            color=color,
            bold=True,
            anchor="mm",
        )
        text(
            draw,
            ((left + right) / 2, top + 145),
            label,
            23,
            color=INK,
            bold=True,
            anchor="mm",
        )
    text(
        draw,
        (1500, 815),
        "天鳳鳳凰卓・東場・2020〜2025年",
        18,
        color=MUTED,
        anchor="ra",
    )
    save(image, "00-cover.png")


def build_main_comparison(analysis: dict[str, Any]) -> None:
    metrics = analysis["primary_child_responders"]
    image, draw = canvas()
    title_block(
        draw,
        "親リーチ後では、現物が増えて無筋が減る",
        "共通支持内で標準化した子respondersの行動率",
    )
    rounded_rectangle(draw, (80, 185, 1520, 755), radius=28, fill=PAPER)
    categories = ["現物", "無筋数牌", "ツモ切り"]
    keys = ["genbutsu_share", "unsuji_numbered_share", "tsumogiri_share"]
    dealer = [decimal_rate(metrics[key], "dealer") * 100 for key in keys]
    child = [decimal_rate(metrics[key], "nondealer") * 100 for key in keys]
    draw_grouped_bar_panel(
        draw,
        box=(95, 200, 895, 745),
        categories=categories,
        dealer_values=dealer,
        child_values=child,
        maximum=75,
        tick_step=15,
    )
    panel_left = 940
    rows = [
        ("現物", dealer[0] - child[0], "+2.20pt", DEALER),
        ("無筋数牌", dealer[1] - child[1], "−1.29pt", CHILD),
        ("ツモ切り", dealer[2] - child[2], "−1.25pt", CHILD),
    ]
    text(draw, (panel_left, 245), "親−子の差", 27, bold=True)
    for index, (label, _, display, color) in enumerate(rows):
        y = 340 + index * 120
        rounded_rectangle(draw, (panel_left, y - 48, 1450, y + 48), radius=20, fill=BG)
        text(draw, (panel_left + 28, y), label, 24, bold=True, anchor="lm")
        text(draw, (1418, y), display, 34, color=color, bold=True, anchor="rm")
    rounded_rectangle(draw, (1080, 675, 1250, 715), radius=20, fill=DEALER)
    text(draw, (1268, 695), "親リーチ後", 19, color=INK, anchor="lm")
    rounded_rectangle(draw, (1080, 727, 1250, 767), radius=20, fill=CHILD)
    text(draw, (1268, 747), "子リーチ後", 19, color=INK, anchor="lm")
    footer(draw, 1)
    save(image, "01-main-comparison.png")


def build_difference_chart(analysis: dict[str, Any]) -> None:
    metrics = analysis["primary_child_responders"]
    ordered = [
        ("現物打牌率", "genbutsu_share"),
        ("区間内の鳴き率", "any_call_share"),
        ("後追いリーチ率", "later_riichi_share"),
        ("1打牌当たりロン率", "focal_ron_per_discard"),
        ("ツモ切り率", "tsumogiri_share"),
        ("無筋数牌率", "unsuji_numbered_share"),
        ("非現物数牌中の無筋率", "unsuji_among_numbered_non_genbutsu"),
    ]
    values = [
        float(
            metrics[key]["common_support_standardized"]["dealer_minus_nondealer"][
                "decimal"
            ]
        )
        * 100
        for _, key in ordered
    ]
    image, draw = canvas()
    title_block(
        draw,
        "行動差は『全面的なベタオリ』ではない",
        "標準化後の差（親リーチ後 − 子リーチ後、ポイント）",
    )
    left, right = 520, 1460
    top, bottom = 210, 775
    minimum, maximum = -1.8, 2.5
    zero_x = left + (0 - minimum) / (maximum - minimum) * (right - left)
    for tick in [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        x_pos = left + (tick - minimum) / (maximum - minimum) * (right - left)
        line(
            draw,
            [(x_pos, top), (x_pos, bottom)],
            fill=INK if tick == 0 else GRID,
            width=2 if tick == 0 else 1,
        )
        text(draw, (x_pos, bottom + 24), f"{tick:+.1f}", 17, color=MUTED, anchor="ma")
    row_height = (bottom - top) / len(ordered)
    for index, ((label, _), value) in enumerate(zip(ordered, values, strict=True)):
        y = top + row_height * (index + 0.5)
        text(draw, (485, y), label, 23, bold=True, anchor="rm")
        x_value = left + (value - minimum) / (maximum - minimum) * (right - left)
        color = DEALER if value > 0 else CHILD
        line(draw, [(zero_x, y), (x_value, y)], fill=color, width=18)
        draw.ellipse(
            xy((x_value - 13, y - 13, x_value + 13, y + 13)),
            fill=color,
        )
        anchor = "lm" if value >= 0 else "rm"
        label_x = x_value + 20 if value >= 0 else x_value - 20
        text(
            draw,
            (label_x, y),
            f"{value:+.3f}pt",
            21,
            color=color,
            bold=True,
            anchor=anchor,
        )
    text(draw, (left, 810), "← 親リーチ後の方が低い", 18, color=CHILD)
    text(draw, (right, 810), "親リーチ後の方が高い →", 18, color=DEALER, anchor="ra")
    footer(draw, 2)
    save(image, "02-adjusted-differences.png")


def build_initial_response(analysis: dict[str, Any]) -> None:
    image, draw = canvas()
    title_block(
        draw,
        "差はリーチ後の最初の打牌から現れる",
        "応答順別の生集計。3+は3回目以降の応答打牌",
    )
    values: dict[str, tuple[list[float], list[float]]] = {}
    for metric in ("genbutsu_share", "unsuji_numbered_share"):
        dealer = [
            breakdown_rate(analysis, metric, "decision_ordinal", value, "dealer") * 100
            for value in ("1", "2", "3+")
        ]
        child = [
            breakdown_rate(analysis, metric, "decision_ordinal", value, "nondealer")
            * 100
            for value in ("1", "2", "3+")
        ]
        values[metric] = dealer, child
    panels = [
        (90, 205, 790, 755, "現物打牌率", "genbutsu_share", 80, 20),
        (830, 205, 1510, 755, "無筋数牌率", "unsuji_numbered_share", 20, 5),
    ]
    for left, top, right, bottom, label, metric, maximum, tick in panels:
        rounded_rectangle(draw, (left, top, right, bottom), radius=28, fill=PAPER)
        text(draw, ((left + right) / 2, top + 42), label, 29, bold=True, anchor="ma")
        dealer, child = values[metric]
        draw_grouped_bar_panel(
            draw,
            box=(left + 10, top + 65, right - 10, bottom),
            categories=["1打目", "2打目", "3打目以降"],
            dealer_values=dealer,
            child_values=child,
            maximum=maximum,
            tick_step=tick,
        )
    rounded_rectangle(draw, (1110, 785, 1220, 818), radius=16, fill=DEALER)
    text(draw, (1235, 801), "親リーチ後", 17, anchor="lm")
    rounded_rectangle(draw, (1350, 785, 1460, 818), radius=16, fill=CHILD)
    text(draw, (1475, 801), "子リーチ後", 17, anchor="lm")
    footer(draw, 3)
    save(image, "03-initial-response.png")


def build_outcomes(analysis: dict[str, Any]) -> None:
    outcomes = analysis["terminal_outcomes"]
    focals = analysis["focals"]
    rates: dict[str, list[float]] = {"dealer": [], "nondealer": []}
    for role in ("dealer", "nondealer"):
        focal = focals[role]
        ron = outcomes[role]["focal_ron"]
        tsumo = outcomes[role]["focal_tsumo"]
        rates[role] = [
            ron / focal * 100,
            tsumo / focal * 100,
            (ron + tsumo) / focal * 100,
            tsumo / (ron + tsumo) * 100,
        ]
    image, draw = canvas()
    title_block(
        draw,
        "親リーチはロンよりツモに寄っている",
        "各局最初の成立リーチについて、局末まで追跡した結果",
    )
    rounded_rectangle(draw, (80, 190, 1520, 760), radius=28, fill=PAPER)
    draw_grouped_bar_panel(
        draw,
        box=(105, 210, 1490, 755),
        categories=["ロン和了率", "ツモ和了率", "全体和了率", "和了時ツモ割合"],
        dealer_values=rates["dealer"],
        child_values=rates["nondealer"],
        maximum=60,
        tick_step=10,
    )
    rounded_rectangle(draw, (1050, 780, 1180, 817), radius=18, fill=DEALER)
    text(draw, (1195, 798), "親リーチ", 18, anchor="lm")
    rounded_rectangle(draw, (1330, 780, 1460, 817), radius=18, fill=CHILD)
    text(draw, (1475, 798), "子リーチ", 18, anchor="lm")
    footer(draw, 4)
    save(image, "04-outcomes.png")


def build_annual_trends(data: dict[str, Any]) -> None:
    years = list(range(2020, 2026))
    metrics = [
        ("現物打牌率の差", "genbutsu_share", DEALER, 2.0, 2.4, "+.3f"),
        ("1打牌当たりロン率の差", "focal_ron_per_discard", TEAL, -0.23, -0.18, "+.3f"),
    ]
    image, draw = canvas()
    title_block(
        draw,
        "方向は2020〜2025年の全年度で一致",
        "標準化後の差（親リーチ後 − 子リーチ後、ポイント）",
    )
    panels = [(90, 205, 790, 755), (830, 205, 1510, 755)]
    for panel, (label, metric, color, y_min, y_max, display_format) in zip(
        panels, metrics, strict=True
    ):
        left, top, right, bottom = panel
        rounded_rectangle(draw, panel, radius=28, fill=PAPER)
        text(draw, ((left + right) / 2, top + 42), label, 28, bold=True, anchor="ma")
        chart_left, chart_right = left + 90, right - 35
        chart_top, chart_bottom = top + 115, bottom - 90
        values = []
        for year in years:
            result = data["periods"][f"year_{year}"]["all_riichi"]
            value = (
                float(
                    result["primary_child_responders"][metric][
                        "common_support_standardized"
                    ]["dealer_minus_nondealer"]["decimal"]
                )
                * 100
            )
            values.append(value)
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            value = y_min + (y_max - y_min) * fraction
            y = chart_bottom - fraction * (chart_bottom - chart_top)
            line(draw, [(chart_left, y), (chart_right, y)], fill=GRID, width=1)
            text(
                draw,
                (chart_left - 12, y),
                f"{value:{display_format}}",
                17,
                color=MUTED,
                anchor="rm",
            )
        points = []
        for index, (year, value) in enumerate(zip(years, values, strict=True)):
            x = chart_left + index / (len(years) - 1) * (chart_right - chart_left)
            y = chart_bottom - (value - y_min) / (y_max - y_min) * (
                chart_bottom - chart_top
            )
            points.append((x, y))
            text(draw, (x, chart_bottom + 25), str(year), 17, color=MUTED, anchor="ma")
        line(draw, points, fill=color, width=6)
        for (x, y), value in zip(points, values, strict=True):
            draw.ellipse(xy((x - 10, y - 10, x + 10, y + 10)), fill=color)
            text(
                draw,
                (x, y - 20),
                f"{value:+.3f}",
                18,
                color=color,
                bold=True,
                anchor="ms",
            )
        direction = "全年度でプラス" if values[0] > 0 else "全年度でマイナス"
        text(
            draw,
            ((left + right) / 2, bottom - 28),
            direction,
            20,
            color=color,
            bold=True,
            anchor="mm",
        )
    footer(draw, 5)
    save(image, "05-annual-trends.png")


def build_method_diagram() -> None:
    image, draw = canvas()
    title_block(
        draw,
        "比較対象を『子』に揃えた",
        "親respondersの攻守傾向を混ぜず、子の反応だけを比較",
    )
    columns = [
        (95, "① focal", "各局で最初に\n成立したリーチ", DEALER, LIGHT_DEALER),
        (470, "② 観測区間", "成立直後から\n二軒目成立直前まで", GOLD, "#F3E6C8"),
        (845, "③ responder", "親リーチ：子3人\n子リーチ：子2人", CHILD, LIGHT_CHILD),
        (1220, "④ 比較", "現物・無筋・鳴き\n後追い・ロン率", TEAL, "#DCECE9"),
    ]
    for index, (left, header, body, color, fill) in enumerate(columns):
        rounded_rectangle(
            draw,
            (left, 250, left + 285, 640),
            radius=30,
            fill=fill,
            outline=color,
            width=3,
        )
        rounded_rectangle(
            draw, (left + 25, 280, left + 260, 340), radius=28, fill=color
        )
        text(draw, (left + 142, 310), header, 23, color=PAPER, bold=True, anchor="mm")
        for line_index, body_line in enumerate(body.splitlines()):
            text(
                draw,
                (left + 142, 440 + line_index * 60),
                body_line,
                25,
                bold=True,
                anchor="mm",
            )
        if index < len(columns) - 1:
            line(draw, [(left + 300, 445), (left + 355, 445)], fill=INK, width=5)
            draw.polygon(
                [xy((left + 355, 429)), xy((left + 380, 445)), xy((left + 355, 461))],
                fill=INK,
            )
    rounded_rectangle(draw, (205, 700, 1395, 780), radius=26, fill=INK)
    text(
        draw,
        (800, 740),
        "巡目・待ち・席順・開閉状態・応答順・安全牌数・年度を標準化",
        26,
        color=PAPER,
        bold=True,
        anchor="mm",
    )
    footer(draw, 6)
    save(image, "06-method.png")


def build_sample_flow(analysis: dict[str, Any]) -> None:
    focals = analysis["focals"]
    image, draw = canvas()
    title_block(
        draw,
        "主分析は約398万局・約3900万打牌",
        "各局最初の成立リーチと、primary child respondersの観測数",
    )
    total_focals = focals["dealer"] + focals["nondealer"]
    metrics = [
        ("最初の成立リーチ", total_focals, INK, "#E5E3DE"),
        ("親リーチ", focals["dealer"], DEALER, LIGHT_DEALER),
        ("子リーチ", focals["nondealer"], CHILD, LIGHT_CHILD),
        ("子の応答打牌", 38_996_157, TEAL, "#DCECE9"),
    ]
    positions = [
        (95, 240, 1505, 390),
        (95, 470, 715, 650),
        (885, 470, 1505, 650),
        (375, 700, 1225, 810),
    ]
    for (label, value, color, fill), box in zip(metrics, positions, strict=True):
        rounded_rectangle(draw, box, radius=28, fill=fill, outline=color, width=3)
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        text(
            draw,
            (center_x, center_y - 25),
            f"{value:,}",
            44,
            color=color,
            bold=True,
            anchor="mm",
        )
        text(
            draw,
            (center_x, center_y + 37),
            label,
            23,
            color=INK,
            bold=True,
            anchor="mm",
        )
    line(draw, [(800, 390), (800, 430)], fill=INK, width=5)
    line(draw, [(405, 430), (1195, 430)], fill=INK, width=5)
    line(draw, [(405, 430), (405, 470)], fill=INK, width=5)
    line(draw, [(1195, 430), (1195, 470)], fill=INK, width=5)
    line(draw, [(405, 650), (405, 680), (800, 680), (800, 700)], fill=INK, width=5)
    line(draw, [(1195, 650), (1195, 680), (800, 680)], fill=INK, width=5)
    footer(draw, 7)
    save(image, "07-sample-flow.png")


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    analysis = data["periods"]["primary_2020_2025"]["all_riichi"]
    build_cover()
    build_main_comparison(analysis)
    build_difference_chart(analysis)
    build_initial_response(analysis)
    build_outcomes(analysis)
    build_annual_trends(data)
    build_method_diagram()
    build_sample_flow(analysis)
    print(f"wrote 8 PNG assets to {OUTPUT}")


if __name__ == "__main__":
    main()
