"""Render reader-facing yakuhai-dash charts from the audited v3 summary."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/yakuhai-dash/summary-v3.json"
OUT = ROOT / "outputs/yakuhai-dash/note"
FONT = Path("C:/Windows/Fonts/NotoSansJP-VF.ttf")

BG = "#f7f5ef"
INK = "#172d2a"
MUTED = "#667570"
GREEN = "#117b63"
PALE_GREEN = "#cee8dc"
ORANGE = "#e69643"
GRID = "#d9dfd9"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size, index=0)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (1400, 790), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((35, 30, 1365, 755), radius=26, fill="white")
    d.text((100, 74), title, font=font(42, True), fill=INK)
    d.text((100, 143), subtitle, font=font(23), fill=MUTED)
    return im, d


def footer(d: ImageDraw.ImageDraw, text: str) -> None:
    d.line((100, 687, 1300, 687), fill=GRID, width=2)
    d.text((100, 704), text, font=font(19), fill=MUTED)


def chart_pairing(data: dict) -> None:
    source = data["overall"]["initial_singletons"]["all"]
    draws = {item["self_draw_number"]: item["cumulative"]["rate"] * 100
             for item in source["self_pair_formation_by_draw"]}
    vals = [draws[1], draws[3], draws[5], draws[10],
            source["outcomes"][0]["rate"] * 100]
    labels = ["1回目", "3回目", "5回目", "10回目", "観測全体"]
    im, d = canvas("手元の役牌が対子になるのは、いつ？",
                   "配牌で1枚だけあった役牌を、切る前に自分で重ねた割合（累積）")
    left, top, bottom = 150, 245, 610
    for tick in range(0, 11, 2):
        y = bottom - tick / 10 * (bottom - top)
        d.line((left, y, 1290, y), fill=GRID, width=2)
        d.text((100, y - 14), f"{tick}%", font=font(21), fill=MUTED)
    for i, (label, val) in enumerate(zip(labels, vals)):
        x = 255 + i * 240
        y = bottom - val / 10 * (bottom - top)
        d.rounded_rectangle((x - 67, y, x + 67, bottom), radius=10,
                            fill=GREEN if i == 4 else PALE_GREEN)
        d.text((x, y - 40), f"{val:.1f}%", anchor="mm", font=font(30, True), fill=INK)
        d.text((x, bottom + 38), label, anchor="mm", font=font(24), fill=INK)
    footer(d, "対象：配牌時の孤立役牌 3,494万例／横軸は局の巡目ではなく本人の自摸回数")
    im.save(OUT / "01-self-pair-cumulative.png", optimize=True)


def chart_opponent(data: dict) -> None:
    source = data["overall"]["initial_singletons"]["all"]
    rows = source["singleton_discards_by_actor_discard_number"][:5]
    able = [row["opponent_pon_capable"]["rate"] * 100 for row in rows]
    pon = [row["actual_pon"]["rate"] * 100 for row in rows]
    im, d = canvas("役牌を切る時、相手はポンできる形？",
                   "配牌で1枚だけあった役牌を切った場面を、本人の何打目かで分類")
    left, right, top, bottom = 170, 1260, 260, 585
    for tick in range(0, 16, 4):
        y = bottom - tick / 16 * (bottom - top)
        d.line((left, y, right, y), fill=GRID, width=2)
        d.text((100, y - 14), f"{tick}%", font=font(21), fill=MUTED)
    xs = [240 + i * 245 for i in range(5)]
    for values, color in ((able, GREEN), (pon, ORANGE)):
        pts = [(x, bottom - v / 16 * (bottom - top)) for x, v in zip(xs, values)]
        d.line(pts, fill=color, width=6, joint="curve")
        for x, y in pts:
            d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    for i, x in enumerate(xs):
        d.text((x, bottom + 36), f"{i + 1}打目", anchor="mm", font=font(24), fill=INK)
        d.text((x, bottom - able[i] / 16 * (bottom - top) - 27),
               f"{able[i]:.1f}", anchor="mm", font=font(24, True), fill=GREEN)
        d.text((x, bottom - pon[i] / 16 * (bottom - top) + 27),
               f"{pon[i]:.1f}", anchor="mm", font=font(24, True), fill=ORANGE)
    d.line((270, 222, 310, 222), fill=GREEN, width=6)
    d.text((320, 205), "相手にポンできる形がある", font=font(21), fill=INK)
    d.line((755, 222, 795, 222), fill=ORANGE, width=6)
    d.text((805, 205), "実際にポンされた", font=font(21), fill=INK)
    footer(d, "1打目は508万例。各点は別の牌譜群で、同じ手牌の早切り・後切り比較ではない")
    im.save(OUT / "02-pon-risk-by-discard.png", optimize=True)


def chart_after_pair(data: dict) -> None:
    source = data["overall"]["initial_singletons"]["all"]
    follow = source["self_pair_followup"]
    im, d = canvas("対子になった後、その形は残った？",
                   "2枚目を自摸した約336万例の、その後最初の自分の打牌直後")
    rows = [
        ("対子を残し、ポンできる形", follow["pon_shape"]["rate"] * 100, GREEN),
        ("対子を崩した", follow["pair_broken"]["rate"] * 100, ORANGE),
        ("対子は残るがリーチ中", follow["riichi_blocked"]["rate"] * 100, MUTED),
        ("次の打牌がない", follow["no_followup_discard"]["rate"] * 100, MUTED),
    ]
    for i, (label, val, color) in enumerate(rows):
        y = 260 + i * 94
        d.text((110, y), label, font=font(25), fill=INK)
        d.rounded_rectangle((545, y + 2, 1165, y + 32), radius=12, fill=GRID)
        width = max(8, int(620 * val / 100))
        d.rounded_rectangle((545, y + 2, 545 + width, y + 32), radius=12, fill=color)
        d.text((1270, y + 15), f"{val:.1f}%", anchor="mm", font=font(27, True), fill=color)
    d.rounded_rectangle((110, 638, 1290, 674), radius=9, fill=BG)
    d.text((130, 644), "※ 96.5%は『対子になった例のうち』。配牌の孤立役牌すべてではありません。",
           font=font(20), fill=MUTED)
    footer(d, "ポンできる形は、実際にポンした・ポンする機会が来た、という意味ではない")
    im.save(OUT / "03-after-pair.png", optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    chart_pairing(data)
    chart_opponent(data)
    chart_after_pair(data)


if __name__ == "__main__":
    main()
