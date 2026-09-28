"""Render exportable figures for the first mahjong expected-value article.

The app figures reproduce the user's supplied screenshot values. The dice
experiment is an explicitly illustrative simulation, not mahjong evidence.
No mahjong decision or scoring engine is implemented here.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "codex-mahjong-article-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/articles/assets/expected-value-part1"
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/meiryo.ttc"),
    Path("C:/Windows/Fonts/NotoSansJP-VF.ttf"),
]

BG = "#faf8f2"
INK = "#193b33"
MUTED = "#58685f"
GREEN = "#176b57"
ORANGE = "#cc7c2e"
PALE_GREEN = "#e1eee6"
PALE_ORANGE = "#f4e7d6"
GRID = "#d7ded7"

CHIITOITSU = [
    {"wait": "四筒（ドラ）待ち", "discard": "西", "remaining": 3,
     "win_probability_percent": 24.46, "displayed_expected_points": 4732},
    {"wait": "西待ち", "discard": "四筒", "remaining": 3,
     "win_probability_percent": 24.46, "displayed_expected_points": 3105},
]
PARENT_POINTS = [
    {"label": "平和ドラ2", "ron": 11600, "tsumo_total": 12000,
     "ron_han_fu": "4翻30符", "tsumo_han_fu": "5翻"},
    {"label": "一通・高め", "ron": 12000, "tsumo_total": 18000,
     "ron_han_fu": "5翻", "tsumo_han_fu": "6翻"},
    {"label": "一通・安め", "ron": 5800, "tsumo_total": 7800,
     "ron_han_fu": "3翻30符", "tsumo_han_fu": "4翻20符"},
]


def setup_style() -> None:
    font_path = next((p for p in FONT_CANDIDATES if p.exists()), None)
    if font_path is None:
        raise FileNotFoundError("A Japanese font is required.")
    font_manager.fontManager.addfont(str(font_path))
    bold_path = Path("C:/Windows/Fonts/meiryob.ttc")
    if font_path.name == "meiryo.ttc" and bold_path.exists():
        font_manager.fontManager.addfont(str(bold_path))
    name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": name,
        "font.size": 16,
        "axes.unicode_minus": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "savefig.facecolor": BG,
        "svg.fonttype": "path",
    })


def base(title: str, subtitle: str, height: float = 6.8) -> plt.Figure:
    fig = plt.figure(figsize=(11, height), dpi=180)
    fig.text(0.055, 0.94, title, fontsize=23, weight="bold", va="top")
    fig.text(0.055, 0.862, subtitle, fontsize=15.5, color=MUTED, va="top")
    return fig


def footer(fig: plt.Figure, line1: str, line2: str = "") -> None:
    fig.text(0.055, 0.078, line1, fontsize=13.5, color=MUTED, va="top")
    if line2:
        fig.text(0.055, 0.042, line2, fontsize=13.5, color=MUTED, va="top")


def export(fig: plt.Figure, stem: str) -> None:
    # Verify that Japanese titles, annotations and axis labels fit the canvas.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    clipped = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        extent = artist.get_window_extent(renderer)
        if extent.width > 0 and (
            extent.x0 < -2 or extent.y0 < -2
            or extent.x1 > width + 2 or extent.y1 > height + 2
        ):
            clipped.append(artist.get_text())
    if clipped:
        raise ValueError(f"Text outside {stem}: {clipped}")
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def box(ax: plt.Axes, x: float, y: float, w: float, h: float,
        heading: str, body: str = "", fill: str = PALE_GREEN) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fill, edgecolor="none", transform=ax.transAxes,
    ))
    heading_y = y + h * (0.73 if body else 0.5)
    ax.text(x + w / 2, heading_y, heading, fontsize=19, weight="bold",
            ha="center", va="center", transform=ax.transAxes)
    if body:
        ax.text(x + w / 2, y + h * 0.33, body, fontsize=16,
                ha="center", va="center", linespacing=1.6,
                transform=ax.transAxes)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          connectionstyle: str = "arc3,rad=0") -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=19,
        linewidth=1.6, color=MUTED, transform=ax.transAxes,
        connectionstyle=connectionstyle,
    ))


def model_scope() -> None:
    fig = base("「期待値」は、何を含めて計算するかで変わる",
               "同じ手牌でも、評価する出来事と結果を先に確認する", 7.6)
    ax = fig.add_axes((0.055, 0.15, 0.89, 0.62))
    ax.set_axis_off()
    box(ax, 0.31, 0.81, 0.38, 0.15, "同じ手牌・同じ巡目")
    arrow(ax, (0.40, 0.79), (0.25, 0.66))
    arrow(ax, (0.60, 0.79), (0.75, 0.66))
    box(ax, 0.04, 0.26, 0.42, 0.36, "一人麻雀の計算例",
        "自分のツモ・手牌変化\nツモ和了の点数", PALE_GREEN)
    box(ax, 0.54, 0.26, 0.42, 0.36, "他家を含む計算例",
        "ロン・放銃・被ツモ・鳴き\n他家の和了・流局など", PALE_ORANGE)
    arrow(ax, (0.25, 0.24), (0.25, 0.17))
    arrow(ax, (0.75, 0.24), (0.75, 0.17))
    ax.text(0.25, 0.10, "ツモ和了の点数期待値", ha="center", va="center",
            fontsize=18, weight="bold", transform=ax.transAxes)
    ax.text(0.75, 0.10, "局収支／順位・段位ptなど", ha="center", va="center",
            fontsize=18, weight="bold", transform=ax.transAxes)
    footer(fig, "概念図。右側に挙げた要素を、すべてのシミュレータが含むわけではない。")
    export(fig, "01-model-scope")


def chiitoitsu_comparison() -> None:
    fig = base("ドラ単騎と西単騎：ツモだけの計算ではどうなる？",
               "提供画像の表示値を比較。どちらも残り3枚・東家・8巡目", 6.8)
    labels = [item["wait"] for item in CHIITOITSU]
    colors = [ORANGE, GREEN]
    ax_p = fig.add_axes((0.055, 0.27, 0.40, 0.47))
    ax_e = fig.add_axes((0.545, 0.27, 0.40, 0.47))
    for ax, field, maximum, title, formatter, ticks in (
        (ax_p, "win_probability_percent", 35, "表示された和了確率",
         lambda x: f"{x:.2f}%", [0, 10, 20, 30]),
        (ax_e, "displayed_expected_points", 6300, "表示された点数期待値",
         lambda x: f"{x:,.0f}点", [0, 2000, 4000, 6000]),
    ):
        values = [item[field] for item in CHIITOITSU]
        positions = [1, 0]
        ax.barh(positions, values, height=0.39, color=colors)
        ax.set_xlim(0, maximum)
        ax.set_ylim(-0.45, 1.5)
        ax.set_yticks([])
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", labelsize=13, length=0, pad=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=18, pad=21)
        for y, label, value in zip(positions, labels, values):
            ax.text(0, y + 0.27, label, fontsize=15.5, ha="left", va="bottom")
            ax.text(value + maximum * 0.035, y, formatter(value),
                    fontsize=17, weight="bold", va="center")
    fig.text(0.055, 0.19, "和了確率は同じでも、打点の違いで期待値に1,627点の差。",
             fontsize=17, weight="bold")
    footer(fig, "pystyle v0.9.8のユーザー提供画像から作図。独立した再計算ではない。",
           "全設定は画像から特定できず、四人麻雀の実測和了率を示すグラフではない。")
    export(fig, "02-chiitoitsu-comparison")


def ron_tsumo_points() -> None:
    fig = base("ツモの1翻は、満貫・跳満の境目で効き方が変わる",
               "すべて立直を含む親の和了点。ツモは3人からの受け取り合計", 7.3)
    ax = fig.add_axes((0.14, 0.255, 0.795, 0.51))
    positions = np.arange(len(PARENT_POINTS))
    width = 0.30
    ron = [item["ron"] for item in PARENT_POINTS]
    tsumo = [item["tsumo_total"] for item in PARENT_POINTS]
    bars_r = ax.bar(positions - width / 2, ron, width, color=GREEN, label="ロン")
    bars_t = ax.bar(positions + width / 2, tsumo, width, color=ORANGE, label="ツモ")
    ax.set_ylim(0, 21500)
    ax.set_yticks([0, 6000, 12000, 18000], ["0", "6,000", "12,000", "18,000"])
    ax.set_ylabel("受け取り点数", labelpad=12, fontsize=15)
    ax.set_xticks(positions, [item["label"] for item in PARENT_POINTS])
    ax.tick_params(axis="both", labelsize=15, length=0, pad=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=16)
    for bar in [*bars_r, *bars_t]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 440,
                f"{bar.get_height():,.0f}", fontsize=15.5,
                weight="bold", ha="center", va="bottom")
    fig.text(0.105, 0.17, "ツモで増える点数：平和ドラ2は＋400点／一通の高めは＋6,000点",
             fontsize=16, weight="bold")
    footer(fig, "裏ドラ・一発・本場等なし。切り上げ満貫なし。各和了形の打点比較。",
           "和了確率や別の手牌進行を含めた、打牌ごとの期待値比較ではない。")
    export(fig, "03-ron-tsumo-points")


def monte_carlo_flow() -> None:
    fig = base("モンテカルロ法：試した結果を平均する",
               "未来の出来事を、設定した確率と行動方針で発生させる", 8.2)
    ax = fig.add_axes((0.055, 0.13, 0.89, 0.67))
    ax.set_axis_off()
    positions = [0.81, 0.56, 0.31, 0.06]
    headings = ["① 条件を決める", "② 1回分の展開を進める",
                "③ 結果を記録する", "④ 多数回の結果を平均する"]
    bodies = ["手牌・河・点棒／確率・行動方針",
              "ツモ・ロン・放銃・他家の行動など",
              "和了・放銃・被ツモ・流局の得失点など",
              "結果の合計 ÷ 試行回数"]
    for index, (y, heading, body) in enumerate(zip(positions, headings, bodies)):
        box(ax, 0.12, y, 0.63, 0.17, heading, body,
            PALE_ORANGE if index == 3 else PALE_GREEN)
        if index < 3:
            arrow(ax, (0.435, y - 0.02), (0.435, positions[index + 1] + 0.19))
    ax.add_patch(FancyArrowPatch(
        (0.775, 0.395), (0.775, 0.645),
        connectionstyle="arc3,rad=0.75", arrowstyle="-|>",
        mutation_scale=20, linewidth=1.7, color=ORANGE,
        transform=ax.transAxes,
    ))
    ax.text(0.94, 0.52, "②・③を\n繰り返す", fontsize=15.5,
            ha="center", va="center", linespacing=1.6, transform=ax.transAxes)
    footer(fig, "別の打牌も計算して比較する。図は一般的な方法を示した概念図。",
           "新nisiの具体的な実装を確認して描いた図ではない。")
    export(fig, "04-monte-carlo-flow")


def sampling_and_model() -> dict:
    count = 100_000
    rng = np.random.default_rng(20260926)
    fair = rng.integers(1, 7, size=count)
    biased = rng.choice(np.arange(1, 7), size=count,
                        p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    samples = np.unique(np.geomspace(20, count, 130).astype(int))
    fair_means = np.cumsum(fair)[samples - 1] / samples
    biased_means = np.cumsum(biased)[samples - 1] / samples
    fig = base("回数を増やしても、設定したモデルの違いは残る",
               "公平なサイコロを調べたいとき、6を出やすく設定したら？", 7.1)
    ax = fig.add_axes((0.105, 0.25, 0.83, 0.52))
    ax.plot(samples, fair_means, color=GREEN, linewidth=2.3,
            label="公平なモデル：各目が1/6")
    ax.plot(samples, biased_means, color=ORANGE, linewidth=2.3,
            label="偏ったモデル：6が50%、他は各10%")
    ax.axhline(3.5, color=GREEN, alpha=0.60, linewidth=1.1, linestyle="--")
    ax.axhline(4.5, color=ORANGE, alpha=0.60, linewidth=1.1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(20, count * 1.08)
    ax.set_xticks([100, 1000, 10000, 100000], ["100回", "1,000回", "1万回", "10万回"])
    ax.set_ylim(min(3.20, float(fair_means.min()) - 0.12),
                max(4.86, float(biased_means.max()) + 0.12))
    ax.set_yticks([3.5, 4.0, 4.5])
    ax.set_xlabel("累積の試行回数（対数目盛）", fontsize=14, labelpad=12)
    ax.set_ylabel("出目の平均", fontsize=15, labelpad=12)
    ax.tick_params(axis="both", labelsize=14, length=0, pad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.25, 0.51),
              frameon=False, fontsize=14)
    ax.annotate("設定した期待値 3.5", (26000, 3.5), xytext=(26000, 3.31),
                fontsize=13.5, color=GREEN)
    ax.annotate("設定した期待値 4.5", (26000, 4.5), xytext=(26000, 4.67),
                fontsize=13.5, color=ORANGE)
    footer(fig, "説明用のサイコロ実験。各10万回を乱数で試行。麻雀の実測データではない。",
           "途中の平均は上下する。試行を増やしても、6を出やすくした設定は変わらない。")
    export(fig, "05-sampling-and-model")
    return {
        "seed": 20260926, "trials_per_model": count,
        "fair_model_expected_value": 3.5,
        "biased_model_expected_value": 4.5,
        "fair_final_mean": float(fair.mean()),
        "biased_final_mean": float(biased.mean()),
        "plotted_sample_counts": samples.tolist(),
        "fair_cumulative_means": fair_means.tolist(),
        "biased_cumulative_means": biased_means.tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    # Arithmetic checks on the display values; no scoring rules are inferred.
    assert CHIITOITSU[0]["displayed_expected_points"] - CHIITOITSU[1]["displayed_expected_points"] == 1627
    assert PARENT_POINTS[0]["tsumo_total"] - PARENT_POINTS[0]["ron"] == 400
    assert PARENT_POINTS[1]["tsumo_total"] - PARENT_POINTS[1]["ron"] == 6000
    model_scope()
    chiitoitsu_comparison()
    ron_tsumo_points()
    monte_carlo_flow()
    experiment = sampling_and_model()
    metadata = {
        "article": "expected-value-part1",
        "chiitoitsu_screenshot_values": CHIITOITSU,
        "parent_scoring_example": PARENT_POINTS,
        "dice_illustration": experiment,
        "provenance": {
            "chiitoitsu": "User screenshot; no independent app rerun; full settings unknown.",
            "payouts": "Parent riichi hands; no kiriage, ura, ippatsu or honba; not complete discard EV.",
            "dice": "Author-generated illustrative simulation, not mahjong observations.",
        },
    }
    (OUT / "figure-data.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created five PNG/SVG figure pairs in {OUT}")
    print(f"Dice final means: fair={experiment['fair_final_mean']:.5f}, biased={experiment['biased_final_mean']:.5f}")


if __name__ == "__main__":
    main()
