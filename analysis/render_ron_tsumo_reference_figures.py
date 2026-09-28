"""Draw the article's explicitly hypothetical ron/tsumo comparison.

The points are supplied examples, not a scoring engine. These figures do not
estimate real win probabilities or replicate the app's complete discard EV.
"""

from __future__ import annotations

import json

from render_expected_value_part1_figures import (
    GREEN, ORANGE, GRID, MUTED, OUT, PALE_GREEN, PALE_ORANGE,
    base, box, export, footer, setup_style,
)

import matplotlib.pyplot as plt


EXAMPLES = [
    {"label": "平和ドラ2", "outcomes": [
        {"remaining": 8, "ron": 7700, "tsumo_total": 8000},
    ]},
    {"label": "一通・高安あり", "outcomes": [
        {"remaining": 4, "ron": 8000, "tsumo_total": 12000},
        {"remaining": 4, "ron": 3900, "tsumo_total": 5200},
    ]},
]


def comparison_data() -> list[dict]:
    data = []
    for example in EXAMPLES:
        remaining = sum(x["remaining"] for x in example["outcomes"])
        ron_index = sum(x["ron"] * x["remaining"] * 3 for x in example["outcomes"])
        tsumo_index = sum(x["tsumo_total"] * x["remaining"] for x in example["outcomes"])
        data.append({
            **example,
            "ron_index": ron_index,
            "tsumo_index": tsumo_index,
            "combined_index": ron_index + tsumo_index,
            "tsumo_normalized": tsumo_index / remaining,
            "combined_normalized": (ron_index + tsumo_index) / (remaining * 4),
        })
    assert data[0]["combined_index"] == 248800
    assert data[1]["combined_index"] == 211600
    assert data[0]["tsumo_normalized"] == 8000
    assert data[1]["tsumo_normalized"] == 8600
    assert data[0]["combined_normalized"] == 7775
    assert data[1]["combined_normalized"] == 6612.5
    return data


def four_channels() -> None:
    fig = base("簡易比較の考え方：ロンの相手3人＋自分のツモ1人",
               "人数を便宜的な重みにする。ロン率75％を意味する式ではない", 7.8)
    ax = fig.add_axes((0.055, 0.16, 0.89, 0.63))
    ax.set_axis_off()
    box(ax, 0.24, 0.79, 0.52, 0.14, "待ち牌は残り8枚と仮定")
    ax.text(0.365, 0.70, "他家3人：ロンで和了", ha="center", va="center",
            fontsize=17, weight="bold", color=GREEN, transform=ax.transAxes)
    ax.text(0.875, 0.70, "自分1人：ツモ和了", ha="center", va="center",
            fontsize=17, weight="bold", color=ORANGE, transform=ax.transAxes)
    for i, heading in enumerate(["他家A", "他家B", "他家C", "自分"]):
        box(ax, 0.015 + i * 0.255, 0.37, 0.21, 0.26, heading,
            "7,700点\n×8枚" if i < 3 else "8,000点\n×8枚",
            PALE_GREEN if i < 3 else PALE_ORANGE)
    ax.text(0.5, 0.24, "7,700 × 8 × 3 ＋ 8,000 × 8 × 1", fontsize=22,
            weight="bold", ha="center", va="center", transform=ax.transAxes)
    box(ax, 0.26, 0.025, 0.48, 0.15, "比較指数：248,800")
    footer(fig, "説明用の子の聴牌形。すべて立直を含む。ツモは受け取りの合計。",
           "待ち牌を他家が切る確率、放銃、他家の和了などは、この参考式に含まれない。")
    export(fig, "08-ron-three-tsumo-one-concept")


def rank_reversal(data: list[dict]) -> None:
    fig = base("ツモだけか、ロンを含めるかで比較は逆転する",
               "同じ待ち8枚と仮定した子の例。指数を共通の重みで割って表示", 7.4)
    labels = [x["label"] for x in data]
    for rect, field, heading in (
        ((0.055, 0.29, 0.40, 0.45), "tsumo_normalized", "ツモのみ：指数 ÷ 8"),
        ((0.545, 0.29, 0.40, 0.45), "combined_normalized", "ロン3＋ツモ1：指数 ÷ 32"),
    ):
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 11200)
        ax.set_ylim(-0.45, 1.5)
        ax.set_yticks([])
        ax.set_xticks([0, 4000, 8000], ["0", "4,000", "8,000"])
        ax.tick_params(axis="x", labelsize=13, length=0, pad=8)
        ax.set_title(heading, loc="left", fontsize=17, pad=21)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y, label, item, color in zip([1, 0], labels, data, [GREEN, ORANGE]):
            value = item[field]
            ax.barh(y, value, height=0.39, color=color)
            ax.text(0, y + 0.27, label, fontsize=15.5, ha="left", va="bottom")
            digits = f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"
            ax.text(value + 280, y, digits, fontsize=16.5, weight="bold", va="center")
    fig.text(0.055, 0.19, "ツモのみでは一通の手が上／ロンを3人分加えると平和ドラ2が上",
             fontsize=16.5, weight="bold")
    footer(fig, "実戦の点数期待値ではない。高め4枚・安め4枚、ロン：ツモ＝3：1の参考計算。",
           "裏ドラ・一発・本場等なし、切り上げ満貫なし。pystyleの表示値の再計算ではない。")
    export(fig, "09-ron-tsumo-reference-comparison")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = comparison_data()
    four_channels()
    rank_reversal(data)
    (OUT / "ron-tsumo-reference-data.json").write_text(
        json.dumps({
            "examples": data,
            "assumptions": {
                "seat": "non-dealer", "remaining_wait_tiles": 8,
                "ron_channel_weight": 3, "tsumo_channel_weight": 1,
                "riichi": True, "ura_ippatsu_honba": False, "kiriage": False,
                "purpose": "Hypothetical comparison index, not measured EV or app reproduction.",
            },
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print("Created two reference figure pairs; arithmetic assertions passed.")


if __name__ == "__main__":
    main()
