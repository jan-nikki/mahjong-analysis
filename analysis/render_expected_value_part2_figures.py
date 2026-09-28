"""Render diagrams and archived performance data for the second AI article.

No mahjong rules or decision engine is implemented here. Archived results do
not estimate the strength of current AI versions.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import render_expected_value_part1_figures as shared

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/articles/assets/expected-value-part2"


def naga_yearly_results() -> None:
    """Display the previously verified ranked-game counts, not current models."""
    source = ROOT / "docs/research/naga25-ranked-yearly.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    cells = []
    totals = dict.fromkeys(("games", "first", "second", "third", "fourth"), 0)
    for record in records:
        counts = {key: int(record[key]) for key in totals}
        assert counts["games"] == sum(counts[key] for key in ("first", "second", "third", "fourth"))
        for key, value in counts.items():
            totals[key] += value
        if counts["games"]:
            average = sum(i * counts[key] for i, key in enumerate(("first", "second", "third", "fourth"), 1)) / counts["games"]
            assert math.isclose(average, float(record["average_placement"]), abs_tol=1e-8)
            stable = float(record["stable_dan"])
            if record["stable_method"] == "tokujou_formula":
                check = (5 * counts["first"] + 2 * counts["second"]) / counts["fourth"] - 2
                assert math.isclose(stable, check, abs_tol=1e-8)
            mark = "*" if record["stable_method"] == "nodocchi_mixed_rooms_display" else ""
            cells.append([f'NAGA {record["year"]}', f'{counts["games"]:,}', f"{average:.4f}", f"{stable:.4f}{mark}", "教師あり†"])
    # Each of these years has the same zero ranked-game count.
    assert all(int(record["games"]) == 0 for record in records if int(record["year"]) >= 2022)
    cells.append(["NAGA\n2022〜2026各年", "0", "—", "—", "—"])
    assert totals["games"] == 26_598
    average = sum(i * totals[key] for i, key in enumerate(("first", "second", "third", "fourth"), 1)) / totals["games"]
    # Mixed-room stable rank is the archived website value, not a new formula.
    cells.append(["NAGA 通算\n（一般・上級含む）", f'{totals["games"]:,}', f"{average:.4f}", "7.2217*", "教師あり†"])
    # Counts independently recorded in docs/research/mahjong-ai-tenhou-comparison.md.
    tokujou_counts = (7070, 7227, 6709, 5397)
    tokujou_games = sum(tokujou_counts)
    tokujou_average = sum(i * count for i, count in enumerate(tokujou_counts, 1)) / tokujou_games
    tokujou_stable = (5 * tokujou_counts[0] + 2 * tokujou_counts[1]) / tokujou_counts[3] - 2
    cells.append(["NAGA 通算\n（特上卓東南戦）", f"{tokujou_games:,}", f"{tokujou_average:.4f}", f"{tokujou_stable:.4f}", "教師あり†"])
    cells.extend([
        ["LuckyJ\n2023年公表", "1,000超‡", "未確認", "10.68", "強化学習＋\n後悔最小化"],
        ["Suphx（フェニックス）\n2020年論文", "5,760", "未確認", "8.74", "教師あり→\n強化学習§"],
        ["Mortal", "未確認", "未確認", "未確認", "深層強化学習"],
        ["MAKA（マカ）", "未確認", "未確認", "未確認", "未確認"],
    ])
    fig = shared.base("麻雀AIの公開対局成績", "NAGAは年別、ほかのAIは公表資料の集計／四人麻雀", 12.7)
    ax = fig.add_axes((0.055, 0.23, 0.89, 0.57))
    ax.set_axis_off()
    table = ax.table(cellText=cells, colLabels=["AI・集計対象", "対局数", "平均順位", "安定段位", "学習方式"],
                     colWidths=[0.27, 0.14, 0.16, 0.14, 0.29], cellLoc="center", bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(shared.GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(shared.GREEN)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(16.5)
        else:
            cell.set_facecolor(shared.PALE_GREEN if row in (4, 6, 7) else (shared.PALE_ORANGE if row in (8, 9) else ("white" if row % 2 else shared.BG)))
            if row in (4, 6, 7, 8, 9):
                cell.get_text().set_weight("bold")
            if col == 0:
                cell.get_text().set_fontsize(14.0)
            if col == 4:
                cell.get_text().set_fontsize(15.5)
            if col == 1 and row == 8:
                cell.get_text().set_fontsize(15.5)
    for y, text in (
        (0.195, "NAGAの行はⓝNAGA25。2019〜2021年は特上卓東南戦。"),
        (0.168, "* 2018年・一般／上級を含む通算は、集計サイトの安定段位表示値。"),
        (0.141, "† 教師ありが基本。複数版の履歴で、現行5タイプの成績ではない。"),
        (0.114, "‡ LuckyJの件数は公表文の「1,000戦超」。厳密な件数は未確認。"),
        (0.087, "§ Suphxの強化学習は打牌モデル。鳴き・立直・槓は教師あり版。"),
        (0.060, "時期・標本数・相手が異なる。現行AIの強さの順位表ではない。"),
        (0.033, "確認：2026年9月26日。NAGAの0戦は段位戦のみ。未確認は0戦と異なる。"),
    ):
        fig.text(0.055, y, text, fontsize=13.0, color=shared.MUTED, va="top")
    shared.export(fig, "03-mahjong-ai-results")


def learning_pathways() -> None:
    fig = shared.base(
        "AIは、何を手掛かりに学ぶ？",
        "人間の行動を手本にする方法と、報酬から改善する方法",
        8.0,
    )
    ax = fig.add_axes((0.055, 0.17, 0.89, 0.61))
    ax.set_axis_off()
    for x, heading, top, middle, bottom, fill in (
        (0.04, "教師あり学習", "人間の牌譜など", "局面と行動を手本に\nモデルを学習", "局面から\n選ばれやすい行動を推定", shared.PALE_GREEN),
        (0.54, "強化学習（自己対局の例）", "対局して行動する", "結果・報酬をもとに\n方策や価値を改善", "学んだ方策や価値で\n次の行動を選ぶ", shared.PALE_ORANGE),
    ):
        ax.text(x + 0.21, 0.97, heading, ha="center", va="top", fontsize=18,
                weight="bold", transform=ax.transAxes)
        shared.box(ax, x, 0.69, 0.42, 0.15, top, fill=fill)
        shared.arrow(ax, (x + 0.21, 0.67), (x + 0.21, 0.57))
        shared.box(ax, x, 0.36, 0.42, 0.18, middle, fill=fill)
        shared.arrow(ax, (x + 0.21, 0.34), (x + 0.21, 0.24))
        shared.box(ax, x, 0.02, 0.42, 0.20, bottom, fill=fill)
    shared.footer(fig,
        "概念図。教師あり学習と強化学習を組み合わせるAIもある。",
        "右側は自己対局を使う例。強化学習のすべてが自己対局とは限らない。")
    shared.export(fig, "01-learning-pathways")


def recommendation_to_understanding() -> None:
    fig = shared.base(
        "推奨から、自分の判断へ",
        "理由を仮説として考え、条件の違いを調べて学習につなげる",
        8.0,
    )
    ax = fig.add_axes((0.055, 0.17, 0.89, 0.61))
    ax.set_axis_off()
    shared.box(ax, 0.04, 0.60, 0.40, 0.27, "1．推奨を確認",
               "モデル・ルール・点況を\n揃えて読む", shared.PALE_GREEN)
    shared.box(ax, 0.56, 0.60, 0.40, 0.27, "2．理由の仮説",
               "打点・受け入れ・安全性\n何を重視したか考える", shared.PALE_ORANGE)
    shared.box(ax, 0.56, 0.09, 0.40, 0.27, "3．条件を比較",
               "似た局面でドラや点況が\n変わるとどうなるか調べる", shared.PALE_ORANGE)
    shared.box(ax, 0.04, 0.09, 0.40, 0.27, "4．自分の判断へ",
               "学んだ条件を言葉に残し\n次の実戦で使う", shared.PALE_GREEN)
    shared.arrow(ax, (0.46, 0.73), (0.53, 0.73))
    shared.arrow(ax, (0.76, 0.57), (0.76, 0.39))
    shared.arrow(ax, (0.53, 0.22), (0.46, 0.22))
    shared.footer(fig,
        "概念図。人間が付けた説明は、AIの実際の判断根拠と一致するとは限らない。",
        "分からない条件は問いとして残し、比較や検討を続ける。")
    shared.export(fig, "02-recommendation-to-understanding")


def record_and_stable_rank() -> None:
    fig = shared.base("最高記録と、期間内の成績を分けて読む", "最高到達段位と安定段位は、違うことを表している", 7.0)
    ax = fig.add_axes((0.055, 0.18, 0.89, 0.61))
    ax.set_axis_off()
    shared.box(ax, 0.025, 0.48, 0.42, 0.40, "最高到達段位", "一度、どこまで\n昇段したか", shared.PALE_GREEN)
    shared.box(ax, 0.555, 0.48, 0.42, 0.40, "安定段位", "ある期間に\nどんな着順を残したか", shared.PALE_ORANGE)
    # A summit symbol, not a measured or simulated rank trajectory.
    ax.plot([0.07, 0.18, 0.235, 0.30, 0.40], [0.18, 0.25, 0.36, 0.25, 0.18], color=shared.GREEN, linewidth=4, transform=ax.transAxes)
    ax.scatter([0.235], [0.36], s=100, color=shared.GREEN, transform=ax.transAxes)
    ax.text(0.235, 0.04, "最高点を見る", ha="center", fontsize=19, weight="bold", transform=ax.transAxes)
    for i, label in enumerate(("1位", "2位", "3位", "4位")):
        shared.box(ax, 0.56 + i * 0.105, 0.23, 0.09, 0.13, label, fill=shared.PALE_ORANGE)
    ax.text(0.765, 0.12, "着順と段位pt配分から計算", ha="center", fontsize=15.5, transform=ax.transAxes)
    ax.text(0.765, 0.04, "期間の成績を見る", ha="center", fontsize=19, weight="bold", transform=ax.transAxes)
    shared.footer(fig, "最高到達が十段でも、安定段位が十段とは限らない。", "概念図。山形の線は最高点のイメージで、実際の段位推移ではない。")
    shared.export(fig, "04-record-and-stable-rank")


def supervised_features() -> None:
    fig = shared.base("教師ありAIは、局面の特徴を学ぶ", "人間の牌譜を手本に、初めての局面でも行動を推定する", 6.7)
    ax = fig.add_axes((0.055, 0.19, 0.89, 0.58))
    ax.set_axis_off()
    for x, heading, body, fill in (
        (0.025, "強者の牌譜", "局面と\n選んだ行動", shared.PALE_GREEN),
        (0.365, "特徴を学ぶ", "手牌・河・点況\nなどの関係", shared.PALE_ORANGE),
        (0.705, "新しい局面", "選ばれそうな\n行動を推定", shared.PALE_GREEN),
    ):
        shared.box(ax, x, 0.32, 0.27, 0.48, heading, body, fill)
    shared.arrow(ax, (0.305, 0.56), (0.345, 0.56))
    shared.arrow(ax, (0.645, 0.56), (0.685, 0.56))
    ax.text(0.5, 0.10, "まったく同じ牌姿が、学習元になくても判断できる", ha="center", fontsize=20, weight="bold", transform=ax.transAxes)
    shared.footer(fig, "概念図。同じ局面で何人がその牌を切ったかを、単純に数える多数決とは異なる。", "教師あり学習という名前だけで、一打の最適性や戦術の限界は決まらない。")
    shared.export(fig, "05-supervised-features")


def choice_and_explanation() -> None:
    fig = shared.base("AIの選択と、人間の説明を分ける", "納得できる説明も、まずは仮説として確かめる", 7.2)
    ax = fig.add_axes((0.055, 0.18, 0.89, 0.61))
    ax.set_axis_off()
    shared.box(ax, 0.025, 0.57, 0.40, 0.31, "AIの選択", "この一打が有力", shared.PALE_GREEN)
    shared.box(ax, 0.575, 0.57, 0.40, 0.31, "人間の説明", "安全牌を残すため？", shared.PALE_ORANGE)
    ax.plot([0.455, 0.545], [0.72, 0.72], color=shared.MUTED, linewidth=2.5, linestyle="--", transform=ax.transAxes)
    ax.text(0.5, 0.43, "この理由は、まだ仮説", ha="center", fontsize=21, weight="bold", transform=ax.transAxes)
    shared.arrow(ax, (0.5, 0.37), (0.5, 0.29))
    shared.box(ax, 0.10, 0.02, 0.80, 0.24, "似た局面で、条件を変えて比較", "ドラ・巡目・点況など", shared.PALE_GREEN)
    shared.footer(fig, "人間が付けた説明は、AIが実際に使った判断根拠と一致するとは限らない。", "概念図。強い手を選ぶ能力と、理由を説明する能力は分けて考える。")
    shared.export(fig, "06-choice-and-explanation")


def similarity_scope() -> None:
    fig = shared.base("「類似度100％」の読み方", "類似度・悪手率・一致率／2024年10月のサポート回答", 7.0)
    ax = fig.add_axes((0.055, 0.18, 0.89, 0.61))
    ax.set_axis_off()
    shared.box(ax, 0.025, 0.46, 0.42, 0.42, "指標の対象", "打牌選択", shared.PALE_GREEN)
    shared.box(ax, 0.555, 0.46, 0.42, 0.42, "指標の対象外", "鳴き・立直・槓\nの判断", shared.PALE_ORANGE)
    ax.text(0.5, 0.20, "100％でも、すべての行動が\n同じだったとは限らない", ha="center", va="center", fontsize=24, weight="bold", linespacing=1.6, transform=ax.transAxes)
    shared.footer(fig, "NAGAサポートからの2024年10月3日付の回答に基づく。", "回答時点の対象範囲を示す図。現在の仕様は別途確認が必要。")
    shared.export(fig, "07-similarity-scope")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shared.OUT = OUT
    shared.setup_style()
    learning_pathways()
    recommendation_to_understanding()
    naga_yearly_results()
    record_and_stable_rank()
    supervised_features()
    choice_and_explanation()
    similarity_scope()
    print(f"Rendered seven figure pairs in {OUT}")


if __name__ == "__main__":
    main()
