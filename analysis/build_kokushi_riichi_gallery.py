"""Build readable four-capture strips and a note-ready gallery manuscript."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "kokushi-riichi-outcomes-2009-2025.json"
DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "outputs" / "kokushi-riichi-captures"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kokushi-riichi-gallery"

TILES_JA = {
    "1m": "一萬",
    "9m": "九萬",
    "1p": "一筒",
    "9p": "九筒",
    "1s": "一索",
    "9s": "九索",
    "E": "東",
    "S": "南",
    "W": "西",
    "N": "北",
    "P": "白",
    "F": "發",
    "C": "中",
}
OUTCOME_JA = {
    "kokushi_win": "和了",
    "other_win": "和了できず（他家和了）",
    "draw": "和了できず（流局）",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build gallery strips and a note manuscript for 604 kokushi riichis."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-sheet", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.per_sheet < 1:
        raise ValueError("--per-sheet must be positive")
    document = json.loads(args.input.read_text(encoding="utf-8"))
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("input records must be a non-empty list")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture_paths = [
        args.capture_dir / _capture_filename(index, record)
        for index, record in enumerate(records, 1)
    ]
    missing = [path for path in capture_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} captures; first missing: {missing[0]}"
        )

    font = _font(30)
    sheets: list[dict[str, Any]] = []
    for sheet_index, start in enumerate(range(0, len(records), args.per_sheet), 1):
        group = records[start : start + args.per_sheet]
        paths = capture_paths[start : start + args.per_sheet]
        filename = f"sheet-{sheet_index:03d}.jpg"
        output = args.output_dir / filename
        _build_sheet(output, group, paths, start_index=start + 1, font=font)
        sheets.append(
            {
                "sheet": sheet_index,
                "file": filename,
                "records": [
                    _manifest_record(start + offset + 1, record)
                    for offset, record in enumerate(group)
                ],
            }
        )
        print(f"[{sheet_index}/{math.ceil(len(records) / args.per_sheet)}] {filename}")

    manifest = {
        "title": "狂気の604局。国士無双テンパイでリーチした全牌姿を並べてみた",
        "summary": document["summary"],
        "per_sheet": args.per_sheet,
        "sheets": sheets,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "note-draft.md").write_text(
        _build_markdown(manifest),
        encoding="utf-8",
        newline="\n",
    )
    (args.output_dir / "note-body.html").write_text(
        _build_html(manifest),
        encoding="utf-8",
        newline="\n",
    )
    print(f"sheets: {len(sheets)}")
    print(f"records: {len(records)}")
    print(f"output: {args.output_dir.resolve()}")
    return 0


def _build_sheet(
    output: Path,
    records: list[dict[str, Any]],
    paths: list[Path],
    *,
    start_index: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    header_height = 58
    width, capture_height = 1280, 720
    sheet = Image.new(
        "RGB",
        (width, (capture_height + header_height) * len(records)),
        "#111111",
    )
    draw = ImageDraw.Draw(sheet)
    for offset, (record, path) in enumerate(zip(records, paths, strict=True)):
        number = start_index + offset
        top = offset * (capture_height + header_height)
        outcome = OUTCOME_JA[record["outcome"]]
        wait = (
            "国士13面待ち"
            if record["wait_type"] == "kokushi_13men"
            else f"{TILES_JA[record['waits'][0]]}待ち"
        )
        label = (
            f"No.{number:03d}  {record['date'][:10]}  "
            f"{record['turn']}巡目  {wait}  {outcome}"
        )
        color = "#61d095" if record["won"] else "#ff8f8f"
        draw.rectangle((0, top, width, top + header_height), fill="#111111")
        draw.text((22, top + 10), label, font=font, fill=color)
        capture = Image.open(path).convert("RGB")
        if capture.size != (width, capture_height):
            capture = capture.resize((width, capture_height), Image.Resampling.LANCZOS)
        sheet.paste(capture, (0, top + header_height))
    sheet.save(output, format="JPEG", quality=84, optimize=True, progressive=True)


def _build_markdown(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    win_rate = summary["win_rate"] * 100
    lines = [
        f"# {manifest['title']}",
        "",
        "スクロールしても、スクロールしても、国士無双。",
        "",
        "前回の記事では、天鳳鳳凰卓の2009〜2025年に成立した17,857,520件のリーチから、国士無双テンパイでのリーチを604件見つけました。",
        "",
        "今回は、その604局を全部並べます。抜粋ではありません。本当に全部です。",
        "",
        "各局について、リーチ宣言牌を切った直後の13枚、牌譜URL、その後に国士無双を和了できたかを載せています。",
        "",
        "## 先に結果だけ",
        "",
        f"- 国士リーチ：{summary['total']}件",
        f"- 国士無双を和了：{summary['kokushi_wins']}件（{win_rate:.1f}％）",
        f"- 和了できず：{summary['not_won']}件",
        f"  - 他家が和了：{summary['other_wins']}件",
        f"  - 流局：{summary['draws']}件",
        f"- 和了の内訳：ロン{summary['ron_wins']}件、ツモ{summary['tsumo_wins']}件",
        "",
        "国士無双テンパイでリーチした局の和了率は47.0％でした。",
        "",
        "## 見方",
        "",
        "画像は上から番号順です。緑の見出しが和了、赤の見出しが非和了です。読みやすさを落とさず記事の画像数を抑えるため、4局を縦につないで1枚にしています。牌姿は604局すべて掲載しています。",
        "",
        "それでは、ここから先は604局連続です。",
        "",
        "## 国士無双リーチ604局、全部",
        "",
    ]
    for sheet in manifest["sheets"]:
        first_index = sheet["records"][0]["index"]
        last_index = sheet["records"][-1]["index"]
        lines.extend(
            (
                f"【No.{first_index:03d}〜{last_index:03d}】",
                f"画像：`{sheet['file']}`",
                "",
            )
        )
        for record in sheet["records"]:
            method = ""
            if record["won"]:
                method = "・ロン" if record["win_method"] == "ron" else "・ツモ"
            lines.append(
                f"No.{record['index']:03d}｜{record['outcome_ja']}{method}｜"
                f"[牌譜を見る]({record['url']})"
            )
        lines.append("")
    lines.extend(
        (
            "## 604局を並べて分かること",
            "",
            "数字だけでは見えなかったものが、牌姿を並べると急に具体的になります。早い巡目でほぼ配牌のままリーチする局もあれば、終盤まで么九牌を拾い続けた局もあります。13面待ちも、字牌単騎も、数牌単騎もあります。",
            "",
            "同時に、これだけ並べても604局しかありません。元になった成立リーチは17,857,520件です。画面上では途方もない量に見える604局が、全体では約29,565リーチに1回しかない。その落差も、この一覧の面白さだと思います。",
            "",
            "次回は、点数状況・残り枚数・フリテン・リーチ後の他家の動きまで追い、『なぜ国士無双でリーチしたのか』を分析します。",
            "",
        )
    )
    return "\n".join(lines)


def _build_html(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    win_rate = summary["win_rate"] * 100
    parts = [
        '<!doctype html><html lang="ja"><meta charset="utf-8"><body>',
        "<p>スクロールしても、スクロールしても、国士無双。</p>",
        "<p>前回の記事では、天鳳鳳凰卓の2009〜2025年に成立した17,857,520件のリーチから、国士無双テンパイでのリーチを604件見つけました。</p>",
        "<p>今回は、その604局を全部並べます。抜粋ではありません。本当に全部です。</p>",
        "<p>各局について、リーチ宣言牌を切った直後の13枚、牌譜URL、その後に国士無双を和了できたかを載せています。</p>",
        "<h2>先に結果だけ</h2>",
        "<ul>",
        f"<li>国士リーチ：{summary['total']}件</li>",
        f"<li>国士無双を和了：{summary['kokushi_wins']}件（{win_rate:.1f}％）</li>",
        f"<li>和了できず：{summary['not_won']}件</li>",
        f"<li>他家が和了：{summary['other_wins']}件</li>",
        f"<li>流局：{summary['draws']}件</li>",
        f"<li>和了の内訳：ロン{summary['ron_wins']}件、ツモ{summary['tsumo_wins']}件</li>",
        "</ul>",
        "<p>国士無双テンパイでリーチした局の和了率は47.0％でした。</p>",
        "<h2>見方</h2>",
        "<p>画像は上から番号順です。緑の見出しが和了、赤の見出しが非和了です。読みやすさを落とさず記事の画像数を抑えるため、4局を縦につないで1枚にしています。牌姿は604局すべて掲載しています。</p>",
        "<p>それでは、ここから先は604局連続です。</p>",
        "<h2>国士無双リーチ604局、全部</h2>",
    ]
    for sheet in manifest["sheets"]:
        first_index = sheet["records"][0]["index"]
        last_index = sheet["records"][-1]["index"]
        parts.append(f"<p>【No.{first_index:03d}〜{last_index:03d}】</p>")
        lines = []
        for record in sheet["records"]:
            method = ""
            if record["won"]:
                method = "・ロン" if record["win_method"] == "ron" else "・ツモ"
            label = html.escape(
                f"No.{record['index']:03d}｜{record['outcome_ja']}{method}｜"
            )
            url = html.escape(record["url"], quote=True)
            lines.append(f'{label}<a href="{url}">牌譜を見る</a>')
        parts.append("<p>" + "<br>".join(lines) + "</p>")
    parts.extend(
        (
            "<h2>604局を並べて分かること</h2>",
            "<p>数字だけでは見えなかったものが、牌姿を並べると急に具体的になります。早い巡目でほぼ配牌のままリーチする局もあれば、終盤まで么九牌を拾い続けた局もあります。13面待ちも、字牌単騎も、数牌単騎もあります。</p>",
            "<p>同時に、これだけ並べても604局しかありません。元になった成立リーチは17,857,520件です。画面上では途方もない量に見える604局が、全体では約29,565リーチに1回しかない。その落差も、この一覧の面白さだと思います。</p>",
            "<p>次回は、点数状況・残り枚数・フリテン・リーチ後の他家の動きまで追い、『なぜ国士無双でリーチしたのか』を分析します。</p>",
            "</body></html>",
        )
    )
    return "\n".join(parts)


def _manifest_record(index: int, record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value["index"] = index
    value["outcome_ja"] = OUTCOME_JA[record["outcome"]]
    value["date_ja"] = _date_ja(record["date"])
    return value


def _date_ja(value: str) -> str:
    date = datetime.fromisoformat(value)
    return f"{date.year}年{date.month}月{date.day}日"


def _capture_filename(index: int, record: dict[str, Any]) -> str:
    return (
        f"{index:03d}_{record['year']}_{record['log_id']}"
        f"_k{record['kyoku_index']:02d}.jpg"
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\meiryo.ttc"),
        Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size, index=0)
    return ImageFont.load_default()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
