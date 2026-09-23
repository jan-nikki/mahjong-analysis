import json
from datetime import datetime
from pathlib import Path

INPUT_PATH = Path(r"outputs\ryuuiisou-capture-targets.json")
OUT_PATH = Path(r"outputs\ryuuiisou-note-list.md")

data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

lines = []

for i, item in enumerate(data, 1):
    dt = datetime.fromisoformat(item["date"])

    date_jp = f"{dt.year}年{dt.month}月{dt.day}日"

    filename = f"{i:03d}_{item['year']}_{item['log_id']}_k{item['kyoku_index']:02d}.png"

    url = (
        f"https://tenhou.net/5/"
        f"?log={item['log_id']}"
        f"&ts={item['kyoku_index']}"
        f"&tw={item['who']}"
    )

    lines.append(f"### No.{i:03d}｜{date_jp}")
    lines.append("")
    lines.append(f"画像：`{filename}`")
    lines.append("")
    lines.append(f"→ [天鳳で牌譜を見る]({url})")
    lines.append("")
    lines.append("")

OUT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("saved:", OUT_PATH)
print("count:", len(data))
