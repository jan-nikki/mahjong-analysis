import sqlite3
import gzip
import json
import re
from pathlib import Path

DB_DIR = Path(r"data\archives\v1.2.0-db")
OUT_PATH = Path(r"outputs\ryuuiisou-capture-targets.json")

RYUUIISOU = 43

tag_re = re.compile(r"<([A-Z]+[0-9]*)\b([^>]*)>")
attr_re = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')

entries = []

for year in range(2009, 2026):
    db_path = DB_DIR / f"{year}.db"
    if not db_path.exists():
        continue

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    for log_id, date_text, blob in con.execute(
        "SELECT id, date, log FROM logs WHERE log IS NOT NULL ORDER BY id"
    ):
        try:
            xml = gzip.decompress(blob).decode("utf-8", errors="replace")
        except Exception:
            continue

        kyoku_index = -1
        tj_base = -1

        for m in tag_re.finditer(xml):
            raw_tag = m.group(1)
            attrs = dict(attr_re.findall(m.group(2)))

            if raw_tag == "INIT":
                kyoku_index += 1
                tj_base = 0
                continue

            if kyoku_index < 0:
                continue

            if raw_tag == "AGARI":
                raw = attrs.get("yakuman")
                if raw:
                    try:
                        yakuman_codes = {int(x) for x in raw.split(",") if x}
                    except ValueError:
                        yakuman_codes = set()

                    if RYUUIISOU in yakuman_codes:
                        who = int(attrs["who"])
                        entry = {
                            "year": year,
                            "date": date_text,
                            "log_id": log_id,
                            "kyoku_index": kyoku_index,
                            "who": who,
                            "fromWho": int(attrs["fromWho"]),
                            "tj_base": tj_base,
                            "base_url": f"https://tenhou.net/5/?log={log_id}&ts={kyoku_index}&tw={who}",
                        }
                        entries.append(entry)

                tj_base += 1
                continue

            tj_base += 1

    con.close()

entries.sort(key=lambda x: (x["year"], x["log_id"], x["kyoku_index"]))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(
    json.dumps(entries, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("saved:", OUT_PATH)
print("count:", len(entries))
print("first:", entries[0] if entries else None)
