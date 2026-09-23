import sqlite3
import gzip
import json
import re
from pathlib import Path

SOURCE_PATH = Path(r"outputs\yakuman-db-scan-2009-2025.json")
DB_DIR = Path(r"data\archives\v1.2.0-db")
OUT_PATH = Path(r"outputs\suukantsu-capture-targets.json")

SUUKANTSU = 51

tag_re = re.compile(r"<([A-Z]+[0-9]*)\b([^>]*)>")
attr_re = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')

source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

hits = source["suukantsu"]

entries = []

for hit in hits:
    year = hit["year"]
    log_id = hit["log_id"]

    db_path = DB_DIR / f"{year}.db"

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    row = con.execute("SELECT date, log FROM logs WHERE id = ?", (log_id,)).fetchone()

    con.close()

    if row is None:
        print("NOT FOUND:", log_id)
        continue

    date_text, blob = row

    xml = gzip.decompress(blob).decode("utf-8", errors="replace")

    kyoku_index = -1
    tj_base = -1
    found = False

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

            codes = set()

            if raw:
                try:
                    codes = {int(x) for x in raw.split(",") if x}
                except ValueError:
                    pass

            who = int(attrs.get("who", -1))

            if (
                kyoku_index == hit["kyoku_index"]
                and who == hit["who"]
                and SUUKANTSU in codes
            ):
                entries.append(
                    {
                        "year": year,
                        "date": date_text,
                        "log_id": log_id,
                        "kyoku_index": kyoku_index,
                        "who": who,
                        "fromWho": int(attrs["fromWho"]),
                        "tj_base": tj_base,
                        "base_url": (
                            f"https://tenhou.net/5/"
                            f"?log={log_id}"
                            f"&ts={kyoku_index}"
                            f"&tw={who}"
                        ),
                    }
                )

                found = True
                break

            tj_base += 1
            continue

        tj_base += 1

    if not found:
        print("TARGET NOT FOUND:", log_id)

entries.sort(key=lambda x: (x["year"], x["log_id"]))

OUT_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

print()
print("saved:", OUT_PATH)
print("count:", len(entries))

for i, e in enumerate(entries, 1):
    print(
        i,
        e["year"],
        e["log_id"],
        "kyoku=" + str(e["kyoku_index"]),
        "tj_base=" + str(e["tj_base"]),
    )
