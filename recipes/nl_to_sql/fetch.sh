#!/usr/bin/env bash
# Fetch the Spider text-to-SQL dataset and prepare one database plus its held-out questions and
# gold SQL. Spider ships real SQLite databases, so the recipe queries an actual schema. Nothing is
# committed; everything lands under data/.
#
# Spider (Yale, CC BY-SA 4.0). The archive is large and hosted on Google Drive / Hugging Face; pass
# --url with the spider.zip location you have access to (the default points at the HF mirror).
#
# Usage: recipes/nl_to_sql/fetch.sh --url URL [--db NAME]
#   --url URL   location of spider.zip (or a directory already containing the extracted dataset)
#   --db NAME   which Spider database to use (default: concert_singer, a small clean one)

source "$(dirname "${BASH_SOURCE[0]}")/../../tools/lib/common.sh"

url=""
db="concert_singer"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url) url="${2:?--url needs a value}"; shift 2 ;;
        --db) db="${2:?--db needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
mkdir -p "$data_dir"
python="$(project_python)"
assert_python_new_enough "$python"

extracted="${data_dir}/spider"
if [[ ! -d "$extracted" ]]; then
    [[ -n "$url" ]] || die "Spider is not present. Pass --url with a spider.zip location (Yale/HuggingFace), or extract the dataset into ${extracted} yourself. See https://yale-lily.github.io/spider"
    require_command curl "Install curl to download the dataset."
    if [[ -d "$url" ]]; then
        note "using extracted dataset at ${url}"; extracted="$url"
    else
        note "downloading ${url}"
        curl -fL --output "${data_dir}/spider.zip" "$url" || die "download failed; check --url."
        require_command unzip "Install unzip to extract the archive."
        unzip -q -o "${data_dir}/spider.zip" -d "$data_dir" || die "could not unzip spider.zip"
    fi
fi

note "preparing database '${db}'"
"$python" - "$extracted" "$data_dir" "$db" <<'PY' || die "preparing the dataset failed (see above)"
import json, shutil, sys
from pathlib import Path

spider, data_dir, db = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sqlite = spider / "database" / db / f"{db}.sqlite"
if not sqlite.is_file():
    raise SystemExit(f"no SQLite file for database {db!r} at {sqlite}")
shutil.copy(sqlite, data_dir / "database.sqlite")

dev = spider / "dev.json"
records = json.loads(dev.read_text("utf-8")) if dev.is_file() else []
held = [r for r in records if r.get("db_id") == db]
if not held:
    raise SystemExit(f"no dev questions for database {db!r}; pick another with --db")

with (data_dir / "heldout.jsonl").open("w", encoding="utf-8") as f:
    for i, r in enumerate(held):
        f.write(json.dumps({"record_id": f"q-{i}", "source": r["question"]},
                           ensure_ascii=False) + "\n")
with (data_dir / "gold.jsonl").open("w", encoding="utf-8") as f:
    for i, r in enumerate(held):
        f.write(json.dumps({"record_id": f"q-{i}", "source": r["question"],
                            "sql": r.get("query", "")}, ensure_ascii=False) + "\n")
print(f">> database {db} copied; {len(held)} held-out questions with gold SQL")
PY

note "done. Point recipes/nl_to_sql/config/storage.toml at data/database.sqlite (already the default). Translate data/heldout.jsonl; score against data/gold.jsonl."
