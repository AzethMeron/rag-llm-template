#!/usr/bin/env bash
# Fetch the Tatoeba English-Polish sentence pairs and build the translation-memory corpus plus a
# held-out set to translate and score against. Nothing is committed; the data lands under data/.
#
# Source: Tatoeba tab-separated bilingual pairs (CC-BY 2.0 FR), via manythings.org's Anki export.
# A shortfall (fewer usable pairs than asked) is an error, never a quietly smaller sample.
#
# Usage: recipes/translation/fetch.sh [--pair pol] [--heldout N] [--url URL]
#   --pair CODE    the ManyThings pair archive (default: pol, i.e. English-Polish)
#   --heldout N    how many held-out lines to reserve for evaluation (default: 200)
#   --url URL      override the download URL

source "$(dirname "${BASH_SOURCE[0]}")/../../tools/lib/common.sh"

pair="pol"
heldout=200
url=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pair) pair="${2:?--pair needs a value}"; shift 2 ;;
        --heldout) heldout="${2:?--heldout needs a value}"; shift 2 ;;
        --url) url="${2:?--url needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ "$heldout" =~ ^[1-9][0-9]*$ ]] || die "--heldout must be a positive integer, got '${heldout}'"
[[ -n "$url" ]] || url="https://www.manythings.org/anki/${pair}-eng.zip"

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
mkdir -p "$data_dir"
python="$(project_python)"
assert_python_new_enough "$python"

archive="${data_dir}/${pair}-eng.zip"
if [[ ! -f "$archive" ]]; then
    require_command curl "Install curl to download the dataset."
    note "downloading ${url}"
    curl -fL --output "$archive" "$url" \
        || die "download failed. Tatoeba/ManyThings may be unreachable; pass --url with a mirror, or place ${pair}-eng.txt in ${data_dir} yourself."
fi

note "extracting and splitting (heldout=${heldout})"
"$python" - "$archive" "$data_dir" "$heldout" <<'PY' || die "building the corpus failed (see above)"
import json, sys, zipfile
from pathlib import Path

archive, data_dir, heldout = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
with zipfile.ZipFile(archive) as zf:
    name = next(n for n in zf.namelist() if n.endswith(".txt"))
    lines = zf.read(name).decode("utf-8").splitlines()

pairs = []
for line in lines:
    cols = line.split("\t")
    if len(cols) >= 2 and cols[0].strip() and cols[1].strip():
        pairs.append((cols[0].strip(), cols[1].strip()))  # (English, Polish)

if len(pairs) < heldout + 100:
    raise SystemExit(f"only {len(pairs)} usable pairs, need at least {heldout + 100}")

# Deterministic split by a stable hash, so the same held-out set reproduces across runs.
import hashlib
def bucket(en): return int(hashlib.sha1(en.encode("utf-8")).hexdigest(), 16) % 1000
pairs.sort(key=lambda p: (bucket(p[0]), p[0]))
held, corpus = pairs[:heldout], pairs[heldout:]

with (data_dir / "reference.jsonl").open("w", encoding="utf-8") as f:
    for en, pl in corpus:
        f.write(json.dumps({"source": en, "target": pl}, ensure_ascii=False) + "\n")
with (data_dir / "heldout.jsonl").open("w", encoding="utf-8") as f:
    for i, (en, _pl) in enumerate(held):
        f.write(json.dumps({"record_id": f"held-{i}", "source": en}, ensure_ascii=False) + "\n")
with (data_dir / "gold.jsonl").open("w", encoding="utf-8") as f:
    for i, (en, pl) in enumerate(held):
        f.write(json.dumps({"record_id": f"held-{i}", "source": en, "target": pl},
                           ensure_ascii=False) + "\n")
print(f">> {len(corpus)} reference pairs, {len(held)} held-out lines")
PY

note "done. Reference memory in data/reference.jsonl; translate data/heldout.jsonl and score against data/gold.jsonl."
