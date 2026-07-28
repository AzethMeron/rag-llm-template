#!/usr/bin/env bash
# Fetch real data for the predictive-maintenance recipe and prepare a held-out decision split:
#   * the MEMORY -- a manuals corpus built from Wikipedia articles on turbofan failure modes
#     (EGT, compressor stall, bearing wear, vibration, ...), fetched via the MediaWiki API (text is
#     CC BY-SA 4.0);
#   * the REQUESTS -- sensor snapshots from NASA's C-MAPSS turbofan degradation dataset (US-Gov
#     public domain), each with a derived operator report, sensor readings, and fault codes, and a
#     gold severity from the engine's remaining useful life.
# Nothing is committed; everything lands under data/.
#
# C-MAPSS ships as a zip of whitespace-delimited text files (train_FD00N.txt). The NASA download is
# behind an unstable link, so pass --cmapss with a train_FD001.txt (or the zip's) location you have.
#
# Usage: recipes/predictive_maintenance/fetch.sh --cmapss PATH_OR_URL [--limit N]
#   --cmapss PATH_OR_URL   a train_FD001.txt file (local path or URL). Required.
#   --limit N              how many held-out snapshots to prepare (default: 600)

source "$(dirname "${BASH_SOURCE[0]}")/../../tools/lib/common.sh"

cmapss=""
limit=600
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cmapss) cmapss="${2:?--cmapss needs a value}"; shift 2 ;;
        --limit) limit="${2:?--limit needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ -n "$cmapss" ]] || die "pass --cmapss with a train_FD001.txt (local path or URL). See https://www.nasa.gov/intelligent-systems-division/ (Prognostics Data Repository, 'Turbofan Engine Degradation')."
[[ "$limit" =~ ^[1-9][0-9]*$ ]] || die "--limit must be a positive integer, got '${limit}'"

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
mkdir -p "$data_dir"
python="$(project_python)"
assert_python_new_enough "$python"

# Resolve the C-MAPSS training file locally.
cmapss_file="${data_dir}/train_FD001.txt"
if [[ -f "$cmapss" ]]; then
    cp "$cmapss" "$cmapss_file"
elif [[ ! -f "$cmapss_file" ]]; then
    require_command curl "Install curl to download the dataset."
    note "downloading C-MAPSS from ${cmapss}"
    curl -fL --output "$cmapss_file" "$cmapss" || die "download failed; check --cmapss."
fi

note "building the manuals memory from Wikipedia (CC BY-SA)"
"$python" - "$data_dir" <<'PY' || die "building the manuals corpus failed (see above)"
import json, sys, urllib.parse, urllib.request
from pathlib import Path

data_dir = Path(sys.argv[1])
titles = ["Turbofan", "Exhaust gas temperature", "Compressor stall", "Foreign object damage",
          "Turbine blade", "Jet engine", "Bearing (mechanical)", "Vibration",
          "Gas turbine", "Aircraft engine controls", "Turbine engine failure"]
api = ("https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts"
       "&explaintext=1&redirects=1&titles=" + urllib.parse.quote("|".join(titles)))
req = urllib.request.Request(api, headers={"User-Agent": "ragkit-predictive-maintenance/1.0"})
with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 -- fixed https host
    pages = json.load(resp)["query"]["pages"]

entries = []
for page in pages.values():
    title, extract = page.get("title", "?"), page.get("extract", "")
    for i, para in enumerate(p.strip() for p in extract.split("\n")):
        if len(para) >= 200:  # skip headings and stubs; keep substantial paragraphs
            entries.append({"id": f"{title}-{i}", "title": title, "text": para})
if len(entries) < 20:
    raise SystemExit(f"only {len(entries)} manual paragraphs fetched; Wikipedia may be unreachable")
with (data_dir / "manuals.jsonl").open("w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
print(f">> {len(entries)} manual paragraphs written to manuals.jsonl")
PY

note "preparing held-out sensor snapshots (up to ${limit})"
"$python" - "$cmapss_file" "$data_dir" "$limit" <<'PY' || die "preparing the C-MAPSS split failed (see above)"
import json, sys
from collections import defaultdict
from pathlib import Path

src, data_dir, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
# C-MAPSS columns: unit, cycle, 3 op settings, then 21 sensors (1-indexed s1..s21).
rows = defaultdict(list)
for line in src.read_text().splitlines():
    parts = line.split()
    if len(parts) < 26:
        continue
    nums = [float(x) for x in parts]
    rows[int(nums[0])].append(nums)
if not rows:
    raise SystemExit("no C-MAPSS rows parsed; is this a train_FD00N.txt file?")

def sev(rul):  # remaining-useful-life -> severity bucket (gold label)
    return "urgent" if rul <= 30 else ("watch" if rul <= 80 else "normal")

heldout, gold = [], []
for unit, cycles in rows.items():
    cycles.sort(key=lambda r: r[1])
    last = cycles[-1][1]
    base = cycles[:5]
    base_egt = sum(r[5] for r in base) / len(base)   # s4 (T50, LPT outlet temp) ~ EGT
    for frac in (0.5, 0.85):                          # a mid-life and a near-end snapshot per unit
        r = cycles[min(len(cycles) - 1, int(len(cycles) * frac))]
        cycle, rul = int(r[1]), last - int(r[1])
        egt, core_speed, fan_speed = round(r[5], 2), round(r[13], 2), round(r[7], 2)  # s4, s12? keep names
        codes = []
        if egt - base_egt > 3:
            codes.append("EGT_HIGH")
        trend = "rising" if egt - base_egt > 1.5 else "steady"
        report = (f"Engine unit {unit} at cycle {cycle}: exhaust gas temperature is {trend}"
                  f" (now {egt}, baseline {round(base_egt, 2)}); core and fan speeds logged."
                  f" Assess wear and recommend maintenance.")
        rid = f"unit{unit}-c{cycle}"
        heldout.append({"record_id": rid, "source": report,
                        "meta": {"fault_codes": codes, "egt": egt, "core_speed": core_speed,
                                 "fan_speed": fan_speed, "cycle": cycle}})
        gold.append({"record_id": rid, "severity": sev(rul), "rul": rul})
        if len(heldout) >= limit:
            break
    if len(heldout) >= limit:
        break

with (data_dir / "heldout.jsonl").open("w", encoding="utf-8") as f:
    for r in heldout:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with (data_dir / "gold.jsonl").open("w", encoding="utf-8") as f:
    for r in gold:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f">> {len(heldout)} held-out snapshots written with gold severity")
PY

note "done. Decide over data/heldout.jsonl with the recipe; score severity against data/gold.jsonl via recipes/predictive_maintenance/eval.py."
