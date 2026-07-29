#!/usr/bin/env bash
# Fetch real data for the legal-procurement passage-retrieval recipe and prepare the memory plus a
# held-out set of questions with gold relevance judgments:
#   * the MEMORY -- the polqa passage corpus (IPIPAN, CC BY-SA): Polish Wikipedia passages used as a
#     stand-in legal/encyclopaedic corpus. The full passages.jsonl is ~3.3 GB, so --max-passages
#     bounds how many of its first lines are loaded (default 200000). Each kept passage is written to
#     data/passages.jsonl as {id, text} with a stable id "ref-<line>", which is the id a citation
#     refers to and the id the retrieval eval scores against.
#   * the QUESTIONS + GOLD -- polqa's test.csv: each row pairs a question with a relevant passage.
#     data/heldout.jsonl holds one {record_id, source, meta} per question; data/gold.jsonl holds
#     {record_id, relevant:[passage ids]} grouping the gold passages per question. Only questions
#     whose gold passage is inside the loaded subset are kept (so Recall is computable); the count
#     kept is reported.
#
# For a real legal corpus instead, swap the memory for EUR-Lex or ELI-Sejm (https://api.sejm.gov.pl/
# eli) documents in the same {id, text} JSONL shape; the recipe is corpus-agnostic.
#
# Nothing is committed; everything lands under data/ (gitignored).
#
# Usage: recipes/legal_procurement/fetch.sh [--max-passages N] [--limit N]
#   --max-passages N   how many of passages.jsonl's first lines to load as the memory (default:
#                      200000). The full file is ~3.3 GB; this flag bounds it but can scale big.
#   --limit N          cap on how many held-out questions to keep (default: 100000)

source "$(dirname "${BASH_SOURCE[0]}")/../../tools/lib/common.sh"

max_passages=200000
limit=100000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-passages) max_passages="${2:?--max-passages needs a value}"; shift 2 ;;
        --limit) limit="${2:?--limit needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ "$max_passages" =~ ^[1-9][0-9]*$ ]] || die "--max-passages must be a positive integer, got '${max_passages}'"
[[ "$limit" =~ ^[1-9][0-9]*$ ]] || die "--limit must be a positive integer, got '${limit}'"

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
mkdir -p "$data_dir"
python="$(project_python)"
assert_python_new_enough "$python"
require_command curl "Install curl to download the polqa dataset."

base_url="https://huggingface.co/datasets/ipipan/polqa/resolve/main/data/"

note "downloading the gold test split (test.csv)"
curl -fL --output "${data_dir}/test.csv" "${base_url}test.csv" \
    || die "downloading test.csv failed; polqa/HuggingFace may be unreachable."

# The passage-building + gold-splitting program, streamed the 3.3 GB passages.jsonl line by line so
# it is never held in memory. Passed via `python -c` (not a heredoc into stdin) precisely because
# stdin is the curl pipe here.
read -r -d '' build_script <<'PY' || true
import csv, json, sys
from pathlib import Path

data_dir, max_passages, limit = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

# Stream the corpus: assign each written passage the id "ref-<n>" that the framework's lexical
# corpus loader will also assign it (it numbers non-blank lines from 1), so a gold passage id and a
# citation both refer to the same thing. Keep a map from the polqa id to that id for the gold split.
id_of = {}
written = 0
with (data_dir / "passages.jsonl").open("w", encoding="utf-8") as out:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        title = str(record.get("title", "")).strip()
        body = str(record.get("text", "")).strip()
        text = f"{title}\n{body}" if title else body
        if not text:
            continue
        written += 1
        ref = f"ref-{written}"
        id_of[str(record["id"])] = ref
        out.write(json.dumps({"id": ref, "text": text}, ensure_ascii=False) + "\n")
        if written >= max_passages:
            break

if written == 0:
    raise SystemExit("no passages read from stdin; the passages.jsonl download may have failed")

# Group test.csv rows into one question each, collecting the ids of its relevant passages that are
# inside the loaded subset. A question with none is dropped -- Recall would be uncomputable for it.
groups = {}
with (data_dir / "test.csv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        qid = row["question_id"]
        group = groups.setdefault(qid, {"question": row["question"], "relevant": set()})
        if not group["question"].strip():
            group["question"] = row["question"]
        if row["relevant"].strip().lower() == "true":
            ref = id_of.get(row["passage_id"])
            if ref is not None:
                group["relevant"].add(ref)

kept = [(qid, g) for qid, g in groups.items() if g["relevant"] and g["question"].strip()]
kept = kept[:limit]

with (data_dir / "heldout.jsonl").open("w", encoding="utf-8") as hf, \
     (data_dir / "gold.jsonl").open("w", encoding="utf-8") as gf:
    for qid, g in kept:
        rid = f"q{qid}"
        relevant = sorted(g["relevant"], key=lambda r: int(r.split("-", 1)[1]))
        hf.write(json.dumps({"record_id": rid, "source": g["question"], "meta": {}},
                            ensure_ascii=False) + "\n")
        gf.write(json.dumps({"record_id": rid, "relevant": relevant}, ensure_ascii=False) + "\n")

print(f">> {written} passages loaded; kept {len(kept)}/{len(groups)} questions with in-subset gold")
PY

note "streaming up to ${max_passages} passages and building the held-out gold split"
# curl may be sent SIGPIPE when the reader stops early at --max-passages; that is expected, not an
# error, so pipefail is off here and the pipeline's status is the Python program's own.
set +o pipefail
curl -fL "${base_url}passages.jsonl" \
    | "$python" -c "$build_script" "$data_dir" "$max_passages" "$limit" \
    || die "building passages.jsonl and the gold split failed (see above)"
set -o pipefail

note "done. Memory in data/passages.jsonl; retrieve over data/heldout.jsonl with the recipe and score retrieval against data/gold.jsonl via recipes/legal_procurement/eval.py."
