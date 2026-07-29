#!/usr/bin/env bash
# Fetch real data for the biomedical evidence-intelligence recipe and prepare a held-out decision
# split:
#   * the MEMORY -- a corpus of biomedical abstracts, built from PubMedQA's expert-labelled set
#     (ori_pqal.json, MIT-licensed), and optionally scaled with real ClinicalTrials.gov v2 study
#     summaries (US-Gov public domain) so the corpus can grow to many GB;
#   * the REQUESTS -- the PubMedQA research questions (yes/no/maybe), each with a gold final decision
#     kept OUT of the prompt and written only to the gold file.
# Nothing is committed; everything lands under data/.
#
# PubMedQA is a JSON dict keyed by PMID; each value has QUESTION, CONTEXTS (the abstract), and
# final_decision (the gold -- never placed in any prompt). ClinicalTrials.gov v2 is paginated via a
# nextPageToken; each study contributes a title + brief summary + conditions as extra corpus.
#
# Usage: recipes/med_evidence/fetch.sh [--limit N] [--trials N]
#   --limit N    cap the number of PubMedQA questions prepared (default: 0 = all, ~1000)
#   --trials N   append N real ClinicalTrials.gov study summaries to the corpus (default: 0 = skip)

source "$(dirname "${BASH_SOURCE[0]}")/../../tools/lib/common.sh"

limit=0
trials=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) limit="${2:?--limit needs a value}"; shift 2 ;;
        --trials) trials="${2:?--trials needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ "$limit" =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer, got '${limit}'"
[[ "$trials" =~ ^[0-9]+$ ]] || die "--trials must be a non-negative integer, got '${trials}'"

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
mkdir -p "$data_dir"
python="$(project_python)"
assert_python_new_enough "$python"
require_command curl "Install curl to download the datasets."

pqal="${data_dir}/ori_pqal.json"
if [[ ! -f "$pqal" ]]; then
    note "downloading PubMedQA (ori_pqal.json, MIT)"
    curl -fL --output "$pqal" \
        "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json" \
        || die "PubMedQA download failed; check network connectivity."
fi

note "building the abstracts memory and held-out questions from PubMedQA (limit ${limit})"
"$python" - "$pqal" "$data_dir" "$limit" <<'PY' || die "building the PubMedQA split failed (see above)"
import json, sys
from pathlib import Path

src, data_dir, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
records = json.loads(src.read_text(encoding="utf-8"))
if not isinstance(records, dict) or not records:
    raise SystemExit("PubMedQA ori_pqal.json is not a non-empty JSON object; is the download intact?")

items = list(records.items())
if limit > 0:
    items = items[:limit]

abstracts, heldout, gold = [], [], []
for pmid, value in items:
    question = str(value.get("QUESTION", "")).strip()
    contexts = value.get("CONTEXTS") or []
    abstract = " ".join(str(c).strip() for c in contexts if str(c).strip())
    decision = str(value.get("final_decision", "")).strip().lower()
    if not question or not abstract or decision not in {"yes", "no", "maybe"}:
        continue  # skip malformed rows rather than emit a request with no memory or no gold
    abstracts.append({"id": str(pmid), "text": abstract})
    heldout.append({"record_id": str(pmid), "source": question, "meta": {"pmid": str(pmid)}})
    gold.append({"record_id": str(pmid), "decision": decision})  # gold ONLY here, never in a prompt

if len(heldout) < 1:
    raise SystemExit("no usable PubMedQA rows parsed; is ori_pqal.json the expert-labelled set?")

with (data_dir / "abstracts.jsonl").open("w", encoding="utf-8") as f:
    for e in abstracts:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
with (data_dir / "heldout.jsonl").open("w", encoding="utf-8") as f:
    for r in heldout:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with (data_dir / "gold.jsonl").open("w", encoding="utf-8") as f:
    for r in gold:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f">> {len(abstracts)} abstracts, {len(heldout)} questions, {len(gold)} gold labels written")
PY

if [[ "$trials" -gt 0 ]]; then
    note "appending up to ${trials} ClinicalTrials.gov study summaries to the corpus (public domain)"
    "$python" - "$data_dir" "$trials" <<'PY' || die "fetching ClinicalTrials.gov summaries failed (see above)"
import json, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

data_dir, want = Path(sys.argv[1]), int(sys.argv[2])
# A stable sort key makes the nextPageToken chain deterministic and complete; without it the
# API occasionally drops the token mid-traversal, silently truncating a deep fetch.
base = ("https://clinicaltrials.gov/api/v2/studies?pageSize=1000"
        "&sort=LastUpdatePostDate"
        "&fields=NCTId,BriefTitle,BriefSummary,Condition")

def fetch(token):
    url = base + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
    req = urllib.request.Request(url, headers={
        "User-Agent": "ragkit-med-evidence/1.0 (recipe corpus fetcher)"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 -- fixed https host
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or attempt == 5:
                raise
            retry_after = exc.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 5 * 2 ** attempt
            time.sleep(min(wait, 120))
    raise SystemExit("ClinicalTrials.gov kept failing after retries")

written, token = 0, None
with (data_dir / "abstracts.jsonl").open("a", encoding="utf-8") as f:
    while written < want:
        page = fetch(token)
        for study in page.get("studies", []):
            section = study.get("protocolSection", {})
            nct = section.get("identificationModule", {}).get("nctId")
            title = section.get("identificationModule", {}).get("briefTitle", "")
            summary = section.get("descriptionModule", {}).get("briefSummary", "")
            conditions = section.get("conditionsModule", {}).get("conditions", []) or []
            if not nct:
                continue
            text = f"{title}. {summary} Conditions: {', '.join(conditions)}".strip()
            f.write(json.dumps({"id": str(nct), "text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= want:
                break
        token = page.get("nextPageToken")
        if not token:
            break  # ran out of studies before reaching the requested count
        time.sleep(1)  # be a polite API citizen between pages
print(f">> {written} ClinicalTrials.gov summaries appended to abstracts.jsonl")
PY
fi

note "done. Answer over data/heldout.jsonl with the recipe; score decisions against data/gold.jsonl via recipes/med_evidence/eval.py."
