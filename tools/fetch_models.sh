#!/usr/bin/env bash
# Download the pinned default GGUF models into ./models, so a fresh checkout can serve them.
#
# Each model is pinned by Hugging Face repo + quantised file, so a months-old run reproduces
# (CLAUDE.md: pin and document the environment). Downloading needs the huggingface CLI or curl;
# the script fails with an actionable message if neither is available rather than proceeding.
#
# Usage: tools/fetch_models.sh [--dir DIR] [--only NAME]
#   --dir DIR    where to place the GGUFs (default: ./models)
#   --only NAME  fetch only the named model (see the list below); with no --only, fetches every
#                CI/GPU-sized default -- NOT the heavier optional ones (currently: med_author,
#                a ~10.5 GB upgrade for a real med_evidence evaluation), which need --only by name

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

dir="${REPO_ROOT}/models"
only=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) dir="${2:?--dir needs a value}"; shift 2 ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

# name  ->  hf_repo  file  (all Apache-2.0/MIT, un-gated; small enough for a 16 GB GPU or CPU CI --
# except med_author, a ~10.5 GB upgrade for a real med_evidence evaluation, not a CI-sized default;
# fetch it explicitly with --only med_author rather than expecting a bare `fetch_models.sh` to want
# it every time).
#
# Every entry below was verified against the Hugging Face API (repo reachable, exact filename
# present in its file list) rather than assembled from a naming pattern. Two entries once were
# not, and named a Qwen3.5 GGUF family that has never existed at those sizes -- both 401'd, and
# because fetch() aborted on the first failure, the documented one-command setup could not
# complete for five of the six recipes. A plausible-looking name is not a verified one.
declare -A REPO FILE
REPO[ci]="Qwen/Qwen3-0.6B-GGUF";                 FILE[ci]="Qwen3-0.6B-Q8_0.gguf"
REPO[producer]="Qwen/Qwen3-4B-GGUF";             FILE[producer]="Qwen3-4B-Q4_K_M.gguf"
REPO[reviewer]="Qwen/Qwen3-1.7B-GGUF";           FILE[reviewer]="Qwen3-1.7B-Q8_0.gguf"
REPO[embed]="Qwen/Qwen3-Embedding-0.6B-GGUF";    FILE[embed]="Qwen3-Embedding-0.6B-Q8_0.gguf"
REPO[rerank]="gpustack/bge-reranker-v2-m3-GGUF"; FILE[rerank]="bge-reranker-v2-m3-Q8_0.gguf"
REPO[med_author]="Qwen/Qwen3-14B-GGUF";          FILE[med_author]="Qwen3-14B-Q5_K_M.gguf"

# Fetched only when named explicitly via --only -- too large to be part of a bare fetch-everything.
OPTIONAL_NAMES=(med_author)
is_optional() { local n; for n in "${OPTIONAL_NAMES[@]}"; do [[ "$n" == "$1" ]] && return 0; done
               return 1; }

if [[ -n "$only" && -z "${REPO[$only]:-}" ]]; then
    die "unknown model '${only}'. Known: ${!REPO[*]}"
fi

mkdir -p "$dir" || die "could not create ${dir}"

have_hf=0; command -v huggingface-cli >/dev/null 2>&1 && have_hf=1
have_curl=0; command -v curl >/dev/null 2>&1 && have_curl=1
[[ "$have_hf" == 1 || "$have_curl" == 1 ]] || \
    die "need 'huggingface-cli' (pip install huggingface_hub[cli]) or 'curl' to download models."

# Records what failed instead of aborting the run. One unreachable entry used to abort the whole
# fetch, so the models *after* it in the loop were never even attempted and the operator saw one
# error with no idea how much of the setup had actually succeeded. Every entry is now attempted
# and the failures are reported together at the end -- the script still exits non-zero, so a
# caller that checks the status is unaffected.
FAILED=()

fetch() {
    local name="$1" repo="${REPO[$1]}" file="${FILE[$1]}" dest="${dir}/${FILE[$1]}"
    if [[ -f "$dest" ]]; then note "${name}: already have ${file}"; return 0; fi
    note "${name}: fetching ${repo}/${file}"
    if [[ "$have_hf" == 1 ]]; then
        huggingface-cli download "$repo" "$file" --local-dir "$dir" && return 0
    else
        curl -fL --output "$dest" "https://huggingface.co/${repo}/resolve/main/${file}" && return 0
        rm -f "$dest"  # curl -o leaves a truncated/empty file behind on failure
    fi
    warn "${name}: download failed for ${repo}/${file}"
    FAILED+=("${name} (${repo}/${file})")
    return 1
}

if [[ -n "$only" ]]; then
    fetch "$only" || true
else
    for name in "${!REPO[@]}"; do
        is_optional "$name" && { note "${name}: optional, skipped (fetch with --only ${name})"; continue; }
        fetch "$name" || true
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '>> %d model(s) could not be downloaded:\n' "${#FAILED[@]}" >&2
    printf '     - %s\n' "${FAILED[@]}" >&2
    die "check the repo/file names against https://huggingface.co and your network, then re-run \
(already-downloaded models are skipped)."
fi
note "done. Models are in ${dir}; serve them with tools/serve_models.sh --models-dir ${dir}"
