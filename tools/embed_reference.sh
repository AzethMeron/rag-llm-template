#!/usr/bin/env bash
# Embed a [pairings] store's rows into its [vector] index, so dense/hybrid retrieval has vectors
# to search -- the vector-side counterpart to import (which only ever builds the lexical/BM25
# side; [reference].file -> [pairings] happens automatically on assemble(), but nothing embeds
# the corpus into [vector] until something calls VectorIndex.reconcile(), which this does).
#
# Resumable/idempotent: reconciles against what's already embedded (VectorIndex.reconcile drops
# orphans and reports what's missing) and only embeds the gap, so interrupting and re-running
# costs nothing and duplicates nothing.
#
# A long, many-times-resumed run on a lancedb [vector] store accumulates one on-disk fragment per
# upsert -- run tools/compact_vector_store.sh periodically against the same --config (e.g. every
# few hundred thousand rows), not just once at the end. An uncompacted table with thousands of
# fragments costs multiple GB of RSS per subsequent batch here, confirmed directly at real corpus
# scale, before this script embeds a single new row.
#
# Usage: tools/embed_reference.sh --config DIR --embedding-url URL [--embedding-model NAME]
#                                 [--batch-size N] [--http-batch-size N]
#   --config DIR            a config directory whose storage.toml has [pairings] AND [vector]
#   --embedding-url URL     the embedding endpoint's base URL (e.g. http://127.0.0.1:8081/v1)
#   --embedding-model NAME  the model name sent in each request (default: local)
#   --batch-size N          pairings embedded + upserted per commit (default: 4000)
#   --http-batch-size N     texts sent per embedding HTTP request (default: 1000 -- tune down for
#                           a slower/smaller-context server, up if the server batches well)

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

config="" embedding_url="" embedding_model="local" batch_size="4000" http_batch_size="1000"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?--config needs a value}"; shift 2 ;;
        --embedding-url) embedding_url="${2:?--embedding-url needs a value}"; shift 2 ;;
        --embedding-model) embedding_model="${2:?--embedding-model needs a value}"; shift 2 ;;
        --batch-size) batch_size="${2:?--batch-size needs a value}"; shift 2 ;;
        --http-batch-size) http_batch_size="${2:?--http-batch-size needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ -n "$config" ]] || die "--config DIR is required (a directory with storage.toml)"
[[ -f "${config}/storage.toml" ]] || die "no storage.toml in ${config}"
[[ -n "$embedding_url" ]] || die "--embedding-url URL is required"
[[ "$batch_size" =~ ^[0-9]+$ && "$batch_size" -ge 1 ]] \
    || die "--batch-size must be a positive integer, got '$batch_size'"
[[ "$http_batch_size" =~ ^[0-9]+$ && "$http_batch_size" -ge 1 ]] \
    || die "--http-batch-size must be a positive integer, got '$http_batch_size'"

python="$(project_python)"
assert_python_new_enough "$python"
has_module "$python" ragkit || export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"$python" - "$config" "$embedding_url" "$embedding_model" "$batch_size" "$http_batch_size" <<'PY' \
    || die "embedding failed (see the error above)"
import sys
import time
from pathlib import Path

from ragkit.store import load_storage
from ragkit.retrieve.embedding import EmbeddingClient
from ragkit.ingest.reference import reconcile_vector

config_dir = Path(sys.argv[1])
embedding_url, embedding_model = sys.argv[2], sys.argv[3]
batch_size, http_batch_size = int(sys.argv[4]), int(sys.argv[5])

storage = load_storage(config_dir / "storage.toml")
if storage.pairings is None:
    raise SystemExit(f"{config_dir}/storage.toml has no [pairings] store configured")
if storage.vector is None:
    raise SystemExit(f"{config_dir}/storage.toml has no [vector] store configured")

embedder = EmbeddingClient(base_url=embedding_url, model=embedding_model,
                           batch_size=http_batch_size, timeout_seconds=180.0)

print(f"pairings: {storage.pairings.count():,} rows; vector store: {storage.vector.count():,} "
      f"rows before reconcile", flush=True)

start = time.monotonic()
def _progress(done: int, total: int) -> None:
    if done == 0:
        print(f"{total:,} passages need embedding", flush=True)
        return
    elapsed = time.monotonic() - start
    rate = done / elapsed if elapsed > 0 else 0
    eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
    print(f"  {done:,}/{total:,} embedded ({elapsed:,.0f}s, {rate:.1f}/s, "
          f"eta {eta_min:,.1f}min)", flush=True)

reconcile_vector(storage.pairings, storage.vector, embedder, batch_size=batch_size,
                 on_batch=_progress)
print(f">> vector store now has {storage.vector.count():,} rows")
PY
