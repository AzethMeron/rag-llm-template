#!/usr/bin/env bash
# Build an ANN index on a recipe's [vector] LanceDB table so search() switches from an O(n)
# brute-force scan of every row to sub-linear approximate search.
#
# Not optional polish. LanceDB's search() silently falls back to a full-table scan whenever no
# index exists -- correct, but at millions of rows a single query then reads the *entire* vector
# column. Confirmed directly against legal_procurement's real corpus: a 7.1M-row/1024-d table with
# no index turned a 956-query dense-retrieval eval into a multi-hour, CPU-pegged, 0%-GPU run that
# produced no progress output, because every query rescanned all 7.1M vectors (~29GB) from scratch.
#
# Only meaningful for the lancedb driver -- Qdrant's HNSW index is built automatically as part of
# its own upsert path, so a qdrant-backed [vector] is reported as a no-op rather than an error.
#
# Do not run this against a table another process is actively writing to. This is not a theoretical
# caution: two embed-job workers accidentally writing to the same legal_procurement table
# concurrently (this repo's own incident) silently produced 5,000 duplicate rows -- same id, two
# rows each -- with no error raised anywhere; it surfaced only much later via an exact
# pairings.count() == vector.count() audit. A clean exit is not evidence nothing broke here.
#
# Usage: tools/build_vector_index.sh --config DIR [--num-partitions N]
#   --config DIR         a config directory whose storage.toml has a [vector] store
#   --num-partitions N   IVF partition count (default: sqrt(row count), the standard IVF heuristic)

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

config=""
num_partitions=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?--config needs a value}"; shift 2 ;;
        --num-partitions) num_partitions="${2:?--num-partitions needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ -n "$config" ]] || die "--config DIR is required (a directory with storage.toml)"
[[ -f "${config}/storage.toml" ]] || die "no storage.toml in ${config}"
if [[ -n "$num_partitions" && ! "$num_partitions" =~ ^[1-9][0-9]*$ ]]; then
    die "--num-partitions must be a positive integer, got '${num_partitions}'"
fi

python="$(project_python)"
assert_python_new_enough "$python"
has_module "$python" ragkit || export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"$python" - "$config" "$num_partitions" <<'PY' || die "index build failed (see the error above)"
import sys
from pathlib import Path

from ragkit.store import load_storage
from ragkit.store.vector.lancedb import LanceVectorIndex

config_dir = Path(sys.argv[1])
num_partitions = int(sys.argv[2]) if sys.argv[2] else None

storage = load_storage(config_dir / "storage.toml")
if storage.vector is None:
    raise SystemExit(f"{config_dir}/storage.toml has no [vector] store configured")

if not isinstance(storage.vector, LanceVectorIndex):
    print(f">> {type(storage.vector).__name__} has no separate ANN-index build step -- no-op")
    raise SystemExit(0)

rows = storage.vector.count()
if rows == 0:
    raise SystemExit(f"{config_dir}: the vector store is empty -- nothing to index")
print(f"vector store: {rows:,} rows -- building IVF_FLAT index "
      f"({num_partitions or 'sqrt(rows)'} partitions)...", flush=True)
storage.vector.create_index(num_partitions=num_partitions)
after = storage.vector.count()
print(f">> index built -- {after:,} rows (unchanged: indexing does not modify data)")
if after != rows:
    raise SystemExit(
        f"row count changed across index build ({rows:,} -> {after:,}); this should never "
        f"happen and means indexing lost or duplicated rows")
PY
