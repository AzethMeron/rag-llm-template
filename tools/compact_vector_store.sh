#!/usr/bin/env bash
# Compact a recipe's [vector] LanceDB table: consolidate the small fragments left by many
# incremental upsert()/delete() calls into a few large ones, and prune old versions.
#
# Not optional polish. A table reconciled/upserted in many small batches over a long, repeatedly-
# resumed embedding job accumulates one fragment per batch without bound, and every subsequent
# read/write re-scans the whole growing fragment list -- confirmed directly against
# legal_procurement's real corpus: 3,717 versions / 1,858 fragments for 2M rows meant simply
# *opening and reconciling against* the table cost multiple GB of RSS per batch, before a single
# new row was embedded. Compacting to 2 fragments fixed that immediately.
#
# tools/embed_reference.sh already runs this periodically during its own run (--compact-every,
# default 50 commits) -- use this script by hand only for a one-off compaction (after disabling
# that with --compact-every 0), or against a store built/written some other way.
#
# Only meaningful for the lancedb driver -- Qdrant's HNSW index has no on-disk fragment-file model
# to compact, so a qdrant-backed [vector] is reported as a no-op rather than an error.
#
# Do not run this against a table another process is actively writing to. LanceDB's behavior under
# that contention has not been verified here beyond one accidental, harmless-seeming case -- treat
# it as unsafe until proven otherwise, not as tolerated.
#
# Usage: tools/compact_vector_store.sh --config DIR
#   --config DIR   a config directory whose storage.toml has a [vector] store

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

config=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?--config needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ -n "$config" ]] || die "--config DIR is required (a directory with storage.toml)"
[[ -f "${config}/storage.toml" ]] || die "no storage.toml in ${config}"

python="$(project_python)"
assert_python_new_enough "$python"
has_module "$python" ragkit || export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"$python" - "$config" <<'PY' || die "compaction failed (see the error above)"
import sys
from pathlib import Path

from ragkit.store import load_storage
from ragkit.store.vector.lancedb import LanceVectorIndex

config_dir = Path(sys.argv[1])
storage = load_storage(config_dir / "storage.toml")
if storage.vector is None:
    raise SystemExit(f"{config_dir}/storage.toml has no [vector] store configured")

if not isinstance(storage.vector, LanceVectorIndex):
    print(f">> {type(storage.vector).__name__} has no on-disk fragments to compact -- no-op")
    raise SystemExit(0)

before = storage.vector.count()
print(f"vector store: {before:,} rows before compaction", flush=True)
storage.vector.compact()
after = storage.vector.count()
print(f">> compacted -- {after:,} rows (unchanged: compaction consolidates files, not data)")
if after != before:
    raise SystemExit(
        f"row count changed across compaction ({before:,} -> {after:,}); this should never "
        f"happen and means compaction lost or duplicated rows")
PY
