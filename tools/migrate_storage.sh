#!/usr/bin/env bash
# Migrate a recipe's pre-overhaul on-disk artifacts (a legacy relational row store, a
# lexicon.jsonl, or a catalog+journal pair) into the new DB-native stores this project uses
# (PairingStore / LexiconStore / RunStore -- see docs/storage-overhaul-plan.md). The legacy split
# store itself (that row store plus its separate FTS5 index) has been retired from the live
# framework; this script's job is purely to bring forward data from before that retirement.
#
# Never touches or deletes the source artifact: each migration is a pure additive row-copy into the
# destination database, safe to interrupt and re-run (resumable/idempotent -- see
# src/ragkit/store/migrate.py, which this wraps). Only point a recipe's storage.toml at the
# destination once you've verified the migration finished and its row count looks right.
#
# Usage:
#   tools/migrate_storage.sh documents --documents PATH --pairings-db PATH \
#       [--pairings-driver sqlite|duckdb] [--batch-size N]
#   tools/migrate_storage.sh lexicon --lexicon PATH --lexicon-db PATH
#   tools/migrate_storage.sh run --catalog PATH --journal PATH --run-db PATH
#
#   documents   fold a legacy row store's (chunk_id, display, meta) rows into a pairing store as
#               source-only pairings. Does NOT read the matching .fts5 index -- the destination
#               builds its own search index as rows are added.
#   lexicon     import a JSONL lexicon into a lexicon store.
#   run         fold a JSONL catalogue + journal into a run store (records + replayed results).
#               --journal may point at a file that does not exist yet (nothing has completed).

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

[[ $# -ge 1 ]] || die "a subcommand is required: documents | lexicon | run (see --help)"
subcommand="$1"; shift
[[ "$subcommand" == "-h" || "$subcommand" == "--help" ]] && { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

documents="" pairings_db="" pairings_driver="sqlite" batch_size="5000"
lexicon="" lexicon_db="" catalog="" journal="" run_db=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --documents) documents="${2:?--documents needs a value}"; shift 2 ;;
        --pairings-db) pairings_db="${2:?--pairings-db needs a value}"; shift 2 ;;
        --pairings-driver) pairings_driver="${2:?--pairings-driver needs a value}"; shift 2 ;;
        --batch-size) batch_size="${2:?--batch-size needs a value}"; shift 2 ;;
        --lexicon) lexicon="${2:?--lexicon needs a value}"; shift 2 ;;
        --lexicon-db) lexicon_db="${2:?--lexicon-db needs a value}"; shift 2 ;;
        --catalog) catalog="${2:?--catalog needs a value}"; shift 2 ;;
        --journal) journal="${2:?--journal needs a value}"; shift 2 ;;
        --run-db) run_db="${2:?--run-db needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

python="$(project_python)"
assert_python_new_enough "$python"
has_module "$python" ragkit || export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

case "$subcommand" in
    documents)
        [[ -n "$documents" ]] || die "documents: --documents PATH is required"
        [[ -n "$pairings_db" ]] || die "documents: --pairings-db PATH is required"
        [[ -f "$documents" ]] || die "documents: no such file: $documents"
        [[ "$pairings_driver" == "sqlite" || "$pairings_driver" == "duckdb" ]] \
            || die "documents: --pairings-driver must be 'sqlite' or 'duckdb', got '$pairings_driver'"
        [[ "$batch_size" =~ ^[0-9]+$ && "$batch_size" -ge 1 ]] \
            || die "documents: --batch-size must be a positive integer, got '$batch_size'"
        mkdir -p "$(dirname "$pairings_db")"
        "$python" - "$documents" "$pairings_db" "$pairings_driver" "$batch_size" <<'PY' \
            || die "documents migration failed (see the error above)"
import sqlite3
import sys
import time
from pathlib import Path

from ragkit.store import PAIRING_STORES
from ragkit.store.migrate import migrate_documents_to_pairings

documents_path = Path(sys.argv[1])
pairings_path, driver, batch_size = sys.argv[2], sys.argv[3], int(sys.argv[4])

with sqlite3.connect(f"file:{documents_path.resolve()}?mode=ro", uri=True) as conn:
    total_source = conn.execute("SELECT count(*) FROM docs").fetchone()[0]
pairing_store = PAIRING_STORES.create(driver, {"path": pairings_path})
start = time.monotonic()


def report(processed: int) -> None:
    elapsed = time.monotonic() - start
    print(f"  {processed:,}/{total_source:,} rows migrated ({elapsed:,.0f}s elapsed)", flush=True)


added = migrate_documents_to_pairings(documents_path, pairing_store, batch_size=batch_size,
                                      on_batch=report)
print(f">> {added:,} pairing(s) added to {pairings_path} ({pairing_store.count():,} total)")
PY
        ;;
    lexicon)
        [[ -n "$lexicon" ]] || die "lexicon: --lexicon PATH is required"
        [[ -n "$lexicon_db" ]] || die "lexicon: --lexicon-db PATH is required"
        [[ -f "$lexicon" ]] || die "lexicon: no such file: $lexicon"
        mkdir -p "$(dirname "$lexicon_db")"
        "$python" - "$lexicon" "$lexicon_db" <<'PY' \
            || die "lexicon migration failed (see the error above)"
import sys
from pathlib import Path

from ragkit.store.lexicon.sqlite import SqliteLexicon
from ragkit.store.migrate import migrate_lexicon_to_store

lexicon_path, store_path = Path(sys.argv[1]), sys.argv[2]
store = SqliteLexicon(store_path)
added = migrate_lexicon_to_store(lexicon_path, store)
print(f">> {added:,} lexicon entr{'y' if added == 1 else 'ies'} added to {store_path}")
PY
        ;;
    run)
        [[ -n "$catalog" ]] || die "run: --catalog PATH is required"
        [[ -n "$journal" ]] || die "run: --journal PATH is required"
        [[ -n "$run_db" ]] || die "run: --run-db PATH is required"
        [[ -f "$catalog" ]] || die "run: no such file: $catalog"
        mkdir -p "$(dirname "$run_db")"
        "$python" - "$catalog" "$journal" "$run_db" <<'PY' \
            || die "run migration failed (see the error above)"
import sys
from pathlib import Path

from ragkit.store.run.sqlite import SqliteRunStore
from ragkit.store.migrate import migrate_run_to_store

catalog_path, journal_path, run_db_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
store = SqliteRunStore(run_db_path)
added = migrate_run_to_store(catalog_path, journal_path, store)
print(f">> {added:,} record(s) added to {run_db_path} ({len(store.completed_ids()):,} completed)")
PY
        ;;
    *) die "unknown subcommand '$subcommand' (expected: documents | lexicon | run)" ;;
esac
