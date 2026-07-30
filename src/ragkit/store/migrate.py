"""One-time migrations from the legacy JSONL/split-store artifacts to the DB-native stores this
project now uses (see ``docs/storage-overhaul-plan.md``).

Each function reads an old artifact and writes a new store, streaming in batches so a multi-GB
corpus never sits fully in RAM. None of them touch or delete the old artifact — leaving it in place
is the caller's safety net until the new one is verified (``tools/migrate_storage.sh`` is the
hardened wrapper that actually runs these against a recipe's data).

Folding a :class:`~ragkit.store.documents.sqlite.SqliteDocuments` row store into a
:class:`~ragkit.core.ports.PairingStore` deliberately does **not** touch the matching
:class:`~ragkit.store.lexical.fts5.Fts5Index` file at all: the new pairing store builds its own
FTS5 index from the same source text as pairings are added (see
:mod:`ragkit.store.pairings.sqlite`), so the old search index carries nothing the migration needs —
only the row data (chunk id, display text, meta) does.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

from ragkit.core.ports import LexiconStore, Pairing, PairingStore, RunResult, RunStore
from ragkit.core.lexicon import read_lexicon
from ragkit.core.records import read_catalog, read_journal


def _iter_legacy_documents(path: Path) -> Iterator[tuple[str, str, dict]]:
    """Stream ``(chunk_id, display, meta)`` rows from a legacy ``SqliteDocuments`` database
    (schema: ``docs(chunk_id, display, meta)``), in original insertion order, never materialising
    the whole table in memory. Opened read-only: a migration must never risk writing the source
    it is reading from."""
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT chunk_id, display, meta FROM docs ORDER BY rowid")
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                return
            for chunk_id, display, meta in rows:
                yield chunk_id, display, json.loads(meta)
    finally:
        conn.close()


def migrate_documents_to_pairings(documents_path: Path, pairing_store: PairingStore, *,
                                  batch_size: int = 5000,
                                  on_batch: Callable[[int], None] | None = None) -> int:
    """Fold a legacy ``DocumentStore``'s rows into ``pairing_store`` as source-only pairings (no
    target/context — a lexical reference entry never had either). Resumable exactly like
    :func:`~ragkit.ingest.reference.import_reference`: the floor is ``pairing_store.count()``, and
    since both the source table and the destination preserve insertion order (``ref-<line>``
    numbering), skipping the first ``floor`` rows continues an interrupted migration rather than
    restarting it. Returns the number of pairings actually added.

    ``on_batch``, when given, is called after each committed batch with the destination's new total
    row count (``pairing_store.count()``, cheaply tracked rather than re-queried) — a multi-million
    row migration run unsupervised needs some sign of life beyond "still running".
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    floor = pairing_store.count()
    total = 0
    batch: list[Pairing] = []
    for index, (chunk_id, display, meta) in enumerate(_iter_legacy_documents(documents_path)):
        if index < floor:
            continue
        batch.append(Pairing(chunk_id=chunk_id, source=display, meta=meta))
        if len(batch) >= batch_size:
            total += pairing_store.add(batch)
            batch = []
            if on_batch is not None:
                on_batch(floor + total)
    if batch:
        total += pairing_store.add(batch)
        if on_batch is not None:
            on_batch(floor + total)
    return total


def migrate_lexicon_to_store(lexicon_path: Path, lexicon_store: LexiconStore) -> int:
    """Import a JSONL lexicon (see :func:`~ragkit.core.lexicon.read_lexicon`) into a
    :class:`~ragkit.core.ports.LexiconStore`. A missing file is "no established terms" — the same
    legitimate empty state ``read_lexicon`` itself treats it as — so this is a no-op, not an
    error."""
    return lexicon_store.add(read_lexicon(lexicon_path))


def migrate_run_to_store(catalog_path: Path, journal_path: Path, run_store: RunStore) -> int:
    """Fold a legacy catalogue + journal pair into a ``RunStore``: add every catalogue record,
    then replay the journal's results in write order, so the store's own later-wins-by-``seq``
    resolution reproduces exactly what the journal already encoded (its last line for a given
    record was the one that mattered). The structured violations/reviews/captured context a
    ``RunResult`` can hold are left at their defaults — the journal never recorded them, so there
    is nothing to migrate into those fields. Returns the number of records added."""
    added = run_store.add_records(read_catalog(catalog_path))
    for record in read_journal(journal_path):
        run_store.append_result(RunResult(record=record))
    return added
