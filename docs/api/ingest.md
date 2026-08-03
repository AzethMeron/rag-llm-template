# ragkit.ingest

`ragkit.ingest` turns a raw source into indexable chunks and, separately, imports a reference
corpus into the framework's reference memory. The pipeline is extract -> normalise -> dedup ->
chunk -> embed -> import: extractors turn bytes/text into `Document`s, `normalise` cleans text,
`dedup` drops exact repeats, chunkers split a document into `Chunk`s with provenance, and
`reference.py` streams a JSONL corpus into a `PairingStore` (optionally embedding into a
`VectorIndex`) and hands back retrievers over it. `writeback.py` closes the loop, folding a
finished run's verified outputs back into that same reference memory. Depends on
`ragkit.retrieve`, `ragkit.store`, and `ragkit.core`; never on the harness.

**Ordering constraint when composing these yourself.** `normalise`'s default
`collapse_whitespace=True` folds every whitespace run — including the blank line between
paragraphs — to a single space. `StructureChunker`, the default chunker, finds its boundaries by
splitting on exactly that blank-line pattern, so normalising *before* chunking leaves it one giant
paragraph: it silently degrades to whole-document sentence packing, with no error. Chunk **before**
you collapse whitespace (normalise each chunk's text afterward instead), or pass
`collapse_whitespace=False` up front. `FixedChunker` (blind to structure) and `SentenceChunker`
(splits on sentence punctuation, which survives collapsing) are unaffected.

## ragkit.ingest.chunk

Chunkers split a `Document` into indexable `Chunk`s, each carrying its provenance in `meta`
(`document_id`, `ordinal`, `char_start`/`char_end`, `section_path` when known, `chunker`). The
default, `StructureChunker`, is structure-aware — it respects paragraph and heading boundaries and
splits an oversized element only as a fallback — because chunking quality is understood to
dominate the choice of ANN index; `FixedChunker` and `SentenceChunker` ship as measurable
alternatives. All three are registered on the module-level `CHUNKERS: Registry[Chunker]` under the
names `"fixed"`, `"sentence"`, `"structure"` (entry-point group `ragkit.chunkers`), so a recipe
selects one by name or dotted path without editing code.

Every chunk's `char_start`/`char_end` are true offsets into `document.text`, found by scanning
forward from the previous piece's end (the module's private `_locate` helper) rather than derived
from a reconstructed, single-space-joined string. This is a correctness-sensitive detail: offsets
computed from a reconstructed string are positions in that reconstruction, not in the original
text, so every span after the first piece would be wrong the moment the source had a paragraph
break or a run of whitespace between pieces — and any citation/highlight feature slicing
`document.text[char_start:char_end]` would show the wrong text. `FixedChunker` never has this
failure mode: it slices `document.text` directly, so its offsets are trivially correct.

### ChunkError(RagkitError)

Raised when a chunker is misconfigured (e.g. `FixedChunker`/`SentenceChunker`/`StructureChunker`
constructed with out-of-range `size`/`overlap`/`target`/`maximum`), or when a chunker produces a
piece that cannot be located as a substring of the source document (a programming error in a
chunker — refusing to record a wrong provenance span is preferred to silently recording one).

### FixedChunker()

Implements the `Chunker` port (`ragkit.core.ports.Chunker`). Fixed-size character windows with
overlap: the simple, predictable baseline, fast but blind to structure, so an answer can straddle a
window boundary. `CONFIG_KEYS = {"size", "overlap"}`.

Constructor: `FixedChunker(size: int = 1000, overlap: int = 150)`. Raises `ChunkError` if
`size < 1` or `overlap` is not in `[0, size)`.

#### from_config(cls, options: Mapping[str, Any]) -> FixedChunker

**Args:** `options` — `{"size": int = 1000, "overlap": int = 150}`, both read through the strict
`ragkit.core.config.read_int` reader (not a bare `int(...)` coercion; `read_int` also rejects a
`bool`, so `size = true` cannot read as `1`).
**Returns:** a new `FixedChunker`.
**Raises:** `ConfigError` if `size`/`overlap` is present but not an integer (`read_int`); `ChunkError`
if the resulting `size`/`overlap` violate the constructor's constraints.

#### chunk(self, document: Document) -> Iterator[Chunk]

Slides a window of `size` characters across `document.text`, advancing `size - overlap` characters
each step; a window that is all-whitespace is skipped (no chunk emitted, but the ordinal counter
still advances only on a real yield). The window's `char_start`/`char_end` are the true slice
bounds, so no substring search is needed here. The last window is emitted once
`start + size >= len(text)`, then the loop stops; an empty or whitespace-only document yields no
chunks at all.

**Args:** `document` — the document to split.
**Returns:** an iterator of `Chunk`, ordinals starting at 0, `meta["chunker"] = "fixed"`.
**Raises:** nothing; pure and lazy (nothing is computed until iterated).

### SentenceChunker()

Implements the `Chunker` port. Packs whole sentences up to a target size, so a chunk never ends
mid-sentence. `CONFIG_KEYS = {"target"}`.

Constructor: `SentenceChunker(target: int = 800)`. Raises `ChunkError` if `target < 1`.

#### from_config(cls, options: Mapping[str, Any]) -> SentenceChunker

**Args:** `options` — `{"target": int = 800}`, read through the strict `read_int` reader (not a bare
`int(...)` coercion).
**Returns:** a new `SentenceChunker`.
**Raises:** `ConfigError` if `target` is present but not an integer; `ChunkError` if the resulting
`target < 1`.

#### chunk(self, document: Document) -> Iterator[Chunk]

Splits `document.text` on sentence-ending punctuation followed by whitespace (regex
`(?<=[.!?])\s+`), then packs the resulting sentences greedily: append a sentence to the current
buffer unless doing so would push its length past `target`, in which case the buffer is flushed as
a chunk first. Buffered pieces are joined with a single space for the chunk's `text`; the chunk's
`char_start`/`char_end` span from the first piece's true start to the last piece's true end in
`document.text` (found via the same forward-scanning locate step `StructureChunker` uses).

**Args:** `document` — the document to split.
**Returns:** an iterator of `Chunk`, `meta["chunker"] = "sentence"`.
**Raises:** `ChunkError` if a produced sentence cannot be located as a substring of
`document.text` at or after the previous piece's end — not expected in normal use since the
splitter cuts the text itself, but possible from a pathological input or a subclass override.

### StructureChunker()

Implements the `Chunker` port. The default chunker: respects paragraph (and, in effect, blank-line
section) boundaries, packing whole paragraphs up to a `target` size; a single paragraph larger than
`maximum` is split by sentence as a fallback rather than emitted as one oversized chunk.
`CONFIG_KEYS = {"target", "maximum"}`.

Constructor: `StructureChunker(target: int = 600, maximum: int = 1200)`. Raises `ChunkError`
unless `1 <= target <= maximum`.

#### from_config(cls, options: Mapping[str, Any]) -> StructureChunker

**Args:** `options` — `{"target": int = 600, "maximum": int = 1200}`, both read through the strict
`read_int` reader (not a bare `int(...)` coercion).
**Returns:** a new `StructureChunker`.
**Raises:** `ConfigError` if `target`/`maximum` is present but not an integer; `ChunkError` if
`1 <= target <= maximum` does not hold.

#### chunk(self, document: Document) -> Iterator[Chunk]

Splits `document.text` on blank lines (regex `\n\s*\n`) into paragraphs; a paragraph longer than
`maximum` is itself split into sentences (regex `(?<=[.!?])\s+`) as a fallback so no single chunk
ever exceeds the packer's per-piece input unbounded. The resulting elements are packed greedily up
to `target`, exactly as `SentenceChunker` packs sentences: a piece is appended to the buffer unless
that would push the buffer's length past `target`, in which case the buffer is flushed first and a
new one started. Buffered pieces are joined with a single space; `char_start`/`char_end` are the
true offsets of the first and last piece in `document.text`, found by scanning forward from the
previous piece's end — never from a reconstructed string, which is what makes citing
`document.text[char_start:char_end]` reliable.

**Args:** `document` — the document to split.
**Returns:** an iterator of `Chunk`, `meta["chunker"] = "structure"`.
**Raises:** `ChunkError` if a produced paragraph/sentence piece cannot be located as a substring of
`document.text` at or after the previous piece's end (see `ChunkError` above).

## ragkit.ingest.dedup

Exact-duplicate removal for documents and chunks before indexing, so a document ingested twice or
a repeated boilerplate paragraph does not waste storage or let a retriever surface several
near-identical hits that crowd a prompt. This is the cheap, exact half: a content hash over the
text **exactly as given**, byte for byte, computed deterministically. Normalising first is the
caller's choice, not this module's — two texts differing only in whitespace are *not*
deduplicated here. Near-duplicate detection (MinHash/SimHash) is a heavier, separate concern left
to a plugin.

#### content_hash(text: str) -> str

The public, stable exact-dedup key: a SHA-256 hex digest of `text.encode("utf-8")`. (The
in-process `seen` sets inside `dedup_documents`/`dedup_chunks` use the raw 32-byte digest instead
of this hex form, since nothing outside the loop reads it and the hex string would cost about
twice the memory per entry for no benefit.)

**Args:** `text` — the exact text to hash.
**Returns:** a 64-character lowercase hex string.
**Raises:** nothing.

#### dedup_documents(documents: Iterable[Document]) -> Iterator[Document]

**Args:** `documents` — an iterable of `Document`, consumed lazily.
**Returns:** an iterator yielding each document in its original order, dropping any whose `.text`
is byte-identical to an earlier document's.
**Raises:** nothing.
**Side effects:** none; holds one SHA-256 digest per distinct text seen so far in memory for the
lifetime of the iteration.

#### dedup_chunks(chunks: Iterable[Chunk]) -> Iterator[Chunk]

Same behavior as `dedup_documents`, over `Chunk.text` instead of `Document.text`.

**Args:** `chunks` — an iterable of `Chunk`, consumed lazily.
**Returns:** an iterator yielding each chunk in order, dropping any whose `.text` duplicates an
earlier chunk's.
**Raises:** nothing.

## ragkit.ingest.extract

Extractors turn a raw source (bytes or `str`) into `Document` objects ready to chunk. Shipped are
the light, dependency-free extractors the framework's recipes need: plain text, JSON Lines, and
structure-aware HTML/Markdown via the standard library's `html.parser` (no binary parsers, no
third-party HTML/PDF libraries). A binary format (PDF/DOCX/email) is documented as a plugin:
register a component under the `Extractor` port and select it by dotted path. All four ship
registered on the module-level `EXTRACTORS: Registry[Extractor]` under `"text"`, `"jsonl"`,
`"html"`, `"markdown"` (entry-point group `ragkit.extractors`).

### ExtractError(RagkitError)

Raised when a source cannot be extracted: invalid JSON on a JSONL line, a JSONL record missing its
configured text or id field, a text field that is present but not a string, or a null id field.

### TextExtractor()

Implements the `Extractor` port (`ragkit.core.ports.Extractor`). The whole source as one
plain-text document. `CONFIG_KEYS = {"doc_id"}`.

Constructor: `TextExtractor(doc_id: str = "text")`.

#### from_config(cls, options: Mapping[str, Any]) -> TextExtractor

**Args:** `options` — `{"doc_id": str = "text"}`.
**Returns:** a new `TextExtractor`.
**Raises:** `ConfigError` if `doc_id` is present but not a string (read through the strict
`ragkit.core.config.read_string` reader); otherwise nothing.

#### extract(self, source: bytes | str, *, meta: Mapping[str, Any] = {}) -> Iterator[Document]

Decodes `source` as UTF-8 if it is `bytes`, and yields exactly one `Document` whose `text` is the
whole decoded source, `meta` is `dict(meta)`, and `doc_id` is `meta["doc_id"]` if present,
otherwise the extractor's configured `doc_id`.

**Args:** `source` — the raw source; `meta` — caller-supplied metadata, merged into the document's
`meta` and consulted for an overriding `doc_id`.
**Returns:** an iterator yielding exactly one `Document`.
**Raises:** nothing (no `UnicodeDecodeError` handling here — invalid UTF-8 bytes propagate
uncaught).

### JsonlExtractor()

Implements the `Extractor` port. One document per JSON Lines record, taking the document text from
a named field. `CONFIG_KEYS = {"field", "id_field"}`.

Constructor: `JsonlExtractor(field: str = "text", id_field: str = "")`. An empty `id_field` (the
default) means synthesize `doc_id` as `f"line-{line_no}"` instead of reading it from the record.

#### from_config(cls, options: Mapping[str, Any]) -> JsonlExtractor

**Args:** `options` — `{"field": str = "text", "id_field": str = ""}`.
**Returns:** a new `JsonlExtractor`.
**Raises:** `ConfigError` if `field`/`id_field` is present but not a string (read through the strict
`read_string` reader); otherwise nothing.

#### extract(self, source: bytes | str, *, meta: Mapping[str, Any] = {}) -> Iterator[Document]

Iterates `source`'s lines (1-based line numbering for error messages), skipping blank lines,
parsing each non-blank line as one JSON record. `meta` is accepted for interface symmetry with the
other extractors but not read here (each document's metadata instead comes from the record's own
fields, minus the text field).

**Args:** `source` — the JSONL text/bytes; `meta` — unused.
**Returns:** an iterator of `Document`, one per non-blank line, `doc_id` from `id_field` (stringified)
if configured (required to be present and non-null) else `f"line-{line_no}"`, `text` from the
configured `field` (required to be a string), `meta` = every other record field.
**Raises:** `ExtractError` if a line is not valid JSON, if a parsed record is not a JSON object or
lacks the configured `field`, if the `field` value is present but not a string (previously
`str()`-coerced — a null/number/list became the string `"None"`/`"42"`/`"['a']"` and was indexed as
content), if `id_field` is configured but absent from the record, or if the `id_field` value is
JSON `null`.

### HtmlExtractor()

Implements the `Extractor` port. Strips HTML to text via the standard library's `html.parser`,
recording the heading trail (text of every `h1`–`h6` element) in `meta["headings"]` when any exist.
`CONFIG_KEYS = {"doc_id"}`.

Constructor: `HtmlExtractor(doc_id: str = "html")`. Internally uses a private `HTMLParser`
subclass that accumulates text into the current paragraph and flushes only at a block-level tag
boundary (`p`, `div`, `li`, `h1`–`h6`, `tr`, etc.), joining inline runs (`b`, `em`, `a`, `span`,
`code`, ...) with a single space rather than treating each text node as its own paragraph — so
`<p>The <b>quick</b> brown fox</p>` becomes one paragraph, `"The quick brown fox"`, not three
word fragments joined by blank lines (which would otherwise defeat `StructureChunker` downstream,
since it splits on exactly those blank lines). Script/style content is dropped entirely.

#### from_config(cls, options: Mapping[str, Any]) -> HtmlExtractor

**Args:** `options` — `{"doc_id": str = "html"}`.
**Returns:** a new `HtmlExtractor`.
**Raises:** `ConfigError` if `doc_id` is present but not a string (read through the strict
`read_string` reader); otherwise nothing.

#### extract(self, source: bytes | str, *, meta: Mapping[str, Any] = {}) -> Iterator[Document]

**Args:** `source` — the HTML text/bytes; `meta` — merged into the document's `meta`, consulted
for an overriding `doc_id`.
**Returns:** an iterator yielding exactly one `Document`, `text` = the extracted paragraphs joined
with `"\n\n"`, `meta["headings"]` = list of heading strings if any headings were found.
**Raises:** nothing (a malformed/unclosed HTML document is tolerated by `html.parser`; a trailing
paragraph with no closing tag is flushed on `close()`).

### MarkdownExtractor()

Implements the `Extractor` port. Splits Markdown into one document per ATX (`#`-prefixed) section,
carrying the heading text in `meta["section_path"]` so a structure-aware chunker keeps each
section intact. `CONFIG_KEYS = {"doc_id"}`.

**Any** line starting with `#` begins a new section, at any heading depth, including one inside a
fenced code block — there is no fence-tracking. This is a deliberate, documented behavior (an
earlier docstring incorrectly said "top-level (`#`/`##`)" only): splitting at every depth is useful
for retrieval, since a deep subsection is still a self-contained passage. A corpus with `#`
comments inside fenced code needs a different extractor.

Constructor: `MarkdownExtractor(doc_id: str = "md")`.

#### from_config(cls, options: Mapping[str, Any]) -> MarkdownExtractor

**Args:** `options` — `{"doc_id": str = "md"}`.
**Returns:** a new `MarkdownExtractor`.
**Raises:** `ConfigError` if `doc_id` is present but not a string (read through the strict
`read_string` reader); otherwise nothing.

#### extract(self, source: bytes | str, *, meta: Mapping[str, Any] = {}) -> Iterator[Document]

Splits `source` into `(heading, body_lines)` sections at every `#`-leading line (the heading text
has its leading `#`s and surrounding whitespace stripped); a section with no heading and no body is
skipped. A section's `text` is `f"{heading}\n\n{body}"` when both are present, else whichever one
is non-empty. `doc_id` is `f"{base}-{emitted}"`, `base` being `meta["doc_id"]` if present else the
extractor's configured `doc_id`, `emitted` a 0-based counter over sections actually yielded.

**Args:** `source` — the Markdown text/bytes; `meta` — merged into each section's `meta` (consulted
for an overriding `doc_id` base), plus `section_path` set to the section's heading when non-empty.
**Returns:** an iterator of `Document`, one per non-empty ATX section (including a leading
pre-heading section if it has a body).
**Raises:** nothing.

## ragkit.ingest.normalise

Text normalisation before indexing — cheap, deterministic cleanup that upstream extraction quality
depends on more than the choice of ANN index. Unicode is folded to a chosen normalisation form (so
the same text spelled two ways matches, and display width measures the same), whitespace is
optionally collapsed, and words hyphenated across a line break (a PDF extraction artefact) can
optionally be rejoined. Kept small and side-effect-free — a single function, no state.

#### normalise(text: str, *, form: str = "NFC", collapse_whitespace: bool = True, dehyphenate: bool = False) -> str

Applies, in order: (1) if `dehyphenate`, rejoin `foo-\nbar` -> `foobar` (regex `(\w)-\n(\w)` ->
`\1\2`); (2) Unicode-normalise to `form` via `unicodedata.normalize`; (3) if
`collapse_whitespace`, fold every run of whitespace (regex `\s+`) to a single space and strip the
ends.

**Args:** `text` — the text to normalise; `form` — one of `"NFC"`, `"NFKC"`, `"NFD"`, `"NFKD"`;
`collapse_whitespace` — fold all whitespace runs to one space and strip; `dehyphenate` — rejoin a
line-break-hyphenated word.
**Returns:** the normalised text.
**Raises:** `ValueError` if `form` is not one of the four recognised Unicode normalisation forms.
**Note:** `collapse_whitespace=True` destroys paragraph structure (blank lines become a single
space), so run this **after** chunking if the chunker is structure-aware — see the ordering
constraint at the top of this document.

## ragkit.ingest.reference

Imports a reference JSONL corpus into a `PairingStore`, and builds retrievers over it. Streams the
corpus, batched, into one *co-located* `PairingStore` — the row and its search entry are written in
the same transaction, so they can never drift apart — with the store's own row count as the
resumable floor. The vector index, when configured, still lives separately from the pairing store
(co-locating an ANN index is a different, harder problem) and is kept in sync via
`VectorIndex.reconcile`.

### ReferenceImportError(RagkitError)

Raised when the reference JSONL corpus is malformed — a line fails to parse as JSON, or a present
`index_field`/`target_field` (source/target) value is not a string. Carries `path` in its context
(via the `path=` keyword passed to `RagkitError`).

#### reference_pairings(path: Path, *, index_field: str = "source", target_field: str = "target", skip: int = 0) -> Iterator[Pairing]

Yields one `Pairing` per non-blank JSONL line beyond `skip`, numbered `chunk_id=f"ref-{line_no}"` —
a fetch script's own line numbering, so citations, gold data, and `ragkit.eval.retrieval` line up.
`skip` fast-forwards past already-imported lines by line number, *without parsing them* — the
mechanism `import_reference`'s resume path relies on. A line whose `index_field` value is absent,
JSON `null`, or the empty string is skipped entirely (never yielded as an empty-source pairing); a
present-but-**non-string** `index_field` raises `ReferenceImportError` instead of being
`str()`-coerced (a JSON `null` used to stringify to the truthy 4-char `"None"`, so the empty-source
guard never fired and `"None"` was silently indexed as content). `target_field` absent or `null`
yields `target=""`; a present-but-non-string `target_field` likewise raises `ReferenceImportError`.

**Args:** `path` — the JSONL corpus file; `index_field` — the record key holding the source text;
`target_field` — the record key holding the target text (optional per record); `skip` — number of
leading physical lines to skip without parsing (1-based line numbers `<= skip` are skipped).
**Returns:** an iterator of `Pairing`, each with `meta` set to the full parsed JSON record.
**Raises:** `ReferenceImportError` if a line beyond `skip` is not valid JSON, or if a present
`index_field`/`target_field` value is not a string.

#### import_reference(path: Path, pairing_store: PairingStore, *, index_field: str = "source", target_field: str = "target", vector: VectorIndex | None = None, embedder: EmbeddingClient | None = None, batch_size: int = 1000) -> int

Streams `path` into `pairing_store` in batches of `batch_size`, never holding the whole corpus in
RAM. **Resumable**: the floor is `pairing_store.count()`, the durable state of the one co-located
store — there is no separate store to reconcile a tail against. A finished on-disk store resumes to
a no-op; an in-memory store is always empty and rebuilds from scratch. (Caveat: the floor is a
*line-skip* count, so if some lines were skipped as blank/sourceless during the original run, a
resume re-parses — but does not re-add, since `PairingStore.add` is idempotent per `chunk_id` — a
few already-imported lines; this costs re-parsing, not correctness.)

If `vector`/`embedder` are given, each batch is embedded and upserted as it is imported (see
`embed_and_upsert`), and `reconcile_vector` runs once at the end — even on a no-op resume — so any
gap a previous crash left between the pairings and the vector index is closed.

**Args:** `path` — the JSONL corpus; `pairing_store` — the destination store; `index_field`,
`target_field` — as in `reference_pairings`; `vector` — optional vector index to keep in sync;
`embedder` — required if `vector` is given; `batch_size` — pairings per flush/embed batch.
**Returns:** the number of pairings actually added — excludes an already-imported line on resume
and a duplicate `chunk_id` within the corpus itself (per `PairingStore.add`).
**Raises:** `ValueError` if `vector` is given without `embedder`, or if `batch_size < 1`;
`ReferenceImportError` propagates from `reference_pairings` on malformed JSON.
**Side effects:** writes to `pairing_store` and, if configured, to `vector` (batched upserts plus
one final reconcile).

#### embed_and_upsert(pairings: Sequence[Pairing], vector: VectorIndex, embedder: EmbeddingClient) -> None

Embeds each pairing's `source` text (de-duplicated via `dedup_embed`) and upserts into `vector`
under its `chunk_id`. Shared by `import_reference`'s per-batch upsert, by `reconcile_vector`, and
by `ragkit.ingest.writeback` for the same re-embed-what's-missing step.

**No metadata is written to the vector index, deliberately** — every pairing is upserted with an
empty metadata dict (`{}`). A search returns `(chunk_id, score)`, and a retriever resolves display
text and metadata through `pairing_store.document()`, which is the single authoritative copy; an
earlier version copied each pairing's entire JSON record into the vector index too, which nothing
ever read and which cost pure disk/write bandwidth at corpus scale (7.1M full records duplicated).
Consequence: a driver that *can* filter on metadata (e.g. Qdrant) has none to filter on through
this path — nothing in the framework passes `where` to a vector search today.

**Args:** `pairings` — the pairings to embed and upsert; `vector` — the destination index;
`embedder` — the embedding client.
**Returns:** `None`.
**Raises:** whatever `embedder.embed` or `vector.upsert` raise (`EmbeddingError` on a bad
embedding response, for instance) — not caught here.
**Side effects:** upserts into `vector`.

#### reconcile_vector(pairing_store: PairingStore, vector: VectorIndex, embedder: EmbeddingClient, *, batch_size: int = 1000, on_batch: Callable[[int, int], None] | None = None, compact_every: int | None = None, on_compact: Callable[[], None] | None = None) -> int

Closes any gap between `pairing_store` (authoritative) and `vector` (derived): computes
`missing = sorted(vector.reconcile(pairing_store.all_ids()))` — which also drops orphan vectors as
a side effect of `VectorIndex.reconcile` — then re-embeds and upserts `missing` in chunks of
`batch_size` (never all at once: a from-scratch reconcile after a wipe or backend migration can
mean `missing` is the entire corpus, and embedding/upserting millions of rows in one call would
hold every one of their vectors in RAM at once). Safe to call when nothing is missing (a no-op).

`on_batch`, if given, is called with `(0, len(missing))` once up front — before any work, so a
caller can report the full scope immediately — and again with `(done, len(missing))` after each
batch. `compact_every`, if given, calls `vector.compact()` (silently skipped if `vector` has no
such method — a LanceDB-specific maintenance operation, not part of the general `VectorIndex`
port) after every `compact_every` batches, calling `on_compact()` first if given. Periodic
compaction bounds a real, measured failure mode: a table upserted in many small batches over a long
run accumulates one on-disk fragment per batch without bound, costing multiple GB of RSS per
subsequent batch once fragments number in the thousands.

**Args:** `pairing_store` — the authoritative store; `vector` — the index to reconcile; `embedder`
— used to re-embed missing pairings; `batch_size` — pairings per re-embed/upsert chunk; `on_batch`
— progress callback; `compact_every` — compact after this many batches, if set; `on_compact` —
called just before each triggered compaction.
**Returns:** the number of ids that were missing (and thus re-embedded).
**Raises:** `ValueError` if `batch_size < 1` or `compact_every` is given and `< 1`. If
`vector.compact()` itself raises, the exception propagates uncaught — the whole call aborts rather
than silently skipping a failed compaction; every batch already embedded up to that point has
already committed (embed-then-upsert happens before the compaction check), so nothing is lost and a
rerun resumes from exactly the remaining gap.
**Side effects:** re-embeds and upserts into `vector`; may call `vector.compact()`.

### PairingRetrievers()

Builds retrievers over a `PairingStore` (with an optional vector index/embedder for the dense
side). Does not implement a port itself — it is a retriever *factory*, not a `Retriever`.
`import_reference` is a module-level function rather than a method here: a pairing store is already
one complete store, so there is no separate lexical/document write path to batch alongside it.

Constructor: `PairingRetrievers(pairing_store: PairingStore, *, vector: VectorIndex | None = None, embedder: EmbeddingClient | None = None)`.
**Raises:** `ValueError` if `vector` is given without `embedder`.

#### resolve(self, chunk_id: str) -> str | None

**Args:** `chunk_id` — the id to resolve.
**Returns:** the pairing's display text via `pairing_store.document(chunk_id)`, or `None` if the
store has no document for that id.
**Raises:** nothing.

#### resolve_meta(self, chunk_id: str) -> Mapping[str, Any]

**Args:** `chunk_id` — the id to resolve.
**Returns:** the pairing's metadata via `pairing_store.document(chunk_id)`, or `{}` if absent.
**Raises:** nothing.

#### lexical_retriever(self, *, default_k: int = 10) -> LexicalRetriever

**Args:** `default_k` — passed through to the `LexicalRetriever` constructor (not itself enforced
here; see `ragkit.retrieve.retrievers.LexicalRetriever`).
**Returns:** a `LexicalRetriever` over `self._pairings` (which satisfies `SearchIndex`), resolving
text/meta through `self.resolve`/`self.resolve_meta`.
**Raises:** nothing.

#### dense_retriever(self) -> DenseRetriever

**Returns:** a `DenseRetriever` over the configured vector index and embedder.
**Raises:** `ValueError` if this instance was built without both `vector` and `embedder`.

#### hybrid_retriever(self, *, reranker: RerankClient | None = None, candidate_pool: int = 40, mmr_lambda: float = 0.7, lexical_min_score: float = 0.30, dense_min_score: float = 0.55) -> HybridRetriever

**Args:** as documented on `ragkit.retrieve.hybrid.HybridRetriever`'s constructor.
**Returns:** a `HybridRetriever` composing `self.lexical_retriever()` and `self.dense_retriever()`.
**Raises:** `ValueError` (from `dense_retriever`) if this instance has no vector index/embedder;
`HybridError` if `candidate_pool`/`mmr_lambda` are out of range (from `HybridRetriever.__init__`).

## ragkit.ingest.writeback

Folds a finished run's outputs into the reference memory as new pairings. A deliberate, separate
post-run step, never automatic: a run is over, its `RunStore` holds the results, and write-back
reads them, keeps the ones trustworthy enough to become a future run's worked example, and writes
them through a `Sink` — closing the loop from a produced `(source, context, target)` triple back
into what a later run can retrieve.

#### write_back(run_store: RunStore, sink: Sink, pairing_store: PairingStore, *, vector: VectorIndex | None = None, embedder: EmbeddingClient | None = None, batch_size: int = 1000) -> int

Reads `run_store.results()`, keeps only the `Status.VERIFIED` ones — the conservative bar for what
is trusted enough to teach a later run; `PRODUCED` covers an output kept *despite* a flagged rule
or an incomplete review, which write-back should not reinforce — folds each kept result's captured
`context_passage` into its record's `meta` (under `PairingSink.CONTEXT_META_KEY`, since `Sink.write`
takes a task-agnostic `Iterable[Record]` and `context_passage` lives on `RunResult`, not `Record`),
and hands the enriched records to `sink.write(...)`.

`sink` and `pairing_store` are both required: `sink` is the generic, injectable reinjection seam
(`write` returns nothing, so nothing it does is directly observable), while `pairing_store` is
needed to measure how many pairings were actually added (`pairing_store.count()` before/after) and
to drive `reconcile_vector`. The caller must pass a `sink` that writes into this same
`pairing_store`, or the reported count is meaningless.

If `vector`/`embedder` are given, `reconcile_vector` runs afterward (re-embedding the gap in chunks
of `batch_size`, not all at once), so a later dense/hybrid retrieval sees the new pairings too.

**Args:** `run_store` — source of finished results; `sink` — writes enriched records into the
reference store; `pairing_store` — the same store `sink` writes into, used to measure the delta and
drive reconciliation; `vector`, `embedder` — optional, kept in sync if both given; `batch_size` —
passed through to `reconcile_vector`.
**Returns:** the number of pairings actually added (`pairing_store.count()` delta).
**Raises:** `ValueError` if `vector` is given without `embedder`.
**Side effects:** writes to `pairing_store` (via `sink`); may re-embed/upsert into `vector` and
call `vector.compact()` (via `reconcile_vector`).
