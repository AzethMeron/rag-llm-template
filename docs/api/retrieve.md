# ragkit.retrieve

`ragkit.retrieve` is the lexical/dense/hybrid retrieval stack plus reranking: first-stage
retrievers over the storage indexes (`LexicalRetriever` over BM25, `DenseRetriever` over
embeddings), the pure fusion math they compose with (Reciprocal Rank Fusion, Maximal Marginal
Relevance), an optional cross-encoder reranker, and the `HybridRetriever` that wires all of it
together behind one `Retriever` port. The stdlib-only fusion math (`ragkit.retrieve.fusion`) needs
nothing beyond the interpreter; the dense path additionally needs numpy and a local embedding
endpoint, the rerank path a local rerank endpoint — each optional and reached only through an
injectable client. Depends on `ragkit.store` and `ragkit.core`; never on the harness.

A shared convention runs through this package, inherited from `ragkit.core.ports`: **every score a
retrieval component returns is "higher is better", normalised toward `[0, 1]`.** A backend whose
native scale is inverted or unbounded converts inside its own driver, never leaking a sign
convention into the fusion/rerank layer. `ragkit.retrieve.fusion.rrf`'s raw output is the one
partial exception — see its entry below for the caveat and how `best_possible_rrf` restores the
`[0, 1]` convention.

## ragkit.retrieve.embedding

A client for a local, OpenAI-compatible `/v1/embeddings` endpoint (llama.cpp, ollama, vLLM).
Encodes texts into L2-normalised dense vectors, so a later dot product between two embeddings is
their cosine similarity. Placeholder tokens (`[[0]]`-style) are stripped before embedding, mirroring
the lexical side, so a token every line shares cannot pull unrelated lines together. numpy is
imported lazily, inside this module only, so the rest of the retrieval layer imports without it
until a dense path is actually built. Behind an injectable `httpx.Client` for testing with an
in-memory transport.

`DEFAULT_EMBEDDING_MIN_SCORE = 0.55` — a sensible starting cosine floor, on a different scale from
the lexical floor: embedding cosines cluster high (a loosely related pair still scores ~0.5), so
the lexical default of 0.30 would admit almost everything on the dense side. Tune per corpus.

### EmbeddingError(RagkitError)

Raised when the embedding endpoint is unreachable or misbehaving, or when numpy is unavailable.

Constructor: `EmbeddingError(reason: str, *, url: str | None = None)`. Sets `self.url` and passes
`endpoint=url` into `RagkitError`'s context.

### EmbeddingClient()

Implements the `Embedder` port (`ragkit.core.ports.Embedder`). Embeds text via a local endpoint,
returning L2-normalised vectors as plain lists of floats (so a caller that does not want numpy need
not import it — only this module's internals touch numpy directly).

Constructor:
```
EmbeddingClient(*, base_url: str, model: str = "local", batch_size: int = 64,
                 timeout_seconds: float = 120.0, max_retries: int = 4,
                 retry_backoff_seconds: float = 1.0, client: httpx.Client | None = None)
```
`base_url` is normalised to `base_url.rstrip("/") + "/embeddings"`. `client`, if given, is used
as-is and never closed by `close()` (ownership stays with the caller); if omitted, a new
`httpx.Client(timeout=timeout_seconds)` is created and owned.
**Raises:** `ValueError` if `batch_size < 1`, `max_retries < 0`, or `retry_backoff_seconds < 0`;
`EmbeddingError` if numpy is not importable.

#### embed(self, texts: Sequence[str]) -> list[Sequence[float]]

Embeds `texts` in batches of `batch_size`, returns rows in **input order**. Internally: each batch
is POSTed to the endpoint (retrying a transient failure — a timeout, dropped connection, or
`429`/`500`/`502`/`503`/`504` status — with exponential backoff `retry_backoff_seconds * 2**attempt`
up to `max_retries` attempts; a `4xx` or malformed reply is raised at once, never retried); the
response's per-item `index` field (not array position) is used to reorder each batch back to input
order, because array order is not guaranteed by the OpenAI embeddings schema and a reordering
server or proxy would otherwise silently pair every vector with the wrong text. The returned
indices must be exactly `range(len(chunk))` for that batch — a missing, repeated, or out-of-range
index is refused. The assembled matrix is L2-row-normalised (`matrix /= norm(axis=1) + 1e-9`).

**Args:** `texts` — texts to embed, in order (an empty sequence returns `[]` with no HTTP call).
**Returns:** one `Sequence[float]` per input text, same order, each vector unit-length (up to the
`1e-9` epsilon).
**Raises:** `EmbeddingError` for: a malformed response (`data` not a list, an item missing
`index`/`embedding`), an embedding count mismatching the input count, indices not a permutation of
`0..n-1`, a non-uniform/ragged/non-numeric embedding matrix, a wrong dimensionality (not 2-D), any
row containing a non-finite (NaN/Inf) value, or a request failing after retries are exhausted.

#### embed_one(self, text: str) -> list[float]

**Args:** `text` — a single text to embed.
**Returns:** `list(self.embed([text])[0])` — the one embedding vector as a plain list.
**Raises:** as `embed`.

#### close(self) -> None

**Returns:** `None`.
**Side effects:** closes the underlying `httpx.Client` only if this instance created it (no
`client` was injected at construction) — an injected client is the injector's to close.

#### dedup_embed(embedder: EmbeddingClient, texts: Sequence[str]) -> dict[str, Sequence[float]]

Embeds the distinct values in `texts` once each (via `dict.fromkeys` to preserve first-seen order),
keyed by text — a pure cost saving for a batch that routinely repeats a value (a duplicate line, a
shared passage), with no cache persisted across calls: a duplicate recurring in a later, separate
call is simply re-embedded.

**Args:** `embedder` — the client to embed with; `texts` — texts to embed, possibly with
duplicates.
**Returns:** a mapping from each distinct text to its embedding.
**Raises:** as `EmbeddingClient.embed`.

## ragkit.retrieve.fusion

Pure retrieval-fusion math: Reciprocal Rank Fusion and Maximal Marginal Relevance. Stdlib-only, no
I/O, no numpy — plain list/dict algorithms over whatever hashable identity the caller uses for a
candidate (a chunk id, an entry, anything `==`/`hash`-stable), kept separate from the retrievers
that use them so each is provable in isolation. `H` is the module's `TypeVar` for that hashable
candidate identity.

`RRF_K = 60` — Cormack et al.'s rank-fusion constant, and the conventional default. Exposed because
a caller that wants to read a fused score on an absolute scale needs the same `k` the fusion used
(see `best_possible_rrf`).

#### rrf(rankings: Sequence[Sequence[H]], *, k: int = RRF_K) -> dict[H, float]

Reciprocal Rank Fusion over any number of best-first rankings. Each ranking contributes
`1 / (k + rank)` (rank 1-based) to every candidate it contains; a candidate's fused score is the sum
of that contribution across every ranking that contains it:

```
score(c) = sum over rankings R containing c of  1 / (k + rank_R(c))
```

So a candidate near the top of several rankings outscores one near the top of only one.
Deterministic, and — unlike weighted-score fusion — needs no calibration between the rankings'
otherwise incomparable native scales. **The raw output is not itself on `[0, 1]`** — the ceiling per
ranking is `1/(k+1)`, and scores from multiple rankings simply add, so a caller wanting a
fixed-scale relevance divides by `best_possible_rrf(len(rankings), k=k)` (this is exactly what
`HybridRetriever` does).

**Args:** `rankings` — any number of best-first sequences of candidates; `k` — the RRF constant
(must be finite and `>= 1`).
**Returns:** a `dict` mapping every candidate appearing in any ranking to its fused score.
**Raises:** `ValueError` if `k` is non-finite (NaN/±inf) or `< 1`.

#### best_possible_rrf(ranking_count: int, *, k: int = RRF_K) -> float

The largest score `rrf` can assign, over `ranking_count` rankings each contributing its maximum
(rank 1, i.e. `1/(k+1)`):

```
best_possible_rrf(n, k) = n / (k + 1)
```

Dividing a fused score by this gives a **query-independent** `[0, 1]` scale on which a floor keeps
a stable meaning: `1.0` means "every ranking's top hit", and with two rankings, `0.5` means "one
ranking's top hit, the other never found it". Deliberately not min-max over the query's own
candidates, which would hand the best of three terrible candidates a perfect `1.0`.

**Args:** `ranking_count` — number of rankings that were fused (must be finite and `>= 1`); `k` —
the RRF constant used (must match the `rrf` call being normalised; must be finite and `>= 1`).
**Returns:** `ranking_count / (k + 1)`.
**Raises:** `ValueError` if `ranking_count` or `k` is non-finite, or either is `< 1`.

#### mmr_order(candidates: Sequence[H], relevance: Mapping[H, float], similarity: Callable[[H, H], float], *, lambda_: float, k: int) -> list[H]

Greedy Maximal Marginal Relevance selection. Repeatedly picks, from the remaining candidates, the
one maximising

```
mmr(c) = lambda_ * relevance[c] - (1 - lambda_) * max(similarity(c, s) for s in selected)
```

(the `max` over an empty `selected` — the first pick — is defined as `0.0`, so the first pick
reduces to plain relevance ranking), so once a candidate is picked, a later candidate resembling it
is penalised — the standard way to keep a top-k list from being dominated by near-duplicates. Ties
break by input order (first-seen wins, via `list.pop`'s use of the first index achieving the max),
so the result is deterministic for deterministic `relevance`/`similarity` callables. `similarity`
is called lazily, only between a not-yet-selected candidate and each already-selected one — never
precomputed as a full pairwise matrix.

**Args:** `candidates` — the pool to select from, any order; `relevance` — a mapping giving every
candidate's base relevance (must have an entry for every candidate); `similarity` — a symmetric(-in-
practice) pairwise similarity function; `lambda_` — the relevance/diversity trade-off, `1.0` is pure
relevance, `0.0` is pure anti-redundancy; `k` — how many to select.
**Returns:** the selected candidates in pick order, length `min(k, len(candidates))` (empty if
`k <= 0`).
**Raises:** `ValueError` if `lambda_` is outside `[0, 1]` (this also rejects NaN/±inf, which fail
the comparison); `ValueError` if any candidate has no entry in `relevance`; `ValueError` if any
`relevance` or `similarity` value encountered is non-finite (a NaN would silently corrupt every
comparison it takes part in, since every comparison against NaN is `False`).

## ragkit.retrieve.hybrid

The hybrid retriever: lexical + dense fusion, optional cross-encoder reranking, MMR diversity.
Composes two first-stage `Retriever` arms — lexical (exact-term) and dense (semantic) — behind one
`Retriever`, rather than forking either. Per query:

1. Each arm retrieves its own `candidate_pool`, gated by its own calibrated floor (the arms' scores
   are on different native scales, so each keeps its own floor rather than sharing one).
2. The two rankings are fused with `rrf` — a candidate both arms agree on outranks one only one arm
   found, with no need to calibrate incomparable scores.
3. If a reranker is configured, the fused pool (capped to `candidate_pool` candidates, ranked by
   fused score then by per-arm confidence as a tiebreak) is re-scored by the cross-encoder, which
   already returns `[0, 1]`. Without one, the fused score is divided by
   `best_possible_rrf(2)` (2 arms) to a fixed, query-independent `[0, 1]` scale.
4. Candidates below the caller's `min_score` (on that final relevance) are dropped.
5. The survivors are diversified with `mmr_order`, using `trigram_similarity` over the retrieved
   texts (no extra embedding round-trip) as the similarity function.

**Fails loud.** A dead reranker propagates `RerankError` — there is no silent fallback to the
unreranked order; a misbehaving reranker demands attention, not a quietly degraded answer.

**Measured caution, carried from the reference project:** equal-weight RRF fusion blends its arms
rather than taking the better one, and on two real corpora it scored *below* the better single arm
unless a reranker was also attached. Measure on your own corpus before adopting it as a default —
the plain embedding retriever is often the better baseline.

### HybridError(RagkitError)

Raised when the hybrid retriever is misconfigured (`candidate_pool < 1` or `mmr_lambda` outside
`[0, 1]`).

### HybridRetriever()

Implements the `Retriever` port (`ragkit.core.ports.Retriever`). Lexical + dense fusion, optional
rerank, MMR diversity, as one `Retriever`.

Constructor:
```
HybridRetriever(lexical: Retriever, dense: Retriever, *, reranker: RerankClient | None = None,
                 candidate_pool: int = 40, mmr_lambda: float = 0.7,
                 lexical_min_score: float = 0.30, dense_min_score: float = 0.55)
```
**Raises:** `HybridError` if `candidate_pool < 1` or `mmr_lambda` is outside `[0, 1]`.

#### retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]

Runs the 5-step pipeline described above: fetches `candidate_pool` hits from each arm at its own
floor (`lexical_min_score`/`dense_min_score`), fuses their chunk-id rankings with `rrf`, ranks the
fused candidates by `(-fused_score, -confidence)` and truncates to `candidate_pool`, computes final
relevance (reranked or RRF-normalised, per above), drops anything below `min_score`, and returns the
top `k` after MMR diversification (`lambda_=mmr_lambda`).

The **confidence** tiebreak: for each arm, `scaled = clamp((hit.score - floor) / (1 - floor), 0, 1)`
per hit, and a candidate's confidence is the best (highest) `scaled` value across the two arms.
This exists because RRF ties are not rare — when the arms disagree outright, both candidates score
exactly `1/(k+1)`, and a plain stable sort would then settle a retrieval-quality question by
argument order (always favoring the same arm).

**Args:** `query` — the query text; `k` — number of results to return (returns `()` if `k <= 0`);
`min_score` — floor on final relevance (`[0, 1]` scale, reranked or RRF-normalised).
**Returns:** up to `k` `Retrieved` hits, MMR-ordered, `score` = the final relevance used for
filtering (**not** the raw fused RRF score). Returns `()` if either arm's fusion is empty or if no
survivor clears `min_score`.
**Raises:** `RerankError` if a configured reranker's endpoint fails or returns a malformed
response — propagates uncaught, no fallback to the unreranked order.
**Side effects:** calls both arms' `retrieve` and, if configured, the reranker's `rerank` — each a
network round-trip to a local endpoint.

#### close(self) -> None

**Returns:** `None`.
**Side effects:** calls `self._reranker.close()` if a reranker was configured; does **not** close
the lexical/dense arms (their lifecycle is the caller's, since this class did not create them).

## ragkit.retrieve.rerank

An optional cross-encoder reranker over a local `/v1/rerank` endpoint (llama.cpp built with
`--reranking`, or any Jina/TEI-compatible rerank server). A first-stage retriever must be cheap
enough to run over the whole candidate pool; a reranker is the opposite trade — expensive but
precise — so it only ever scores a short list a first stage has already narrowed down. It never
replaces a first-stage retriever on its own; `HybridRetriever` is what puts one in front of one.
Behind an injectable `httpx.Client` so it is testable with an in-memory transport and no server.

**Normalisation happens here, in the driver.** `rerank()` returns relevance already in `[0, 1]`,
the convention the vector drivers also follow: each driver knows its own backend's scale and
converts, rather than leaving every caller to guess. The scales genuinely differ — llama.cpp
returns an unbounded cross-encoder logit, while Jina and Cohere return a `relevance_score` already
in `[0, 1]` — and getting `score_scale` wrong is silent, not loud (squashing an already-`[0, 1]`
score with `sigmoid` maps it into `[0.5, 0.731]`, which preserves ranking but makes every floor
meaningless).

`SCORE_SCALES = frozenset({"logit", "unit"})` — what an endpoint's `relevance_score` means:
`"logit"` is an unbounded cross-encoder logit (llama.cpp `--reranking`), squashed with `sigmoid`;
`"unit"` is already in `[0, 1]` (Jina, Cohere, most TEI deployments), passed through with a clamp.

### RerankError(RagkitError)

Raised when the rerank endpoint is unreachable or returns something the client cannot use, so a
caller can catch it and fail clearly (naming the endpoint) rather than let a raw HTTP or parsing
exception escape.

Constructor: `RerankError(reason: str, *, url: str | None = None)`. Sets `self.url` and passes
`endpoint=url` into `RagkitError`'s context.

### RerankClient()

Implements the `Reranker` port (`ragkit.core.ports.Reranker`). A thin client for one `/v1/rerank`
endpoint. Does no I/O at construction (a reranker has no index to build); every call is a fresh
request over the candidates it is given.

Constructor:
```
RerankClient(*, base_url: str, model: str = "local", timeout_seconds: float = 120.0,
              score_scale: str = "logit", client: httpx.Client | None = None)
```
`base_url` is normalised to `base_url.rstrip("/") + "/rerank"`. `client` ownership follows the same
rule as `EmbeddingClient`: an injected client is never closed by `close()`.
**Raises:** `RerankError` if `score_scale` is not one of `SCORE_SCALES`.

#### rerank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]

Scores `documents` against `query`, best first. POSTs `{"model", "query", "documents"}`, reads
`response["results"]` as a list of `{"index", "relevance_score"}`, converts each score to relevance
via `sigmoid` (if `score_scale == "logit"`) or a `[0, 1]` clamp (if `"unit"`), validates, and
re-sorts by descending relevance itself rather than trusting the endpoint's order (both scales are
monotone, so sorting before or after normalisation gives the same order).

**Args:** `query` — the query text; `documents` — candidate texts to score (empty short-circuits
with no HTTP call).
**Returns:** `(index, relevance)` pairs, best first, `relevance` in `[0, 1]`; `index` is a
permutation of `range(len(documents))`, so a caller may index by it without defending itself.
**Raises:** `RerankError` for: an HTTP-level failure (`httpx.HTTPError`), a malformed response
(`results` missing or unparseable, an item missing `index`/`relevance_score`), a result count not
matching the document count, an `index` out of range, a `relevance_score` that is NaN, or duplicate
result indices (right count and all in-range does not imply distinct — e.g.
`[(0, .9), (0, .8)]` would otherwise pass).

#### close(self) -> None

**Returns:** `None`.
**Side effects:** closes the underlying `httpx.Client` only if this instance created it.

#### sigmoid(x: float) -> float

Squashes a cross-encoder's unbounded logit into `[0, 1]` for any finite `x`, computed branch-wise
for numerical stability:

```
sigmoid(x) = 1 / (1 + e^{-x})   if x >= 0
           = e^{x} / (1 + e^{x})  if x < 0
```

Branching on the sign is the only correct form here: the naive single-expression `1/(1+exp(-x))`
calls `exp` on a large *positive* argument for very negative `x` and raises `OverflowError` past
about `x = -710`, so one outlier logit would kill a whole rerank call. Each branch as written only
ever exponentiates a non-positive number, which underflows harmlessly to `0.0` instead.

**Args:** `x` — any finite logit.
**Returns:** a value in `(0, 1)` (never exactly 0 or 1 for finite `x`).
**Raises:** `ValueError` if `x` is NaN — there is no relevance it could honestly represent, and it
would silently poison the ranking downstream.

## ragkit.retrieve.retrievers

First-stage retrievers over the storage indexes: lexical (BM25) and dense (embeddings). Each
adapts a storage index to the `Retriever` port — `retrieve(query, *, k, min_score) ->
tuple[Retrieved, ...]`, best-first, deterministic, safe to share across a run's concurrent workers.
The underlying index returns `(chunk_id, score)`; a `Resolver` callable maps an id back to its text
(a hit is dropped, not surfaced as an empty result, if `resolve` returns `None` — the text store no
longer has it). A hit whose score is below `min_score` is dropped. `Resolver = Callable[[str], str
| None]`; `MetaResolver = Callable[[str], Mapping[str, Any]]` (defaults to a resolver returning
`{}`).

### LexicalRetriever()

Implements the `Retriever` port, over a `SearchIndex` (BM25). Strong on names, ids, and recurring
terminology. Depends only on the narrow `SearchIndex` read side (never indexes or deletes through a
retriever), which is exactly what `PairingStore` exposes.

Constructor: `LexicalRetriever(index: SearchIndex, resolve: Resolver, *, meta: MetaResolver = <returns {}>, default_k: int = 10)`.
Note `default_k` is accepted by the constructor but not read anywhere in `retrieve` (which always
requires an explicit `k`); it exists for callers/factories (e.g.
`PairingRetrievers.lexical_retriever`) that want a documented default to pass along themselves.

#### retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]

**Args:** `query` — the query text; `k` — how many results to request from the index (returns `()`
immediately if `k <= 0`); `min_score` — floor on `SearchIndex`'s `[0, 1)` score.
**Returns:** up to `k` `Retrieved` hits, best-first, in the index's own order, with any hit below
`min_score` or whose text cannot be resolved dropped.
**Raises:** whatever `self._index.search` raises; not caught here.

### DenseRetriever()

Implements the `Retriever` port, over a `VectorIndex` plus an `EmbeddingClient`. Strong on
paraphrase and cross-script matches the lexical retriever is blind to.

Constructor: `DenseRetriever(index: VectorIndex, embedder: EmbeddingClient, resolve: Resolver, *, meta: MetaResolver = <returns {}>)`.

#### retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]

Embeds `query` with `self._embedder.embed_one(query)`, then searches `self._index` with the
resulting vector.

**Args:** `query` — the query text; `k` — how many results to request (returns `()` immediately if
`k <= 0`, with no embedding call made); `min_score` — floor on `VectorIndex`'s `[0, 1]` cosine
score.
**Returns:** up to `k` `Retrieved` hits, best-first, any hit below `min_score` or unresolvable
dropped.
**Raises:** `EmbeddingError` from the embed step; whatever `self._index.search` raises from the
search step.

#### trigram_similarity(a: str, b: str) -> float

Jaccard similarity over character trigrams — a stdlib, embedding-free measure of how alike two
texts look, used by `HybridRetriever` to diversify a shortlist via MMR without an extra embedding
round-trip per candidate. Each text is casefolded and stripped; if the result has fewer than 3
characters it is treated as a single-element set containing the whole (non-empty) string, else it
is the set of all length-3 substrings:

```
similarity(a, b) = |trigrams(a) ∩ trigrams(b)| / |trigrams(a) ∪ trigrams(b)|
```

**Args:** `a`, `b` — the two texts to compare.
**Returns:** a value in `[0, 1]`; `0.0` if either text's trigram set is empty (i.e. either
casefolded-and-stripped text is itself empty).
**Raises:** nothing.

## ragkit.retrieve.tuning

Config-driven retrieval tuning: `retrieval.toml` -> a validated `RetrievalSettings`. The retrieval
stack (lexical + dense arms, RRF fusion, optional cross-encoder rerank, MMR) is otherwise wired in
Python; this module lets a recipe pick the arms, the per-arm score floors, the candidate pool, the
MMR trade-off, and the rerank model from config, without editing code. Parsing lives here (pure,
dependency-free, testable in isolation); wiring the actual endpoints and indexes from these settings
is the CLI's job, in `cli/app.py`.

The final relevance floor for a run stays the retrieved *block*'s `min_score` (`context.toml`); the
per-arm `lexical`/`dense` floors here are the fusion-*input* floors (previously hardcoded
constants), applicable only to the `hybrid` `kind` — see `RetrievalSettings` below for exactly
which keys apply to which `kind`.

### RetrievalSettings()

A fully-validated, immutable (`@dataclass(frozen=True, slots=True)`) retrieval configuration.
`kind` selects the stack (`"lexical"`, `"dense"`, or `"hybrid"`); the dense/hybrid kinds name an
embedding model, and rerank (when enabled, hybrid-only) names a rerank model — both logical model
names resolved against `models.toml`.

**Attributes:**
- `kind: str = "lexical"` — one of `"lexical"`, `"dense"`, `"hybrid"`.
- `candidate_pool: int = 40` — per-arm candidate pool size before fusion; must be `>= 1`; applies
  only when `kind == "hybrid"` (only fusion fetches a per-arm pool).
- `mmr_lambda: float = 0.7` — MMR relevance/diversity trade-off; must be in `[0, 1]`; applies only
  when `kind == "hybrid"` (only the hybrid stack diversifies with MMR).
- `lexical_min_score: float = 0.30` — the lexical arm's fusion-input floor; must be in `[0, 1]`;
  applies only when `kind == "hybrid"`.
- `dense_min_score: float = 0.55` — the dense arm's fusion-input floor; must be in `[0, 1]`;
  applies only when `kind == "hybrid"`.
- `embedding_model: str = ""` — the embedding model name (from `models.toml`); required (non-empty)
  when `needs_embedding` is true, and must be empty otherwise.
- `rerank_enabled: bool = False` — whether to rerank the fused pool; only meaningful (and only
  permitted to be `True`) when `kind == "hybrid"`.
- `rerank_model: str = ""` — the rerank model name; required (non-empty) when `rerank_enabled` is
  `True`.
- `rerank_score_scale: str = "logit"` — one of `ragkit.retrieve.rerank.SCORE_SCALES`
  (`"logit"`/`"unit"`).

**Raises** (in `__post_init__`, on construction): `ValueError` if `kind` is not one of the three
recognised kinds; if `candidate_pool < 1`; if `mmr_lambda` is outside `[0, 1]`; if
`lexical_min_score`/`dense_min_score` is outside `[0, 1]`; if `needs_embedding` is true and
`embedding_model` is empty; if `embedding_model` is set but `needs_embedding` is false; if
`rerank_enabled` is true and `rerank_model` is empty; if `rerank_score_scale` is not a recognised
scale; if `rerank_enabled` is true but `kind != "hybrid"` (an enabled reranker under any other kind
would be constructed but never called — refused at construction rather than silently ignored at
query time).

#### needs_embedding (property, not a method — access as `settings.needs_embedding`, no parentheses)

**Returns:** `True` if `kind` is `"dense"` or `"hybrid"` (i.e. this configuration embeds queries),
else `False`.
**Raises:** nothing.

#### load_retrieval(path: Path) -> RetrievalSettings

Parses and validates `retrieval.toml` at `path`. Unknown top-level/section keys are refused (via
`reject_unknown`), as is a key the *chosen* `kind` would silently ignore (e.g. `candidate_pool`
under `kind = "lexical"` would otherwise parse, validate, and then be discarded with no signal to
the user) — checked only after construction, once `kind` is known to be one of the three valid
values, so an applicability complaint about an invalid `kind` never buries the more fundamental
"unrecognised kind" error. Range checks and cross-key invariants are enforced by
`RetrievalSettings.__post_init__` itself.

**Args:** `path` — the `retrieval.toml` file to load.
**Returns:** a validated `RetrievalSettings`.
**Raises:** `ConfigError` for: an unparseable TOML file; an unknown key anywhere in
`[retrieval]`/`[retrieval.lexical]`/`[retrieval.dense]`/`[retrieval.rerank]`; a value of the wrong
type; any `RetrievalSettings.__post_init__` invariant violation (wrapped as `ConfigError`); or a key
present that the resolved `kind` does not apply to.
