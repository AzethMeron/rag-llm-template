"""Assemble a run from a configuration directory, and execute it.

A run is described entirely by config files in one directory — ``models.toml``, ``personas.toml``,
``rules.toml``, ``context.toml``, an optional ``storage.toml``, and a ``recipe.toml`` that names the
task's output schema, its extra validators, its input label, and (optionally) a reference corpus to
retrieve from. Every component is resolved through its registry, so a user's own driver, validator,
block, or schema is selected the same way a built-in is — no framework change.

The assembly does all of its config reading and consistency checks (including the model-pool
capacity/thrash guard) *before* any server is contacted, so a misconfiguration is reported for what
it is rather than masked by a "no server" error. That ordering is what matters, not that
``assemble`` never touches a server at all: with ``retrieval.kind`` of ``dense`` or ``hybrid``,
building the retriever embeds the reference corpus, which does call the embedding endpoint — after
every config check has already passed. ``client_factory`` is injectable so a test drives the whole
run against an in-memory transport with no server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragkit.core.config import (
    ConfigError,
    as_table,
    load_toml,
    read_bool,
    read_string,
    reject_unknown,
    tables,
)
from ragkit.core.lexicon import read_lexicon
from ragkit.core.ports import Retriever, SqlStore, VectorIndex
from ragkit.harness import (
    Harness,
    OUTPUT_SCHEMAS,
    VALIDATORS,
    RuleSet,
    ValidatorPipeline,
    load_context,
    load_panel,
)
from ragkit.harness.memory import OutputMemory
from ragkit.ingest.reference import PairingRetrievers, ReferenceImportError, import_reference
from ragkit.llm.pool import ClientFactory, ModelPool, load_models
from ragkit.retrieve import (
    RETRIEVERS,
    EmbeddingClient,
    RerankClient,
    RetrievalSettings,
    load_retrieval,
)
from ragkit.store import Storage, load_storage


class CliError(ConfigError):
    """A run cannot be assembled as configured."""


@dataclass(slots=True)
class Assembled:
    """Everything a run needs, built and consistency-checked before any server is contacted."""

    harness: Harness
    pool: ModelPool
    storage: Storage
    retriever: Retriever | None


def assemble(config_dir: Path, *, substitutions: dict[str, str] | None = None,
             client_factory: ClientFactory | None = None,
             extra_validators: list[Any] | None = None,
             retriever: Retriever | None = None) -> Assembled:
    """Build a :class:`Assembled` run from the config files in ``config_dir``.

    ``substitutions`` fill the ``{placeholder}`` tokens in persona instructions (the language pair
    for translation, say). ``extra_validators`` are task-specific validator instances a recipe
    passes in directly (in addition to any named in ``recipe.toml``).

    ``retriever`` injects a pre-built :class:`~ragkit.core.ports.Retriever`, overriding whatever
    ``[reference]`` would build. This is the replace-without-editing-our-code seam for a
    *corpus-stateful* retriever (one that must hold our reference corpus): a recipe or host
    constructs it and passes it here, exactly as ``client_factory`` and ``extra_validators`` are
    injected. A *corpus-free* custom retriever (one with its own backend) needs no injection — name
    it by dotted path in ``[reference].retriever`` and it is resolved through the ``RETRIEVERS``
    registry.
    """
    config_dir = Path(config_dir)
    pool = load_models(config_dir / "models.toml", client_factory=client_factory)
    panel = load_panel(config_dir / "personas.toml", substitutions)
    ruleset = RuleSet.load(config_dir / "rules.toml")
    context = load_context(config_dir / "context.toml")
    storage = _load_storage(config_dir)
    recipe = _load_recipe(config_dir / "recipe.toml")

    # The model pool must not hold more distinct models than an endpoint keeps resident -- checked
    # here, before the server. (The from_rules-needs-advisory check runs in the Harness constructor
    # below, also before any server contact.)
    pool.check_capacity(panel.models_in_use())

    output_schema = OUTPUT_SCHEMAS.create(recipe.output_schema, recipe.output_schema_options)
    lexicon = read_lexicon(config_dir / "lexicon.jsonl")
    validators = [VALIDATORS.create(spec, options) for spec, options in recipe.validators]
    validators.extend(extra_validators or [])
    # The external read-only store, its introspector, and the reference retriever are made
    # available to any pluggable validator (the generated-SQL safety check reads the schema; the
    # grounded-decision check re-retrieves the memory to verify a citation).
    external = _external_sql(storage)
    if retriever is None:
        retrieval_path = config_dir / "retrieval.toml"
        retriever = (_build_retrieval(load_retrieval(retrieval_path), recipe, config_dir, storage,
                                      pool, client_factory)
                     if retrieval_path.is_file()
                     else _build_reference(recipe, config_dir, storage))
    shared = {"sql_store": external, "introspector": storage.introspector,
              "retriever": retriever}
    pipeline = ValidatorPipeline(ruleset, lexicon, extra=validators, shared=shared)
    harness = Harness(pool, panel, ruleset, output_schema, pipeline, context,
                      memory=OutputMemory() if recipe.use_memory else None,
                      retriever=retriever, sql_store=external, introspector=storage.introspector,
                      input_label=recipe.input_label, stand_in=recipe.stand_in)
    return Assembled(harness=harness, pool=pool, storage=storage, retriever=retriever)


def _load_storage(config_dir: Path) -> Storage:
    path = config_dir / "storage.toml"
    return load_storage(path) if path.is_file() else Storage()


def _external_sql(storage: Storage) -> SqlStore | None:
    # The external data source is a read-only SqlStore; the framework's own store (if any) is
    # writable. Only a read-only binding is exposed to the sql_rows context block / NL->SQL task.
    if storage.sql is not None and storage.sql.read_only:
        return storage.sql
    return None


@dataclass(frozen=True, slots=True)
class _Recipe:
    """A parsed ``recipe.toml``. The field defaults *are* the loader's defaults — read from
    ``_RECIPE_DEFAULTS`` below rather than retyped in :func:`_load_recipe`, so a default and its
    loader cannot drift (the same pattern as ``retrieve.tuning``)."""

    output_schema: str = "json_field"
    output_schema_options: dict[str, Any] = field(default_factory=dict)
    validators: tuple[tuple[str, dict[str, Any]], ...] = ()
    input_label: str = "Input to act on:"
    stand_in: str = "they"
    use_memory: bool = False
    reference_file: str = ""
    reference_retriever: str = "lexical"
    reference_index_field: str = "source"
    reference_target_field: str = "target"
    reference_options: dict[str, Any] = field(default_factory=dict)


_RECIPE_DEFAULTS = _Recipe()


def _load_recipe(path: Path) -> _Recipe:
    """Parse ``recipe.toml`` through the strict ``core.config`` readers.

    Every field goes through ``read_string``/``read_bool``/``as_table``, not ``str()``/``bool()``/
    ``dict()``. Raw coercion made two silent failures reachable here that the readers exist
    specifically to refuse: ``use_memory = "false"`` became ``bool("false")`` -> **True**, quietly
    inverting the gate, and ``output_schema_options = 5`` raised a bare ``TypeError`` from inside
    ``dict()`` that escaped ``main()``'s ``except RagkitError`` as an unstructured traceback.
    """
    data = load_toml(path, what="recipe file")
    reject_unknown(data, {"task", "validator", "reference"}, label="the recipe file", path=path)
    task = reject_unknown(data.get("task", {}),
                          {"output_schema", "output_schema_options", "input_label", "stand_in",
                           "use_memory"}, label="[task]", path=path)
    validators = tuple(
        (str(entry["kind"]), {k: v for k, v in entry.items() if k != "kind"})
        for entry in tables(data, "validator", path=path)
        if "kind" in entry or _missing_kind(path))
    reference = reject_unknown(
        data.get("reference", {}),
        {"file", "retriever", "index_field", "target_field", "options"},
        label="[reference]", path=path)
    d = _RECIPE_DEFAULTS
    return _Recipe(
        output_schema=read_string(task, "output_schema", d.output_schema, label="[task]",
                                  path=path),
        output_schema_options=as_table(task.get("output_schema_options", {}),
                                       label="[task].output_schema_options", path=path),
        validators=validators,
        input_label=read_string(task, "input_label", d.input_label, label="[task]", path=path),
        stand_in=read_string(task, "stand_in", d.stand_in, label="[task]", path=path),
        use_memory=read_bool(task, "use_memory", d.use_memory, label="[task]", path=path),
        reference_file=read_string(reference, "file", d.reference_file, label="[reference]",
                                   path=path),
        reference_retriever=read_string(reference, "retriever", d.reference_retriever,
                                        label="[reference]", path=path),
        reference_index_field=read_string(reference, "index_field", d.reference_index_field,
                                          label="[reference]", path=path),
        reference_target_field=read_string(reference, "target_field", d.reference_target_field,
                                           label="[reference]", path=path),
        reference_options=as_table(reference.get("options", {}), label="[reference].options",
                                   path=path))


def _missing_kind(path: Path) -> bool:
    raise ConfigError("a [[validator]] entry needs a 'kind'", path=path)


def _build_reference(recipe: _Recipe, config_dir: Path, storage: Storage) -> Retriever | None:
    """Build a reference retriever from config. A ``[reference].retriever`` that is a dotted path or
    an entry-point name is a **custom retriever with its own backend**, resolved through the
    ``RETRIEVERS`` registry and built without our corpus. The built-in ``"lexical"`` is built by
    importing the JSONL corpus into ``storage.pairings`` (``index_field``/``target_field`` are what
    a pairing's ``source``/``target`` are read from). A custom retriever that must read *our* corpus
    is corpus-stateful — inject it via ``assemble(retriever=...)`` instead."""
    spec = recipe.reference_retriever
    # A custom retriever -- named by dotted path, by a registered name, or (with no corpus file to
    # import) by an entry-point name -- is resolved through the registry, which raises a clear
    # RegistryError on a genuinely unknown name. The old `spec in available()` gate missed
    # entry-point names (they resolve lazily in create(), not in available()), so an unresolvable
    # name with no [reference].file silently returned None and left the harness with no retriever.
    if spec != "lexical" and (":" in spec or spec in RETRIEVERS.available()
                              or not recipe.reference_file):
        return RETRIEVERS.create(spec, recipe.reference_options)
    if not recipe.reference_file:
        return None
    path = (config_dir / recipe.reference_file).resolve()
    if not path.is_file():
        raise CliError("reference corpus not found", path=path)
    retrievers = _load_pairing_retrievers(recipe, path, storage)
    if spec == "lexical":
        return retrievers.lexical_retriever()
    raise CliError(
        f"reference.retriever={spec!r} is not built-in. Use 'lexical', a dotted path / entry point "
        f"to a corpus-free retriever, or inject a corpus-backed one via assemble(retriever=...). "
        f"('dense' needs an embedding endpoint wired by a recipe.)", path=path)


def _load_pairing_retrievers(recipe: _Recipe, path: Path, storage: Storage, *,
                             vector: VectorIndex | None = None,
                             embedder: EmbeddingClient | None = None) -> PairingRetrievers:
    """Import the reference JSONL into ``storage.pairings`` (resumable; a no-op past the first
    call) and return the retriever-builder over it."""
    if storage.pairings is None:
        raise CliError(
            "the recipe has a [reference].file to retrieve from but storage.toml sets no "
            "[pairings] store to import it into", path=path)
    try:
        import_reference(path, storage.pairings, index_field=recipe.reference_index_field,
                         target_field=recipe.reference_target_field, vector=vector,
                         embedder=embedder)
    except ReferenceImportError as exc:
        raise CliError(exc.reason, path=path) from exc
    return PairingRetrievers(storage.pairings, vector=vector, embedder=embedder)


def _build_retrieval(settings: RetrievalSettings, recipe: _Recipe, config_dir: Path,
                     storage: Storage, pool: ModelPool,
                     client_factory: ClientFactory | None) -> Retriever:
    """Assemble the retriever configured by ``retrieval.toml``: a lexical, dense, or hybrid stack
    over the reference corpus, with the per-arm floors, candidate pool, MMR trade-off, and rerank
    model from config. Embedding and rerank clients route over the endpoint of their named model
    (``models.toml``), through ``client_factory`` so a test drives them with no server."""
    if not recipe.reference_file:
        raise CliError("retrieval.toml is present but the recipe sets no [reference].file to "
                       "build the corpus from")
    path = (config_dir / recipe.reference_file).resolve()
    if not path.is_file():
        raise CliError("reference corpus not found", path=path)

    embedder = (_embedding_client(settings, pool, client_factory)
                if settings.needs_embedding else None)
    vector = storage.vector if settings.needs_embedding else None
    if settings.needs_embedding and vector is None:
        raise CliError(f"retrieval.kind={settings.kind!r} needs a [vector] store in storage.toml "
                       f"(built with the embedding model's output dimension)")

    retrievers = _load_pairing_retrievers(recipe, path, storage, vector=vector, embedder=embedder)

    # The single-arm stacks take no per-arm floor, candidate pool, MMR trade-off, or reranker: those
    # tune fusion, and `load_retrieval` refuses them for any kind but hybrid, so there is nothing
    # here that a user could have set and this branch would drop (see `tuning._APPLICABILITY`).
    if settings.kind == "lexical":
        return retrievers.lexical_retriever()
    if settings.kind == "dense":
        return retrievers.dense_retriever()
    reranker = _rerank_client(settings, pool, client_factory) if settings.rerank_enabled else None
    return retrievers.hybrid_retriever(
        reranker=reranker, candidate_pool=settings.candidate_pool, mmr_lambda=settings.mmr_lambda,
        lexical_min_score=settings.lexical_min_score, dense_min_score=settings.dense_min_score)


def _embedding_client(settings: RetrievalSettings, pool: ModelPool,
                      client_factory: ClientFactory | None) -> EmbeddingClient:
    spec = pool.model_spec(settings.embedding_model)
    if spec.kind != "embedding":
        raise CliError(f"[retrieval.dense].model {settings.embedding_model!r} is a {spec.kind!r} "
                       f"model, not an embedding model")
    endpoint = pool.endpoint(spec.endpoint)
    if not endpoint.provider_profile.supports("embedding"):
        raise CliError(
            f"[retrieval.dense].model {settings.embedding_model!r} runs on endpoint "
            f"{spec.endpoint!r}, whose provider {endpoint.provider!r} does not serve embeddings; "
            f"use an embedding-capable endpoint or a lexical-only retrieval config")
    client = client_factory(endpoint.base_url, endpoint.timeout_seconds) if client_factory else None
    return EmbeddingClient(base_url=endpoint.base_url, model=spec.model_id,
                           timeout_seconds=endpoint.timeout_seconds,
                           max_retries=endpoint.max_retries,
                           retry_backoff_seconds=endpoint.retry_backoff_seconds, client=client)


def _rerank_client(settings: RetrievalSettings, pool: ModelPool,
                   client_factory: ClientFactory | None) -> RerankClient:
    spec = pool.model_spec(settings.rerank_model)
    if spec.kind != "rerank":
        raise CliError(f"[retrieval.rerank].model {settings.rerank_model!r} is a {spec.kind!r} "
                       f"model, not a rerank model")
    endpoint = pool.endpoint(spec.endpoint)
    if not endpoint.provider_profile.supports("rerank"):
        raise CliError(
            f"[retrieval.rerank].model {settings.rerank_model!r} runs on endpoint "
            f"{spec.endpoint!r}, whose provider {endpoint.provider!r} has no rerank endpoint "
            f"(Ollama, for one, does not); serve the reranker on a llamacpp-router or "
            f"openai-compatible endpoint, or disable reranking")
    client = client_factory(endpoint.base_url, endpoint.timeout_seconds) if client_factory else None
    return RerankClient(base_url=endpoint.base_url, model=spec.model_id,
                        score_scale=settings.rerank_score_scale, max_retries=endpoint.max_retries,
                        retry_backoff_seconds=endpoint.retry_backoff_seconds, client=client)
