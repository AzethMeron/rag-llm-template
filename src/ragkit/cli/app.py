"""Assemble a run from a configuration directory, and execute it.

A run is described entirely by config files in one directory — ``models.toml``, ``personas.toml``,
``rules.toml``, ``context.toml``, an optional ``storage.toml``, and a ``recipe.toml`` that names the
task's output schema, its extra validators, its input label, and (optionally) a reference corpus to
retrieve from. Every component is resolved through its registry, so a user's own driver, validator,
block, or schema is selected the same way a built-in is — no framework change.

The assembly does all of its config reading and consistency checks (including the model-pool
capacity/thrash guard) *before* the server is contacted, so a misconfiguration is reported for what
it is rather than masked by a "no server" error. ``client_factory`` is injectable so a test drives
the whole run against an in-memory transport with no server.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragkit.core.config import ConfigError, load_toml, reject_unknown, tables
from ragkit.core.lexicon import read_lexicon
from ragkit.core.ports import Retriever, SqlStore
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
from ragkit.ingest.corpus import Corpus, CorpusItem
from ragkit.llm.pool import ClientFactory, ModelPool, load_models
from ragkit.store import Storage, load_storage
from ragkit.store.lexical.fts5 import Fts5Index


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
             extra_validators: list[Any] | None = None) -> Assembled:
    """Build a :class:`Assembled` run from the config files in ``config_dir``.

    ``substitutions`` fill the ``{placeholder}`` tokens in persona instructions (the language pair
    for translation, say). ``extra_validators`` are task-specific validator instances a recipe
    passes in directly (in addition to any named in ``recipe.toml``).
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
    retriever = _build_reference(recipe, config_dir, storage)
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
    output_schema: str
    output_schema_options: dict[str, Any]
    validators: tuple[tuple[str, dict[str, Any]], ...]
    input_label: str
    stand_in: str
    use_memory: bool
    reference_file: str
    reference_retriever: str
    reference_index_field: str
    reference_display_field: str


def _load_recipe(path: Path) -> _Recipe:
    data = load_toml(path, what="recipe file")
    reject_unknown(data, {"task", "validator", "reference"}, label="the recipe file", path=path)
    task = reject_unknown(data.get("task", {}),
                          {"output_schema", "output_schema_options", "input_label", "stand_in",
                           "use_memory"}, label="[task]", path=path)
    validators = tuple(
        (str(entry["kind"]), {k: v for k, v in entry.items() if k != "kind"})
        for entry in tables(data, "validator", path=path)
        if "kind" in entry or _missing_kind(path))
    reference = reject_unknown(data.get("reference", {}),
                               {"file", "retriever", "index_field", "display_field"},
                               label="[reference]", path=path)
    return _Recipe(
        output_schema=str(task.get("output_schema", "json_field")),
        output_schema_options=dict(task.get("output_schema_options", {})),
        validators=validators,
        input_label=str(task.get("input_label", "Input to act on:")),
        stand_in=str(task.get("stand_in", "they")),
        use_memory=bool(task.get("use_memory", False)),
        reference_file=str(reference.get("file", "")),
        reference_retriever=str(reference.get("retriever", "lexical")),
        reference_index_field=str(reference.get("index_field", "source")),
        reference_display_field=str(reference.get("display_field", "")))


def _missing_kind(path: Path) -> bool:
    raise ConfigError("a [[validator]] entry needs a 'kind'", path=path)


def _build_reference(recipe: _Recipe, config_dir: Path, storage: Storage) -> Retriever | None:
    """Build a reference retriever from a JSONL corpus, if the recipe configures one. Each line is a
    JSON object; ``index_field`` is what matching happens on, ``display_field`` (or the whole line)
    is what a hit shows."""
    if not recipe.reference_file:
        return None
    path = (config_dir / recipe.reference_file).resolve()
    if not path.is_file():
        raise CliError("reference corpus not found", path=path)
    corpus = _load_corpus(path, recipe, storage)
    if recipe.reference_retriever == "lexical":
        return corpus.lexical_retriever()
    raise CliError(
        f"reference.retriever={recipe.reference_retriever!r} needs an embedding endpoint wired in; "
        f"only 'lexical' is buildable from config alone in this build", path=path)


def _load_corpus(path: Path, recipe: _Recipe, storage: Storage) -> Corpus:
    import json
    lexical = storage.lexical or Fts5Index()
    embedder = None  # dense reference retrieval is wired by a recipe, not from config alone here
    corpus = Corpus(lexical=lexical, embedder=embedder)
    items: list[CorpusItem] = []
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CliError(f"line {line_no}: invalid JSON in reference corpus: {exc}",
                           path=path) from exc
        index_text = str(record.get(recipe.reference_index_field, ""))
        if not index_text:
            continue
        display = (str(record[recipe.reference_display_field])
                   if recipe.reference_display_field else _default_display(record))
        items.append(CorpusItem(chunk_id=f"ref-{line_no}", index_text=index_text,
                                display_text=display, meta=record))
    corpus.add_all(items)
    return corpus


def _default_display(record: dict[str, Any]) -> str:
    if "source" in record and "target" in record:
        return f"{record['source']} -> {record['target']}"
    return str(record.get("text", record))
