"""The framework contract: records, the component registry, the ports, and the stdlib-only
primitives (config loading, display width, placeholders, terminology, rule violations).

Everything here is task-agnostic and depends on nothing outside the standard library, so the
contract can never break because a machine-learning or storage dependency was upgraded — a
guarantee ``tests/test_boundaries.py`` enforces by importing this package with the optional
dependencies made unimportable. The layers above (``store``, ``ingest``, ``retrieve``, ``llm``,
``harness``) depend on these ports; nothing here depends on them.
"""
from __future__ import annotations

from .config import (
    ConfigError,
    as_table,
    load_toml,
    read_bool,
    read_float,
    read_int,
    read_string,
    read_string_list,
    reject_unknown,
    tables,
)
from .errors import RagkitError
from .lexicon import Entry, LexiconError, read_lexicon, relevant_entries, write_lexicon
from .placeholders import PLACEHOLDER, placeholder_indices
from .ports import (
    Backend,
    Chunk,
    Chunker,
    ContextBlock,
    Document,
    Embedder,
    Extractor,
    Filter,
    FilterOp,
    LexicalIndex,
    Message,
    OutputSchema,
    Predicate,
    Provider,
    Reranker,
    Retrieved,
    Retriever,
    SchemaIntrospector,
    Sink,
    Source,
    SqlStore,
    StructuredRequest,
    Validator,
    VectorIndex,
)
from .records import (
    CatalogError,
    Record,
    Status,
    export_jsonl,
    import_jsonl,
    make_record_id,
    merge_journal,
    read_catalog,
    read_journal,
    write_catalog,
)
from .registry import Registry, RegistryError
from .rules import Severity, Violation, blocking
from .width import display_columns

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # errors
    "RagkitError",
    # records
    "Record", "Status", "CatalogError", "make_record_id",
    "read_catalog", "write_catalog", "read_journal", "merge_journal",
    "import_jsonl", "export_jsonl",
    # config
    "ConfigError", "load_toml", "as_table", "reject_unknown", "tables",
    "read_int", "read_float", "read_bool", "read_string", "read_string_list",
    # registry
    "Registry", "RegistryError",
    # rules
    "Severity", "Violation", "blocking",
    # lexicon
    "Entry", "LexiconError", "read_lexicon", "write_lexicon", "relevant_entries",
    # width / placeholders
    "display_columns", "PLACEHOLDER", "placeholder_indices",
    # value types
    "Message", "StructuredRequest", "Document", "Chunk", "Retrieved",
    "Filter", "FilterOp", "Predicate",
    # ports
    "Source", "Sink", "Extractor", "Chunker", "Embedder",
    "VectorIndex", "LexicalIndex", "SqlStore", "SchemaIntrospector",
    "Retriever", "Reranker", "ContextBlock", "Validator", "OutputSchema",
    "Backend", "Provider",
]
