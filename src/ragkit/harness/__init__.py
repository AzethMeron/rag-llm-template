"""The harness: the bounded produce → mechanical check → review panel → revise loop, the
personas that drive it, the rule set and validators that check its output, the declarative context
builder, the run-local output memory, and the resumable batch runner.

Task-agnostic: what the output *is* (a translation, a SQL statement, a filled form) is an
:class:`~ragkit.core.ports.OutputSchema`; the reviewers, budgets, rules and context blocks are all
configuration. Depends on the ``llm`` and ``core`` layers, never on a concrete store or retriever.
"""
from __future__ import annotations

from .agents import Attempt, Harness, Outcome, Review, learn_memory
from .context import ContextAssembler, load_context
from .memory import OutputMemory
from .roles import Leniency, Limits, Panel, Persona, load_panel
from .rules import RuleSet
from .runner import (
    Progress,
    RunnerError,
    catalog_order,
    completed_ids,
    group_duplicates,
    pending_records,
    run_batch,
)
from .schemas import OUTPUT_SCHEMAS, FormField, FormSchema, JsonFieldSchema
from .validators import VALIDATORS, ValidatorPipeline, blocking, check_mechanical

__all__ = [
    # harness loop
    "Harness", "Outcome", "Review", "Attempt", "learn_memory",
    # personas
    "Panel", "Persona", "Limits", "Leniency", "load_panel",
    # policy
    "RuleSet", "ValidatorPipeline", "VALIDATORS", "check_mechanical", "blocking",
    # context
    "ContextAssembler", "load_context",
    # output schema
    "OUTPUT_SCHEMAS", "JsonFieldSchema", "FormSchema", "FormField",
    # memory
    "OutputMemory",
    # runner
    "run_batch", "Progress", "RunnerError", "pending_records", "completed_ids",
    "group_duplicates", "catalog_order",
]
