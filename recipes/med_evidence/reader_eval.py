"""PubMedQA-standard *reader* evaluation — the accuracy directly comparable to the published
benchmark.

The default eval (``eval.py``) makes the model *retrieve* the answer-bearing abstract among ~600k
distractors and *then* decide — a materially harder task than PubMedQA, whose leaderboard hands the
gold abstract to the model as context. This reader eval reproduces that benchmark setup: a
:class:`SelfAbstractRetriever` returns each question's OWN gold abstract as the sole context, so the
score isolates *decision* quality and can be read against the published numbers (majority-class
≈ 0.55; human ≈ 0.78; strong instruction-tuned LLMs ≈ 0.75–0.80).

Same recipe, same panel and grounding — only the retrieval is swapped (injected via
``assemble(retriever=...)``), so the two evals differ by exactly one variable: retrieval.

Run (needs a served model per the recipe's ``models.toml``)::

    PYTHONPATH=src:. python -m recipes.med_evidence.reader_eval \\
        --config recipes/med_evidence/config \\
        --abstracts recipes/med_evidence/data/abstracts.jsonl \\
        --heldout recipes/med_evidence/data/heldout.jsonl \\
        --gold recipes/med_evidence/data/gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ragkit.cli.app import assemble
from ragkit.core.ports import Retrieved
from ragkit.core.records import Record, export_jsonl
from ragkit.harness import run_batch
from ragkit.llm.pool import ClientFactory
from ragkit.store.run.sqlite import SqliteRunStore

from .eval import EvalError, evaluate, load_gold


class SelfAbstractRetriever:
    """Returns each question's own gold abstract as the single retrieved passage, so the decision is
    graded against the correct context (the PubMedQA reader setup) rather than on retrieval. A query
    it does not know returns nothing — the grounding guard then blocks, as for a real miss."""

    def __init__(self, abstract_of_question: Mapping[str, str]) -> None:
        self._by_question = dict(abstract_of_question)

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        del k, min_score  # part of the Retriever port; a self-retriever ignores depth/floor
        text = self._by_question.get(query.strip())
        return (Retrieved("gold-0", text, 1.0),) if text else ()


def load_reader_set(abstracts: Path, heldout: Path) -> tuple[list[Record], dict[str, str]]:
    """Build the held-out records and a question→own-abstract map (joined on the PMID) for the
    reader run. Raises :class:`EvalError` if a question has no abstract."""
    if not abstracts.is_file():
        raise EvalError(f"abstracts file not found: {abstracts}")
    if not heldout.is_file():
        raise EvalError(f"heldout file not found: {heldout}")
    by_pmid: dict[str, str] = {}
    for line in abstracts.read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_pmid[str(row["id"])] = str(row["text"])
    records: list[Record] = []
    by_question: dict[str, str] = {}
    for line in heldout.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("meta", {})
        pmid = str(meta.get("pmid", row["record_id"]))
        if pmid not in by_pmid:
            raise EvalError(f"no abstract for question {row['record_id']!r} (pmid {pmid})")
        by_question[str(row["source"]).strip()] = by_pmid[pmid]
        records.append(Record(record_id=str(row["record_id"]), source=str(row["source"]),
                              meta=meta))
    return records, by_question


def main(argv: Sequence[str] | None = None, *, client_factory: ClientFactory | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PubMedQA-standard reader accuracy (the gold abstract is given as context).")
    parser.add_argument("--config", type=Path, required=True, help="recipe config directory")
    parser.add_argument("--abstracts", type=Path, required=True,
                        help="abstracts.jsonl (id -> text)")
    parser.add_argument("--heldout", type=Path, required=True, help="held-out questions (JSONL)")
    parser.add_argument("--gold", type=Path, required=True, help="gold decisions (JSONL)")
    parser.add_argument("--journal", type=Path, default=Path("work/med_reader.jsonl"),
                        help="where to write the run journal")
    args = parser.parse_args(argv)
    try:
        records, by_question = load_reader_set(args.abstracts, args.heldout)
        gold = load_gold(args.gold)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.journal.parent.mkdir(parents=True, exist_ok=True)
    run_db = args.journal.with_name(args.journal.stem + ".run.db")
    store = SqliteRunStore(str(run_db))
    store.add_records(records)
    assembled = assemble(args.config, retriever=SelfAbstractRetriever(by_question),
                         client_factory=client_factory)
    run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
    export_jsonl(store, args.journal)  # journal.jsonl artifact, for read_journal/eval.py parity

    produced: dict[str, str | None] = {
        result.record.record_id:
            (result.record.output if result.record.status.is_injectable else None)
        for result in store.results()}
    report = evaluate([(rid, produced.get(rid), decision) for rid, decision in gold.items()])
    print(f"[reader / gold-context] decided {report.produced}/{report.total} | "
          f"accuracy {report.accuracy:.3f} | "
          f"answered accuracy (over {report.answered}) {report.answered_accuracy:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
