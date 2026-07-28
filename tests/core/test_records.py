"""Record invariants and the durable catalogue/journal, including the failure paths they
defend against (a torn journal, an orphaned result, a mid-write crash)."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ragkit.core.records import (
    CatalogError,
    Record,
    Status,
    fsync_directory,
    make_record_id,
    merge_journal,
    read_catalog,
    read_journal,
    write_catalog,
)


class TestStatus:
    def test_injectable_statuses(self) -> None:
        assert Status.VERIFIED.is_injectable
        assert Status.PRODUCED.is_injectable
        assert not Status.REJECTED.is_injectable
        assert not Status.PENDING.is_injectable
        assert not Status.SKIPPED.is_injectable

    def test_done_statuses(self) -> None:
        assert Status.VERIFIED.is_done
        assert Status.SKIPPED.is_done
        assert not Status.PRODUCED.is_done
        assert not Status.REJECTED.is_done

    def test_value_is_the_member(self) -> None:
        assert Status("verified") is Status.VERIFIED


class TestRecordInvariants:
    def test_span_end_before_start_is_refused(self) -> None:
        with pytest.raises(CatalogError, match="span_end 1 precedes span_start 5"):
            Record(record_id="x", source="s", span_start=5, span_end=1)

    def test_span_equal_is_allowed(self) -> None:
        Record(record_id="x", source="s", span_start=3, span_end=3)

    def test_injectable_without_output_is_refused(self) -> None:
        with pytest.raises(CatalogError, match="injectable but output is None"):
            Record(record_id="x", source="s", status=Status.VERIFIED)

    def test_injectable_with_output_is_allowed(self) -> None:
        Record(record_id="x", source="s", status=Status.PRODUCED, output="o")

    def test_non_mapping_meta_is_refused(self) -> None:
        with pytest.raises(CatalogError, match="meta must be a mapping"):
            Record(record_id="x", source="s", meta=[("a", 1)])  # type: ignore[arg-type]

    def test_meta_is_frozen_and_defensively_copied(self) -> None:
        original = {"a": 1}
        record = Record(record_id="x", source="s", meta=original)
        original["a"] = 2  # mutating the caller's dict must not affect the record
        assert record.meta["a"] == 1
        with pytest.raises(TypeError):
            record.meta["a"] = 3  # type: ignore[index]

    def test_fields_are_immutable(self) -> None:
        record = Record(record_id="x", source="s")
        with pytest.raises(FrozenInstanceError):
            record.source = "t"  # type: ignore[misc]

    def test_with_output_produces_a_settled_copy(self) -> None:
        record = Record(record_id="x", source="s")
        done = record.with_output("o", Status.VERIFIED)
        assert done.output == "o" and done.status is Status.VERIFIED
        assert record.output is None  # original untouched


class TestRecordJson:
    def test_round_trip_preserves_everything(self) -> None:
        record = Record(
            record_id="id1", source="hello", output="cześć", status=Status.VERIFIED,
            rel_path="f.txt", line_no=7, span_start=10, span_end=20,
            notes=("a", "b"), meta={"placeholders": [0, 1], "nested": {"k": "v"}})
        clone = Record.from_json(record.to_json())
        assert clone == record
        assert clone.meta["nested"]["k"] == "v"
        assert clone.notes == ("a", "b")

    def test_minimal_record_needs_only_id_and_source(self) -> None:
        clone = Record.from_json(json.dumps({"record_id": "x", "source": "s"}))
        assert clone.record_id == "x" and clone.source == "s"
        assert clone.status is Status.PENDING

    def test_invalid_json_is_a_catalog_error(self) -> None:
        with pytest.raises(CatalogError, match="invalid JSON"):
            Record.from_json("{not json", path=Path("c.jsonl"), line_no=3)

    def test_non_object_json_is_refused(self) -> None:
        with pytest.raises(CatalogError, match="must be a JSON object"):
            Record.from_json("[1, 2, 3]")

    def test_missing_required_fields_are_named(self) -> None:
        with pytest.raises(CatalogError, match=r"missing fields \['source'\]"):
            Record.from_json(json.dumps({"record_id": "x"}))

    def test_non_object_meta_is_refused(self) -> None:
        with pytest.raises(CatalogError, match="meta must be a JSON object"):
            Record.from_json(json.dumps({"record_id": "x", "source": "s", "meta": [1]}))

    def test_unknown_status_is_a_bad_field_value(self) -> None:
        with pytest.raises(CatalogError, match="bad field value"):
            Record.from_json(json.dumps({"record_id": "x", "source": "s", "status": "nope"}))


class TestMakeRecordId:
    def test_is_deterministic(self) -> None:
        assert make_record_id("f", "text", 0) == make_record_id("f", "text", 0)

    def test_ordinal_separates_repeats(self) -> None:
        assert make_record_id("f", "text", 0) != make_record_id("f", "text", 1)

    def test_is_sixteen_hex_chars(self) -> None:
        rid = make_record_id("f", "text", 0)
        assert len(rid) == 16 and all(c in "0123456789abcdef" for c in rid)


class TestCatalogRoundTrip:
    def test_write_then_read(self, tmp_path: Path) -> None:
        records = [Record(record_id=str(i), source=f"s{i}") for i in range(3)]
        path = tmp_path / "sub" / "cat.jsonl"
        assert write_catalog(records, path) == 3
        assert [r.record_id for r in read_catalog(path)] == ["0", "1", "2"]

    def test_read_missing_catalogue_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogError, match="catalogue not found"):
            list(read_catalog(tmp_path / "absent.jsonl"))

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        path.write_text(
            Record(record_id="a", source="s").to_json() + "\n\n"
            + Record(record_id="b", source="s").to_json() + "\n", encoding="utf-8")
        assert [r.record_id for r in read_catalog(path)] == ["a", "b"]

    def test_write_is_atomic_no_partial_on_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="good", source="s")], path)

        def failing() -> list[Record]:
            yield Record(record_id="a", source="s")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            write_catalog(failing(), path)
        # The original file is intact and no .partial sibling was left behind.
        assert [r.record_id for r in read_catalog(path)] == ["good"]
        assert not (tmp_path / "c.jsonl.partial").exists()


class TestJournal:
    def test_absent_journal_yields_nothing(self, tmp_path: Path) -> None:
        assert list(read_journal(tmp_path / "none.jsonl")) == []

    def test_reads_in_write_order(self, tmp_path: Path) -> None:
        path = tmp_path / "j.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for rid in ("a", "b"):
                handle.write(Record(record_id=rid, source="s").to_json() + "\n")
        assert [r.record_id for r in read_journal(path)] == ["a", "b"]

    def test_torn_final_record_is_tolerated(self, tmp_path: Path) -> None:
        # The shape a process killed mid-write leaves: a complete record, then a truncated one
        # with no trailing newline. The good record survives; the torn tail is dropped.
        path = tmp_path / "j.jsonl"
        path.write_text(
            Record(record_id="a", source="s").to_json() + "\n" + '{"record_id": "b", "sou',
            encoding="utf-8")
        assert [r.record_id for r in read_journal(path)] == ["a"]

    def test_malformed_interior_record_raises(self, tmp_path: Path) -> None:
        # A malformed line that is NOT the final one is corruption, not a torn write: skipping
        # it would discard a completed result while reporting success.
        path = tmp_path / "j.jsonl"
        path.write_text(
            "{bad}\n" + Record(record_id="b", source="s").to_json() + "\n", encoding="utf-8")
        with pytest.raises(CatalogError, match="corrupt journal record"):
            list(read_journal(path))

    def test_a_malformed_final_record_with_newline_still_raises(self, tmp_path: Path) -> None:
        # The tolerance is only for a torn write (no trailing newline). A malformed final line
        # that DID finish writing (has its newline) is corruption.
        path = tmp_path / "j.jsonl"
        path.write_text("{bad}\n", encoding="utf-8")
        with pytest.raises(CatalogError, match="corrupt journal record"):
            list(read_journal(path))

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "j.jsonl"
        path.write_text("\n" + Record(record_id="a", source="s").to_json() + "\n\n",
                        encoding="utf-8")
        assert [r.record_id for r in read_journal(path)] == ["a"]


class TestMergeJournal:
    def test_later_results_win_and_count(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="a", source="s"),
                       Record(record_id="b", source="s")], catalog)
        journal = tmp_path / "j.jsonl"
        journal.write_text(
            Record(record_id="a", source="s", status=Status.VERIFIED, output="o").to_json()
            + "\n", encoding="utf-8")
        output = tmp_path / "merged.jsonl"
        written, updated = merge_journal(catalog, journal, output)
        assert (written, updated) == (2, 1)
        merged = {r.record_id: r for r in read_catalog(output)}
        assert merged["a"].status is Status.VERIFIED and merged["a"].output == "o"
        assert merged["b"].status is Status.PENDING

    def test_missing_journal_raises(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="a", source="s")], catalog)
        with pytest.raises(CatalogError, match="journal not found"):
            merge_journal(catalog, tmp_path / "none.jsonl", tmp_path / "o.jsonl")

    def test_orphan_result_is_refused(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="a", source="s")], catalog)
        journal = tmp_path / "j.jsonl"
        journal.write_text(
            Record(record_id="ghost", source="s", status=Status.VERIFIED, output="o").to_json()
            + "\n", encoding="utf-8")
        with pytest.raises(CatalogError, match="no matching record"):
            merge_journal(catalog, journal, tmp_path / "o.jsonl")


class TestFsyncDirectory:
    def test_survives_a_normal_directory(self, tmp_path: Path) -> None:
        fsync_directory(tmp_path)  # no exception

    def test_survives_an_unopenable_path(self, tmp_path: Path) -> None:
        # A path that cannot be opened as a directory must not raise -- some filesystems do not
        # support it, and the rename is durable anyway.
        fsync_directory(tmp_path / "does-not-exist")

    def test_survives_fsync_failing_on_an_opened_directory(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The directory opens, but fsync on it is unsupported (some filesystems reject it).
        # That must be swallowed, not raised: the rename is already durable.
        import os

        def failing_fsync(_fd: int) -> None:
            raise OSError("fsync unsupported on this directory")

        monkeypatch.setattr(os, "fsync", failing_fsync)
        fsync_directory(tmp_path)  # no exception
