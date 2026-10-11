import csv
import hashlib
import json
from pathlib import Path

import pytest

from routetune.config import ExperimentConfig
from routetune.data import (
    DataError,
    DatasetSplits,
    LoadedSource,
    Provenance,
    Record,
    SourceSpec,
    build_splits,
    fetch_source,
    normalize_text,
    read_csv,
    validate_splits,
)


CONFIG = ExperimentConfig(
    labels=("alpha", "beta"),
    seed=42,
    train_per_class=2,
    validation_per_class=1,
    test_per_class=1,
)


def write_csv(path: Path, rows: list[tuple[str, str]], columns=("text", "category")):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def loaded(source: str, rows: list[tuple[str, str]]) -> LoadedSource:
    records = tuple(Record(text, category, source) for text, category in rows)
    return LoadedSource(records, Provenance(source, f"https:///{source}", "0" * 64, 1, "test"))


def pools():
    train = [
        (f"{label} development {index}", label)
        for label in CONFIG.labels
        for index in range(5)
    ]
    test = [
        (f"{label} held out {index}", label)
        for label in CONFIG.labels
        for index in range(3)
    ]
    return loaded("official_train", train), loaded("official_test", test)


def test_normalize_text_uses_casefold_and_whitespace_normalization():
    assert normalize_text("  Straße\n  PAYMENT\t") == "strasse payment"


def test_same_seed_produces_same_balanced_selection():
    train, test = pools()
    first = build_splits(train, test, CONFIG)
    second = build_splits(train, test, CONFIG)
    assert first == second
    assert [record.text for record in first.train] == [
        record.text for record in second.train
    ]
    for split, expected in ((first.train, 2), (first.validation, 1), (first.test, 1)):
        assert {label: sum(row.category == label for row in split) for label in CONFIG.labels} == {
            label: expected for label in CONFIG.labels
        }


def test_duplicate_rows_are_not_selected_twice():
    train, test = pools()
    duplicated = LoadedSource(train.records + (train.records[0],), train.provenance)
    splits = build_splits(duplicated, test, CONFIG)
    normalized = [normalize_text(record.text) for record in splits.train + splits.validation]
    assert len(normalized) == len(set(normalized))


def test_cross_split_leakage_is_rejected():
    train = Record("same text", "alpha", "official_train")
    validation = Record(" SAME\tTEXT ", "alpha", "official_train")
    test = Record("different", "alpha", "official_test")
    splits = DatasetSplits((train,), (validation,), (test,))
    one_label = ExperimentConfig(("alpha",), 42, 1, 1, 1)
    with pytest.raises(DataError, match="duplicate normalized text"):
        validate_splits(splits, one_label)


def test_official_train_test_boundary_is_preserved():
    train, test = pools()
    splits = build_splits(train, test, CONFIG)
    assert {row.source for row in splits.train + splits.validation} == {"official_train"}
    assert {row.source for row in splits.test} == {"official_test"}
    with pytest.raises(DataError, match="training pool"):
        build_splits(test, train, CONFIG)


@pytest.mark.parametrize(
    ("columns", "rows", "message"),
    [
        (("text",), [("hello",)], "missing required columns"),
        (("text", "category"), [("", "alpha")], "empty required value"),
        (("text", "category"), [("hello", "")], "empty required value"),
        (("text", "category"), [("hello", "alpha", "extra")], "malformed"),
    ],
)
def test_malformed_csv_is_rejected(tmp_path, columns, rows, message):
    path = tmp_path / "bad.csv"
    write_csv(path, rows, columns)
    with pytest.raises(DataError, match=message):
        read_csv(path, "fixture")


def test_insufficient_unique_samples_fails_clearly():
    train, test = pools()
    too_small = loaded(
        "official_train",
        [("one", "alpha"), (" ONE ", "alpha"), ("two", "beta")],
    )
    with pytest.raises(DataError, match="unique 'alpha' rows"):
        build_splits(too_small, test, CONFIG)


def test_source_provenance_matches_content(tmp_path):
    payload = b"text,category\nhello,alpha\n"
    spec = SourceSpec("official_train", "https://example.test/train.csv", "train.csv")

    class Response:
        headers = {"ETag": '"fixture"', "Last-Modified": "yesterday"}

        def __init__(self):
            self.remaining = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, _size):
            chunk, self.remaining = self.remaining, b""
            return chunk

    source = fetch_source(spec, tmp_path, opener=lambda *_args, **_kwargs: Response())
    assert source.provenance.url == spec.url
    assert source.provenance.sha256 == hashlib.sha256(payload).hexdigest()
    assert source.provenance.size_bytes == len(payload)
    assert source.provenance.etag == '"fixture"'
    assert source.records[0].source == "official_train"


def test_valid_cache_is_reused_without_network(tmp_path):
    path = tmp_path / "train.csv"
    write_csv(path, [("hello", "alpha")])
    spec = SourceSpec("official_train", "https://example.test/train.csv", path.name)
    metadata = {
        "source": spec.name,
        "url": spec.url,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("network should not be called")

    source = fetch_source(spec, tmp_path, offline=True, opener=fail_if_called)
    assert source.provenance.cache_status == "validated_cache"


def test_corrupt_cache_is_replaced_only_with_valid_download(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("corrupt", encoding="utf-8")
    path.with_suffix(".metadata.json").write_text(
        json.dumps({"url": "wrong", "sha256": "wrong", "size_bytes": 7}),
        encoding="utf-8",
    )
    payload = b"text,category\nrecovered,alpha\n"
    spec = SourceSpec("official_train", "https://example.test/train.csv", path.name)

    class Response:
        headers = {}

        def __enter__(self):
            self.remaining = payload
            return self

        def __exit__(self, *args):
            return None

        def read(self, _size):
            chunk, self.remaining = self.remaining, b""
            return chunk

    source = fetch_source(spec, tmp_path, opener=lambda *_args, **_kwargs: Response())
    assert source.records[0].text == "recovered"
    assert source.provenance.cache_status == "downloaded_and_validated"
    assert not list(tmp_path.glob("*.part"))


def test_network_failure_is_explicit_and_does_not_promote_partial_file(tmp_path):
    spec = SourceSpec("official_train", "https://example.test/train.csv", "train.csv")

    class InterruptedResponse:
        headers = {}

        def __enter__(self):
            self.read_count = 0
            return self

        def __exit__(self, *args):
            return None

        def read(self, _size):
            self.read_count += 1
            if self.read_count == 1:
                return b"partial"
            raise OSError("network unavailable")

    with pytest.raises(DataError, match="download failed.*network unavailable"):
        fetch_source(
            spec,
            tmp_path,
            opener=lambda *_args, **_kwargs: InterruptedResponse(),
        )
    assert not (tmp_path / spec.filename).exists()
    assert not list(tmp_path.glob("*.part"))


def test_offline_mode_rejects_missing_or_corrupt_cache(tmp_path):
    spec = SourceSpec("official_train", "https://example.test/train.csv", "train.csv")
    with pytest.raises(DataError, match="offline mode"):
        fetch_source(spec, tmp_path, offline=True)

