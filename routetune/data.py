"""Validated, deterministic BANKING77 ingestion and split construction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import DEFAULT_CONFIG, ExperimentConfig


TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)
REQUIRED_COLUMNS = frozenset({"text", "category"})


class DataError(ValueError):
    """Raised when data cannot satisfy the experiment contract."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    filename: str


@dataclass(frozen=True)
class Record:
    text: str
    category: str
    source: str


@dataclass(frozen=True)
class Provenance:
    source: str
    url: str
    sha256: str
    size_bytes: int
    cache_status: str
    fetched_at_utc: str | None = None
    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class LoadedSource:
    records: tuple[Record, ...]
    provenance: Provenance


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[Record, ...]
    validation: tuple[Record, ...]
    test: tuple[Record, ...]


OFFICIAL_SOURCES = (
    SourceSpec("official_train", TRAIN_URL, "train.csv"),
    SourceSpec("official_test", TEST_URL, "test.csv"),
)


def normalize_text(text: str) -> str:
    """Normalize text only for duplicate detection, not model input."""

    return " ".join(text.casefold().split())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, source: str) -> tuple[Record, ...]:
    """Read and validate every row before filtering target categories."""

    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DataError(f"cannot read {source} CSV {path}: {exc}") from exc

    try:
        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DataError(f"{source} CSV has no header")
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
            if missing:
                raise DataError(
                    f"{source} CSV is missing required columns: "
                    f"{', '.join(sorted(missing))}"
                )

            records: list[Record] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise DataError(f"{source} CSV row {line_number} is malformed")
                text = row.get("text")
                category = row.get("category")
                if (
                    text is None
                    or category is None
                    or not text.strip()
                    or not category.strip()
                ):
                    raise DataError(
                        f"{source} CSV row {line_number} has an empty required value"
                    )
                records.append(Record(text.strip(), category.strip(), source))
    except (UnicodeError, csv.Error) as exc:
        raise DataError(f"{source} CSV is malformed: {exc}") from exc

    return tuple(records)


def _metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".metadata.json")


def _load_valid_cache(spec: SourceSpec, cache_dir: Path) -> LoadedSource | None:
    csv_path = cache_dir / spec.filename
    metadata_path = _metadata_path(csv_path)
    if not csv_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        size = csv_path.stat().st_size
        digest = file_sha256(csv_path)
        if (
            metadata["source"] != spec.name
            or metadata["url"] != spec.url
            or metadata["sha256"] != digest
            or metadata["size_bytes"] != size
        ):
            return None
        records = read_csv(csv_path, spec.name)
    except (DataError, KeyError, OSError, TypeError, json.JSONDecodeError):
        return None

    return LoadedSource(
        records,
        Provenance(
            source=spec.name,
            url=spec.url,
            sha256=digest,
            size_bytes=size,
            cache_status="validated_cache",
            fetched_at_utc=metadata.get("fetched_at_utc"),
            etag=metadata.get("etag"),
            last_modified=metadata.get("last_modified"),
        ),
    )


def _download(
    spec: SourceSpec,
    destination: Path,
    timeout_seconds: float,
    opener: Callable[..., BinaryIO],
) -> dict[str, str | None]:
    request = Request(spec.url, headers={"User-Agent": "RouteTune/0.1"})
    try:
        with opener(request, timeout=timeout_seconds) as response:
            headers = response.headers
            with destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
    except (OSError, URLError) as exc:
        raise DataError(f"download failed for {spec.name} ({spec.url}): {exc}") from exc
    return {
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
    }


def fetch_source(
    spec: SourceSpec,
    cache_dir: Path,
    *,
    offline: bool = False,
    timeout_seconds: float = 30.0,
    opener: Callable[..., BinaryIO] = urlopen,
) -> LoadedSource:
    """Return a validated cache entry, downloading atomically when needed."""

    cached = _load_valid_cache(spec, cache_dir)
    if cached is not None:
        return cached
    if offline:
        raise DataError(
            f"offline mode: no validated cache is available for {spec.name}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / spec.filename
    temporary_path: Path | None = None
    metadata_temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_dir, prefix=f".{spec.filename}.", suffix=".part", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        response_metadata = _download(spec, temporary_path, timeout_seconds, opener)
        records = read_csv(temporary_path, spec.name)
        size = temporary_path.stat().st_size
        digest = file_sha256(temporary_path)
        fetched_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "source": spec.name,
            "url": spec.url,
            "sha256": digest,
            "size_bytes": size,
            "fetched_at_utc": fetched_at,
            **response_metadata,
        }
        os.replace(temporary_path, csv_path)
        temporary_path = None
        metadata_path = _metadata_path(csv_path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_dir,
            prefix=f".{metadata_path.name}.",
            suffix=".part",
            delete=False,
        ) as temporary_metadata:
            json.dump(metadata, temporary_metadata, indent=2, sort_keys=True)
            temporary_metadata.write("\n")
            metadata_temporary_path = Path(temporary_metadata.name)
        os.replace(metadata_temporary_path, metadata_path)
        metadata_temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if metadata_temporary_path is not None:
            metadata_temporary_path.unlink(missing_ok=True)

    return LoadedSource(
        records,
        Provenance(
            source=spec.name,
            url=spec.url,
            sha256=digest,
            size_bytes=size,
            cache_status="downloaded_and_validated",
            fetched_at_utc=fetched_at,
            etag=response_metadata["etag"],
            last_modified=response_metadata["last_modified"],
        ),
    )


def _derived_seed(seed: int, namespace: str) -> int:
    material = f"{seed}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _usable_by_label(
    records: Iterable[Record], labels: tuple[str, ...]
) -> dict[str, list[Record]]:
    selected_labels = set(labels)
    by_label: dict[str, list[Record]] = defaultdict(list)
    seen: dict[str, str] = {}
    for record in records:
        if record.category not in selected_labels:
            continue
        normalized = normalize_text(record.text)
        previous_label = seen.get(normalized)
        if previous_label is not None:
            if previous_label != record.category:
                raise DataError(
                    "the same normalized text has conflicting target labels: "
                    f"{previous_label!r} and {record.category!r}"
                )
            continue
        seen[normalized] = record.category
        by_label[record.category].append(record)
    return by_label


def _sample(
    candidates: list[Record], count: int, seed: int, namespace: str
) -> tuple[Record, ...]:
    derived_seed = _derived_seed(seed, namespace)

    def rank(item: tuple[int, Record]) -> bytes:
        index, record = item
        value = f"{derived_seed}:{index}:{normalize_text(record.text)}".encode("utf-8")
        return hashlib.sha256(value).digest()

    ranked = sorted(enumerate(candidates), key=rank)
    return tuple(record for _, record in ranked[:count])


def validate_splits(splits: DatasetSplits, config: ExperimentConfig) -> None:
    expected = {
        "train": config.train_per_class,
        "validation": config.validation_per_class,
        "test": config.test_per_class,
    }
    owners: dict[str, str] = {}
    for split_name, records in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        counts = Counter(record.category for record in records)
        wanted = {label: expected[split_name] for label in config.labels}
        if counts != wanted:
            raise DataError(
                f"{split_name} is not balanced as required: {dict(sorted(counts.items()))}"
            )
        for record in records:
            normalized = normalize_text(record.text)
            previous_split = owners.get(normalized)
            if previous_split is not None:
                location = "within" if previous_split == split_name else "across"
                raise DataError(
                    f"duplicate normalized text detected {location} selected splits: "
                    f"{previous_split} and {split_name}"
                )
            owners[normalized] = split_name


def build_splits(
    official_train: LoadedSource,
    official_test: LoadedSource,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> DatasetSplits:
    """Create balanced splits without crossing official source boundaries."""

    if official_train.provenance.source != "official_train":
        raise DataError("training pool must come from official_train")
    if official_test.provenance.source != "official_test":
        raise DataError("held-out pool must come from official_test")
    if any(record.source != "official_train" for record in official_train.records):
        raise DataError("training pool contains a record outside official_train")
    if any(record.source != "official_test" for record in official_test.records):
        raise DataError("held-out pool contains a record outside official_test")

    train_by_label = _usable_by_label(official_train.records, config.labels)
    test_by_label = _usable_by_label(official_test.records, config.labels)
    development_count = config.train_per_class + config.validation_per_class
    train: list[Record] = []
    validation: list[Record] = []
    test: list[Record] = []

    for label in config.labels:
        development_candidates = train_by_label[label]
        held_out_candidates = test_by_label[label]
        if len(development_candidates) < development_count:
            raise DataError(
                f"official_train has {len(development_candidates)} unique {label!r} rows; "
                f"{development_count} required"
            )
        if len(held_out_candidates) < config.test_per_class:
            raise DataError(
                f"official_test has {len(held_out_candidates)} unique {label!r} rows; "
                f"{config.test_per_class} required"
            )
        development = _sample(
            development_candidates, development_count, config.seed, f"train:{label}"
        )
        train.extend(development[: config.train_per_class])
        validation.extend(development[config.train_per_class :])
        test.extend(
            _sample(
                held_out_candidates,
                config.test_per_class,
                config.seed,
                f"test:{label}",
            )
        )

    splits = DatasetSplits(tuple(train), tuple(validation), tuple(test))
    validate_splits(splits, config)
    return splits


def selection_fingerprint(splits: DatasetSplits) -> str:
    """Fingerprint the ordered records from all three fixed splits."""

    payload = json.dumps(
        [
            [record.text, record.category, record.source]
            for split in (splits.train, splits.validation, splits.test)
            for record in split
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _counts(records: Iterable[Record], labels: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(record.category for record in records)
    return {label: counts[label] for label in labels}


def inspect_official_data(cache_dir: Path, offline: bool = False) -> dict[str, object]:
    sources = {
        spec.name: fetch_source(spec, cache_dir, offline=offline)
        for spec in OFFICIAL_SOURCES
    }
    splits = build_splits(sources["official_train"], sources["official_test"])
    return {
        "official_source_status": {
            name: source.provenance.cache_status for name, source in sources.items()
        },
        "available_target_rows": {
            name: _counts(source.records, DEFAULT_CONFIG.labels)
            for name, source in sources.items()
        },
        "selected_counts": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "per_class_distribution": {
            "train": _counts(splits.train, DEFAULT_CONFIG.labels),
            "validation": _counts(splits.validation, DEFAULT_CONFIG.labels),
            "test": _counts(splits.test, DEFAULT_CONFIG.labels),
        },
        "duplicate_check": {
            "within_selected_splits": 0,
            "cross_split_leakage": 0,
            "status": "passed",
        },
        "fingerprints": {
            name: asdict(source.provenance) for name, source in sources.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the official BANKING77 data")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--offline", action="store_true", help="require an already validated local cache"
    )
    args = parser.parse_args(argv)
    try:
        report = inspect_official_data(args.cache_dir, args.offline)
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

