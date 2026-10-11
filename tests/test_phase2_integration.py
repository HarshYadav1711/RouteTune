from pathlib import Path

import pytest

from routetune.challenge import CHALLENGE_CASES, assert_no_normalized_overlap
from routetune.config import EXPECTED_SELECTION_FINGERPRINT
from routetune.data import DataError, OFFICIAL_SOURCES, build_splits, fetch_source, selection_fingerprint


@pytest.mark.integration
def test_locked_official_selection_and_challenge_non_overlap():
    cache_dir = Path(__file__).parents[1] / "data" / "cache"
    try:
        sources = {
            spec.name: fetch_source(spec, cache_dir, offline=True)
            for spec in OFFICIAL_SOURCES
        }
    except DataError as exc:
        pytest.skip(f"validated BANKING77 cache unavailable: {exc}")
    splits = build_splits(sources["official_train"], sources["official_test"])

    assert selection_fingerprint(splits) == EXPECTED_SELECTION_FINGERPRINT
    assert_no_normalized_overlap(
        splits.train + splits.validation + splits.test, CHALLENGE_CASES
    )

