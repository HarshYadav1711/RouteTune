import pytest

from routetune.challenge import (
    CHALLENGE_CASES,
    DIAGNOSTIC_CATEGORIES,
    EXPECTED_CHALLENGE_FINGERPRINT,
    ChallengeCase,
    assert_no_normalized_overlap,
    challenge_fingerprint,
    validate_challenge_cases,
)
from routetune.config import TARGET_LABELS
from routetune.data import Record


def test_frozen_challenge_suite_is_valid_and_covers_every_label():
    validate_challenge_cases()
    assert 16 <= len(CHALLENGE_CASES) <= 24
    assert len({case.case_id for case in CHALLENGE_CASES}) == len(CHALLENGE_CASES)
    assert {case.expected_class for case in CHALLENGE_CASES}.issuperset(TARGET_LABELS)
    assert all(
        case.diagnostic_category in DIAGNOSTIC_CATEGORIES for case in CHALLENGE_CASES
    )
    assert all(
        case.expected_class is None or case.expected_class in TARGET_LABELS
        for case in CHALLENGE_CASES
    )


def test_challenge_serialization_fingerprint_is_frozen_and_deterministic():
    assert challenge_fingerprint() == challenge_fingerprint(tuple(CHALLENGE_CASES))
    assert challenge_fingerprint() == EXPECTED_CHALLENGE_FINGERPRINT


def test_ambiguous_and_out_of_domain_cases_have_no_forced_label():
    assert all(
        case.expected_class is None
        for case in CHALLENGE_CASES
        if case.diagnostic_category in {"ambiguous", "boundary", "out_of_domain"}
    )


def test_overlap_detection_uses_phase_one_normalization():
    case = ChallengeCase("overlap", "  SAME   Text ", "card_arrival", "clear_intent")
    records = (Record("same text", "card_arrival", "official_train"),)
    with pytest.raises(ValueError, match="overlap"):
        assert_no_normalized_overlap(records, (case,))

