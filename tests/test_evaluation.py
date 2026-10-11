import pytest

from routetune.evaluation import (
    EvaluationError,
    PredictionRecord,
    evaluate_predictions,
    parse_strict_label,
    prediction_record,
    timed_call,
)


LABELS = ("alpha", "beta")


@pytest.mark.parametrize("raw", ["alpha", " alpha ", "alpha\n", "\talpha"])
def test_strict_parser_accepts_exact_label_with_surrounding_whitespace(raw):
    assert parse_strict_label(raw, LABELS) == "alpha"


@pytest.mark.parametrize(
    "raw",
    [
        "Alpha",
        "unknown",
        "alpha, beta",
        "alpha because it fits",
        "alpha.",
        "Label: alpha",
        "",
        "   ",
    ],
)
def test_strict_parser_rejects_non_exact_output(raw):
    assert parse_strict_label(raw, LABELS) is None


def record(identifier, expected, raw):
    return prediction_record(
        example_id=identifier,
        input_message=f"message {identifier}",
        expected_class=expected,
        raw_output=raw,
        labels=LABELS,
    )


def test_metrics_include_invalid_predictions_and_confusion_totals():
    predictions = (
        record("1", "alpha", "alpha"),
        record("2", "alpha", "not a label"),
        record("3", "beta", "alpha"),
        record("4", "beta", "beta"),
    )
    report = evaluate_predictions(predictions, LABELS)

    assert report.prediction_count == 4
    assert report.correct_count == 2
    assert report.accuracy == 0.5
    assert report.validity_rate == 0.75
    assert report.invalid_prediction_count == 1
    assert report.macro_precision == pytest.approx(0.75)
    assert report.macro_recall == pytest.approx(0.5)
    assert report.macro_f1 == pytest.approx((0.5 + 2 / 3) / 2)
    assert report.per_class["alpha"].support == 2
    assert report.per_class["alpha"].precision == 0.5
    assert report.per_class["alpha"].recall == 0.5
    assert report.per_class["beta"].precision == 1.0
    assert report.confusion_columns == ("alpha", "beta", "__invalid__")
    assert report.confusion_matrix == ((1, 0, 1), (1, 1, 0))
    assert sum(map(sum, report.confusion_matrix)) == report.prediction_count


def test_zero_denominators_are_zero_not_errors():
    report = evaluate_predictions((record("1", "alpha", "alpha"),), ("alpha", "beta"))
    assert report.per_class["beta"].precision == 0.0
    assert report.per_class["beta"].recall == 0.0
    assert report.per_class["beta"].f1 == 0.0
    assert report.per_class["beta"].support == 0


def test_complete_prediction_records_are_preserved_in_report_denominator():
    predictions = tuple(record(str(index), "alpha", "") for index in range(3))
    report = evaluate_predictions(predictions, LABELS)
    assert report.prediction_count == len(predictions)
    assert report.invalid_prediction_count == len(predictions)
    assert report.per_class["alpha"].support == len(predictions)


def test_inconsistent_prediction_record_is_rejected():
    invalid = PredictionRecord("1", "text", "alpha", "alpha", "alpha", False, True)
    with pytest.raises(EvaluationError, match="validity fields"):
        evaluate_predictions((invalid,), LABELS)


def test_invalid_prediction_requires_null_parsed_class():
    invalid = PredictionRecord(
        "1", "text", "alpha", "unknown", "unknown", False, False
    )
    with pytest.raises(EvaluationError, match="null parsed class"):
        evaluate_predictions((invalid,), LABELS)


def test_timed_call_synchronizes_before_and_after_operation():
    events = []
    times = iter((10.0, 12.5))

    value, elapsed = timed_call(
        lambda: events.append("operation") or 7,
        synchronize=lambda: events.append("sync"),
        clock=lambda: next(times),
    )

    assert value == 7
    assert elapsed == 2.5
    assert events == ["sync", "operation", "sync"]

