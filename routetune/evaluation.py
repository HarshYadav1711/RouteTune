"""Shared strict prediction records, metrics, and timing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Callable, TypeVar


INVALID_OUTCOME = "__invalid__"
T = TypeVar("T")


class EvaluationError(ValueError):
    """Raised when prediction records violate the evaluation contract."""


@dataclass(frozen=True)
class PredictionRecord:
    example_id: str
    input_message: str
    expected_class: str
    raw_output: str | None
    predicted_class: str | None
    output_valid: bool
    correct: bool
    latency_seconds: float | None = None


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    validity_rate: float
    invalid_prediction_count: int
    prediction_count: int
    correct_count: int
    labels: tuple[str, ...]
    confusion_columns: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: dict[str, ClassMetrics]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["labels"] = list(self.labels)
        result["confusion_columns"] = list(self.confusion_columns)
        result["confusion_matrix"] = [list(row) for row in self.confusion_matrix]
        return result


def parse_strict_label(raw_output: str, labels: tuple[str, ...]) -> str | None:
    """Accept one exact configured label plus surrounding whitespace only."""

    stripped = raw_output.strip()
    return stripped if stripped in labels else None


def prediction_record(
    *,
    example_id: str,
    input_message: str,
    expected_class: str,
    raw_output: str,
    labels: tuple[str, ...],
    latency_seconds: float | None = None,
) -> PredictionRecord:
    if expected_class not in labels:
        raise EvaluationError(f"unknown expected class: {expected_class!r}")
    predicted = parse_strict_label(raw_output, labels)
    return PredictionRecord(
        example_id=example_id,
        input_message=input_message,
        expected_class=expected_class,
        raw_output=raw_output,
        predicted_class=predicted,
        output_valid=predicted is not None,
        correct=predicted == expected_class,
        latency_seconds=latency_seconds,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_predictions(
    predictions: tuple[PredictionRecord, ...], labels: tuple[str, ...]
) -> EvaluationReport:
    if not predictions:
        raise EvaluationError("at least one prediction is required")
    if len(labels) != len(set(labels)) or not labels:
        raise EvaluationError("labels must be non-empty and unique")
    identifiers = [prediction.example_id for prediction in predictions]
    if len(identifiers) != len(set(identifiers)):
        raise EvaluationError("prediction example identifiers must be unique")

    label_indexes = {label: index for index, label in enumerate(labels)}
    confusion = [[0] * (len(labels) + 1) for _ in labels]
    valid_count = 0
    correct_count = 0
    for prediction in predictions:
        if prediction.expected_class not in label_indexes:
            raise EvaluationError(
                f"unknown expected class: {prediction.expected_class!r}"
            )
        if prediction.output_valid != (prediction.predicted_class in label_indexes):
            raise EvaluationError("prediction validity fields are inconsistent")
        if not prediction.output_valid and prediction.predicted_class is not None:
            raise EvaluationError("invalid predictions must have a null parsed class")
        expected_correct = (
            prediction.output_valid
            and prediction.predicted_class == prediction.expected_class
        )
        if prediction.correct != expected_correct:
            raise EvaluationError("prediction correctness field is inconsistent")
        if prediction.latency_seconds is not None and (
            not isfinite(prediction.latency_seconds)
            or prediction.latency_seconds < 0
        ):
            raise EvaluationError("prediction latency must be finite and non-negative")
        row = label_indexes[prediction.expected_class]
        column = (
            label_indexes[prediction.predicted_class]
            if prediction.output_valid and prediction.predicted_class is not None
            else len(labels)
        )
        confusion[row][column] += 1
        valid_count += int(prediction.output_valid)
        correct_count += int(prediction.correct)

    per_class: dict[str, ClassMetrics] = {}
    for index, label in enumerate(labels):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_total = sum(row[index] for row in confusion)
        precision = _ratio(true_positive, predicted_total)
        recall = _ratio(true_positive, support)
        f1 = _ratio(2 * precision * recall, precision + recall)
        per_class[label] = ClassMetrics(precision, recall, f1, support)

    count = len(predictions)
    if sum(sum(row) for row in confusion) != count:
        raise EvaluationError("confusion matrix denominator mismatch")
    if sum(metric.support for metric in per_class.values()) != count:
        raise EvaluationError("per-class support denominator mismatch")

    return EvaluationReport(
        accuracy=correct_count / count,
        macro_precision=sum(metric.precision for metric in per_class.values())
        / len(labels),
        macro_recall=sum(metric.recall for metric in per_class.values()) / len(labels),
        macro_f1=sum(metric.f1 for metric in per_class.values()) / len(labels),
        validity_rate=valid_count / count,
        invalid_prediction_count=count - valid_count,
        prediction_count=count,
        correct_count=correct_count,
        labels=labels,
        confusion_columns=labels + (INVALID_OUTCOME,),
        confusion_matrix=tuple(tuple(row) for row in confusion),
        per_class=per_class,
    )


def timed_call(
    operation: Callable[[], T],
    *,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> tuple[T, float]:
    """Time one operation with optional accelerator synchronization."""

    if synchronize is not None:
        synchronize()
    started = clock()
    result = operation()
    if synchronize is not None:
        synchronize()
    elapsed = clock() - started
    return result, elapsed
