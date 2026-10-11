"""TF-IDF and logistic-regression validation baseline."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import Record
from .evaluation import (
    EvaluationError,
    PredictionRecord,
    evaluate_predictions,
    prediction_record,
    timed_call,
)


class BaselineError(ValueError):
    """Raised when the validation baseline cannot run safely."""


def build_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word", lowercase=True, ngram_range=(1, 2)
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    random_state=seed,
                    max_iter=1_000,
                ),
            ),
        ]
    )


def fit_baseline(
    training_records: tuple[Record, ...], seed: int
) -> tuple[Pipeline, float]:
    if not training_records:
        raise BaselineError("training records cannot be empty")
    pipeline = build_pipeline(seed)
    fitted, elapsed = timed_call(
        lambda: pipeline.fit(
            [record.text for record in training_records],
            [record.category for record in training_records],
        )
    )
    return fitted, elapsed


def predict_validation(
    pipeline: Pipeline,
    validation_records: tuple[Record, ...],
    labels: tuple[str, ...],
) -> tuple[PredictionRecord, ...]:
    return predict_records(pipeline, validation_records, labels, "validation")


def predict_records(
    pipeline: Pipeline,
    records: tuple[Record, ...],
    labels: tuple[str, ...],
    split_name: str,
) -> tuple[PredictionRecord, ...]:
    if not split_name or any(character.isspace() for character in split_name):
        raise BaselineError("split name must be a non-empty token")
    predictions: list[PredictionRecord] = []
    for index, record in enumerate(records):
        output, elapsed = timed_call(lambda: str(pipeline.predict([record.text])[0]))
        predictions.append(
            prediction_record(
                example_id=f"{split_name}-{index:03d}",
                input_message=record.text,
                expected_class=record.category,
                raw_output=output,
                labels=labels,
                latency_seconds=elapsed,
            )
        )
    return tuple(predictions)


def run_validation_baseline(
    *,
    training_records: tuple[Record, ...],
    validation_records: tuple[Record, ...],
    labels: tuple[str, ...],
    seed: int,
    dataset_fingerprint: str,
    output_dir: Path,
) -> Path:
    """Fit on training only, evaluate validation only, and save one JSON artifact."""

    if output_dir.exists():
        raise BaselineError(f"output directory already exists: {output_dir}")
    if not validation_records:
        raise BaselineError("validation records cannot be empty")
    if any(record.source != "official_train" for record in training_records):
        raise BaselineError("training received records outside official_train")
    if any(record.source != "official_train" for record in validation_records):
        raise BaselineError("validation received records outside official_train")
    if any(record.category not in labels for record in training_records):
        raise BaselineError("training contains a category outside the label vocabulary")
    if any(record.category not in labels for record in validation_records):
        raise BaselineError("validation contains a category outside the label vocabulary")

    pipeline, fit_seconds = fit_baseline(training_records, seed)
    predictions = predict_validation(pipeline, validation_records, labels)
    report = evaluate_predictions(predictions, labels)
    if report.validity_rate != 1.0:
        raise BaselineError("classical baseline produced an out-of-vocabulary label")

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "result.json"
    result: dict[str, Any] = {
        "schema_version": "1",
        "evaluation_split": "validation",
        "official_test_evaluated": False,
        "algorithm": "TfidfVectorizer + LogisticRegression",
        "parameters": {
            "tfidf_ngram_range": [1, 2],
            "tfidf_analyzer": "word",
            "tfidf_lowercase": True,
            "logistic_regression_c": 1.0,
            "logistic_regression_max_iter": 1_000,
            "logistic_regression_random_state": seed,
            "logistic_regression_solver": "lbfgs",
        },
        "training_sample_count": len(training_records),
        "validation_sample_count": len(validation_records),
        "random_seed": seed,
        "dataset_selection_fingerprint": dataset_fingerprint,
        "scikit_learn_version": sklearn.__version__,
        "fit_seconds": fit_seconds,
        "prediction_timing": (
            "Each latency_seconds measures CPU Pipeline.predict([text]), including "
            "TF-IDF transform and logistic-regression prediction; model fitting is separate."
        ),
        "metrics": report.to_dict(),
        "predictions": [asdict(prediction) for prediction in predictions],
        "artifact_location": str(artifact_path.resolve()),
    }
    _atomic_json_write(artifact_path, result)
    return artifact_path


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            json.dump(value, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value["schema_version"] != "1"
            or value["evaluation_split"] != "validation"
            or value["official_test_evaluated"] is not False
            or not isinstance(value["metrics"], dict)
            or not isinstance(value["predictions"], list)
        ):
            raise BaselineError(f"invalid baseline artifact: {path}")
        predictions = tuple(
            PredictionRecord(**prediction) for prediction in value["predictions"]
        )
        labels = tuple(value["metrics"]["labels"])
        if len(predictions) != value["validation_sample_count"]:
            raise BaselineError(f"incomplete baseline artifact: {path}")
        if evaluate_predictions(predictions, labels).to_dict() != value["metrics"]:
            raise BaselineError(f"baseline metrics do not match predictions: {path}")
    except BaselineError:
        raise
    except (
        EvaluationError,
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise BaselineError(f"cannot load baseline artifact {path}: {exc}") from exc
    return value


def prediction_errors(result: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if limit < 1:
        raise BaselineError("error limit must be positive")
    return [row for row in result["predictions"] if not row["correct"]][:limit]

