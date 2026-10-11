import inspect
import json

import pytest

from routetune.baseline import (
    BaselineError,
    fit_baseline,
    load_result,
    predict_validation,
    run_validation_baseline,
)
from routetune.data import Record


LABELS = ("alpha", "beta")
TRAIN = (
    Record("alpha apple account", "alpha", "official_train"),
    Record("alpha apricot balance", "alpha", "official_train"),
    Record("beta banana card", "beta", "official_train"),
    Record("beta berry payment", "beta", "official_train"),
)
VALIDATION = (
    Record("alpha validationonlytoken", "alpha", "official_train"),
    Record("beta purchase", "beta", "official_train"),
)


def test_fit_uses_training_only_not_validation():
    pipeline, _ = fit_baseline(TRAIN, seed=42)
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    assert "validationonlytoken" not in vocabulary


def test_phase_two_baseline_interface_has_no_test_input():
    parameters = inspect.signature(run_validation_baseline).parameters
    assert "training_records" in parameters
    assert "validation_records" in parameters
    assert not any("test" in name for name in parameters)


def test_tiny_fixture_predictions_are_valid_and_reproducible():
    first, _ = fit_baseline(TRAIN, seed=42)
    second, _ = fit_baseline(TRAIN, seed=42)
    first_predictions = predict_validation(first, VALIDATION, LABELS)
    second_predictions = predict_validation(second, VALIDATION, LABELS)

    assert [row.predicted_class for row in first_predictions] == [
        row.predicted_class for row in second_predictions
    ]
    assert all(row.output_valid for row in first_predictions)
    assert {row.predicted_class for row in first_predictions}.issubset(LABELS)


def test_validation_run_writes_complete_consumable_artifact(tmp_path):
    artifact = run_validation_baseline(
        training_records=TRAIN,
        validation_records=VALIDATION,
        labels=LABELS,
        seed=42,
        dataset_fingerprint="fixture-fingerprint",
        output_dir=tmp_path / "run",
    )
    result = load_result(artifact)

    assert result["evaluation_split"] == "validation"
    assert result["official_test_evaluated"] is False
    assert result["training_sample_count"] == len(TRAIN)
    assert result["validation_sample_count"] == len(VALIDATION)
    assert result["metrics"]["prediction_count"] == len(VALIDATION)
    assert len(result["predictions"]) == len(VALIDATION)
    assert json.loads(artifact.read_text(encoding="utf-8")) == result


def test_baseline_rejects_test_source_records(tmp_path):
    test_record = Record("held out", "alpha", "official_test")
    with pytest.raises(BaselineError, match="outside official_train"):
        run_validation_baseline(
            training_records=TRAIN + (test_record,),
            validation_records=VALIDATION,
            labels=LABELS,
            seed=42,
            dataset_fingerprint="fixture",
            output_dir=tmp_path / "run",
        )


def test_baseline_refuses_to_overwrite_artifacts(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(BaselineError, match="already exists"):
        run_validation_baseline(
            training_records=TRAIN,
            validation_records=VALIDATION,
            labels=LABELS,
            seed=42,
            dataset_fingerprint="fixture",
            output_dir=output,
        )


def test_saved_metrics_are_verified_against_complete_predictions(tmp_path):
    artifact = run_validation_baseline(
        training_records=TRAIN,
        validation_records=VALIDATION,
        labels=LABELS,
        seed=42,
        dataset_fingerprint="fixture",
        output_dir=tmp_path / "run",
    )
    result = json.loads(artifact.read_text(encoding="utf-8"))
    result["metrics"]["accuracy"] = 1.0 - result["metrics"]["accuracy"]
    artifact.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(BaselineError, match="metrics do not match"):
        load_result(artifact)

