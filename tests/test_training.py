from dataclasses import replace
import json

import pytest

from routetune.config import TARGET_LABELS
from routetune.data import Record
from routetune.evaluation import prediction_record
from routetune.training import (
    FIXED_QLORA_CONFIG,
    TrainingError,
    adapter_parameter_evidence,
    assert_identical_order,
    pad_tokenized_batch,
    prompt_completion,
    tokenize_records,
    tokenize_training_record,
    validate_qlora_config,
    verify_gpu_environment,
    verify_evaluation_artifact,
    verify_masked_batch,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        prompt = [10, 11, 12]
        if add_generation_prompt:
            assert [message["role"] for message in messages] == ["system", "user"]
            return prompt
        label = TARGET_LABELS.index(messages[-1]["content"]) + 20
        return prompt + [label, 2]

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return TARGET_LABELS[tokens[0] - 20]


RECORD = Record("Where is my card?", "card_arrival", "official_train")


def test_prompt_completion_reuses_inference_prompt_and_exact_label():
    example = prompt_completion(RECORD)
    assert example["prompt"][1] == {"role": "user", "content": RECORD.text}
    assert example["completion"] == [{"role": "assistant", "content": RECORD.category}]
    assert RECORD.category not in example["prompt"][1]["content"]


def test_tokenization_masks_prompt_and_preserves_completion():
    tokenized = tokenize_training_record(RECORD, FakeTokenizer())
    assert tokenized == {
        "input_ids": [10, 11, 12, 20, 2],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, -100, 20, 2],
    }


def test_truncation_never_silently_removes_target():
    with pytest.raises(TrainingError, match="refusing to truncate"):
        tokenize_training_record(RECORD, FakeTokenizer(), max_length=4)


def test_every_class_is_required_for_mask_verification():
    records = tuple(Record(label, label, "official_train") for label in TARGET_LABELS)
    assert len(tokenize_records(records, FakeTokenizer())) == len(TARGET_LABELS)
    with pytest.raises(TrainingError, match="at least one example per class"):
        tokenize_records(records[:-1], FakeTokenizer())


def test_padding_is_excluded_from_loss_and_invalid_mask_fails():
    first = {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}
    second = {"input_ids": [3], "attention_mask": [1], "labels": [3]}
    batch = pad_tokenized_batch([first, second], 0)
    assert batch["labels"][1] == [3, -100]
    assert batch["attention_mask"][1] == [1, 0]

    batch["labels"][1][1] = 0
    with pytest.raises(TrainingError, match="padding token"):
        verify_masked_batch(batch)


def test_frozen_model_configuration_cannot_drift():
    validate_qlora_config(FIXED_QLORA_CONFIG)
    with pytest.raises(TrainingError, match="frozen configuration"):
        validate_qlora_config(replace(FIXED_QLORA_CONFIG, lora_rank=16))


class Parameter:
    def __init__(self, size, requires_grad):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self.size


class Module:
    lora_A = object()


class FakeModel:
    def named_parameters(self):
        yield "base.weight", Parameter(1000, False)
        yield "block.q_proj.lora_A.default.weight", Parameter(10, True)

    def named_modules(self):
        for target in FIXED_QLORA_CONFIG.target_modules:
            yield f"block.{target}", Module()


def test_adapter_parameter_accounting_requires_only_lora_trainables():
    evidence = adapter_parameter_evidence(FakeModel())
    assert evidence["trainable_parameters"] == 10
    assert evidence["total_parameters"] == 1010
    assert evidence["base_parameters_frozen"] is True


def test_dependency_failure_is_clear_without_touching_gpu(monkeypatch):
    monkeypatch.setattr(
        "routetune.training.verify_training_apis",
        lambda: (_ for _ in ()).throw(TrainingError("incompatible TRL")),
    )
    with pytest.raises(TrainingError, match="incompatible TRL"):
        verify_gpu_environment()


def test_evaluation_conditions_must_keep_identical_order():
    def rows(order):
        return tuple(
            prediction_record(
                example_id=identifier,
                input_message=identifier,
                expected_class="card_arrival",
                raw_output="card_arrival",
                labels=TARGET_LABELS,
            )
            for identifier in order
        )

    assert_identical_order(rows(("a", "b")), rows(("a", "b")))
    with pytest.raises(TrainingError, match="identical ordered"):
        assert_identical_order(rows(("a", "b")), rows(("b", "a")))


def test_evaluation_artifact_recomputes_metrics_and_order(tmp_path):
    row = prediction_record(
        example_id="test-000",
        input_message="message",
        expected_class="card_arrival",
        raw_output="card_arrival",
        labels=TARGET_LABELS,
    )
    from routetune.evaluation import evaluate_predictions

    path = tmp_path / "evaluation.json"
    value = {
        "complete": True,
        "metrics": evaluate_predictions((row,), TARGET_LABELS).to_dict(),
        "latency_seconds": {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "maximum": None,
        },
        "predictions": [row.__dict__],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert verify_evaluation_artifact(path, ["test-000"])["complete"] is True

    value["metrics"]["accuracy"] = 0.0
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TrainingError, match="metrics do not match"):
        verify_evaluation_artifact(path, ["test-000"])
