from collections import UserDict
from dataclasses import replace
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tomllib
from types import ModuleType, SimpleNamespace

import pytest

from routetune.config import MODEL_ID, TARGET_LABELS
from routetune.data import Record
from routetune.evaluation import evaluate_predictions, prediction_record
from routetune.experiment import create_run, load_manifest, update_run
from routetune.inference import GenerationResult
from routetune.training import (
    FIXED_QLORA_CONFIG,
    GPU_PACKAGES,
    PINNED_GPU_PACKAGE_VERSIONS,
    TrainingError,
    _evaluate_qwen,
    _load_quantized_base,
    _load_recorded_quantized_base,
    _promote_trainable_adapters_to_fp32,
    assess_pip_check,
    adapter_parameter_evidence,
    assert_identical_order,
    build_sft_config,
    execute_phase3_training,
    pad_tokenized_batch,
    parameter_dtype_histogram,
    prepare_trainer_precision,
    prompt_completion,
    resolve_model_revision,
    tokenize_records,
    tokenize_training_record,
    validate_qlora_config,
    verify_gpu_environment,
    verify_evaluation_artifact,
    verify_masked_batch,
    verify_pinned_gpu_dependencies,
    verify_torch_runtime,
    verify_training_precision,
    verify_transformers_runtime,
)


MODEL_SHA = "0123456789abcdef0123456789abcdef01234567"


def _mock_phase3_until_smoke(
    monkeypatch, tmp_path, *, smoke=None, smoke_error=None, load_error=None
):
    fingerprints = {"selection": "fixture"}
    run = create_run(
        tmp_path,
        configuration={"fixture": True},
        fingerprints=fingerprints,
        environment={},
    )
    monkeypatch.setattr(
        "routetune.training.load_locked_splits",
        lambda *_args, **_kwargs: (SimpleNamespace(validation=()), fingerprints),
    )
    monkeypatch.setattr(
        "routetune.training.verify_gpu_environment", lambda: {"cuda": True}
    )
    monkeypatch.setattr("routetune.training.build_sft_config", lambda *_args: object())
    monkeypatch.setattr("routetune.training.parameter_dtype_histogram", lambda _model: {})
    monkeypatch.setattr("routetune.training.resolve_model_revision", lambda: MODEL_SHA)

    torch = ModuleType("torch")
    torch.manual_seed = lambda _seed: None
    torch.cuda = SimpleNamespace(
        manual_seed_all=lambda _seed: None, synchronize=lambda: None
    )
    datasets = ModuleType("datasets")
    datasets.Dataset = object
    trl = ModuleType("trl")
    trl.SFTConfig = object
    trl.SFTTrainer = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "trl", trl)

    if load_error is None:
        monkeypatch.setattr(
            "routetune.training._load_quantized_base",
            lambda _revision: (object(), SimpleNamespace(), MODEL_SHA),
        )
    else:
        monkeypatch.setattr(
            "routetune.training._load_quantized_base",
            lambda _revision: (_ for _ in ()).throw(load_error),
        )

    class SmokeAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def classify(self, _message):
            if smoke_error is not None:
                raise smoke_error
            return smoke

    monkeypatch.setattr("routetune.training.QwenAdapter", SmokeAdapter)
    monkeypatch.setattr(
        "routetune.training._evaluate_qwen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TrainingError("reached validation")),
    )
    return run


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, wrap=lambda token_ids: token_ids):
        self.wrap = wrap

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict
    ):
        assert tokenize is True
        assert return_dict is True
        prompt = [10, 11, 12]
        if add_generation_prompt:
            assert [message["role"] for message in messages] == ["system", "user"]
            return self.wrap(prompt)
        label = TARGET_LABELS.index(messages[-1]["content"]) + 20
        return self.wrap(prompt + [label, 2])

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return TARGET_LABELS[tokens[0] - 20]


RECORD = Record("Where is my card?", "card_arrival", "official_train")


class TensorLike:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class BatchEncodingLike(UserDict):
    pass


def test_prompt_completion_reuses_inference_prompt_and_exact_label():
    example = prompt_completion(RECORD)
    assert example["prompt"][1] == {"role": "user", "content": RECORD.text}
    assert example["completion"] == [{"role": "assistant", "content": RECORD.category}]
    assert RECORD.category not in example["prompt"][1]["content"]


@pytest.mark.parametrize(
    "wrap",
    [
        lambda token_ids: token_ids,
        lambda token_ids: [token_ids],
        lambda token_ids: TensorLike([token_ids]),
        lambda token_ids: {"input_ids": token_ids},
        lambda token_ids: BatchEncodingLike({"input_ids": TensorLike([token_ids])}),
    ],
    ids=["flat-list", "one-example-list", "tensor", "dictionary", "batch-encoding"],
)
def test_tokenization_accepts_supported_outputs_and_preserves_loss_masks(wrap):
    tokenized = tokenize_training_record(RECORD, FakeTokenizer(wrap))
    assert tokenized == {
        "input_ids": [10, 11, 12, 20, 2],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, -100, 20, 2],
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "missing input_ids"),
        ({"input_ids": "tokens"}, "one-example token sequence"),
        ({"input_ids": []}, "sequence is empty"),
        ({"input_ids": [[10], [11]]}, "unexpected batch of 2"),
        ({"input_ids": [[[10, 11]]]}, "only integers"),
        ({"input_ids": [10, "11"]}, "only integers"),
        ({"input_ids": [10, -1]}, "non-negative integers"),
    ],
)
def test_tokenization_rejects_missing_or_malformed_input_ids(value, message):
    with pytest.raises(TrainingError, match=message):
        tokenize_training_record(RECORD, FakeTokenizer(lambda _token_ids: value))


def test_tokenization_still_rejects_prompt_prefix_mismatch():
    def mismatch(token_ids):
        return token_ids if len(token_ids) == 3 else [99, *token_ids[1:]]

    with pytest.raises(TrainingError, match="preserve the inference prompt prefix"):
        tokenize_training_record(RECORD, FakeTokenizer(mismatch))


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


class RecordingSFTConfig:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, [] if name == "report_to" and value == "none" else value)

    def get_warmup_steps(self, total_steps):
        return math.ceil(total_steps * self.warmup_steps)


def test_sft_config_uses_supported_warmup_fraction_and_frozen_settings(tmp_path):
    config = build_sft_config(RecordingSFTConfig, tmp_path / "trainer")
    kwargs = RecordingSFTConfig.last_kwargs

    assert "warmup_ratio" not in kwargs
    assert kwargs["warmup_steps"] == 0.1
    assert config.get_warmup_steps(100) == 10
    assert config.fp16 is True
    assert config.bf16 is False
    assert kwargs == {
        "output_dir": str(tmp_path / "trainer"),
        "num_train_epochs": 2,
        "learning_rate": 1.5e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 0.1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "per_device_eval_batch_size": 8,
        "max_length": 256,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": True,
        "bf16": False,
        "logging_steps": 10,
        "logging_strategy": "steps",
        "eval_strategy": "epoch",
        "save_strategy": "no",
        "report_to": "none",
        "seed": 42,
        "data_seed": 42,
        "remove_unused_columns": False,
        "completion_only_loss": True,
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }
    assert kwargs["per_device_train_batch_size"] * kwargs["gradient_accumulation_steps"] == 16


def test_sft_config_rejects_unsupported_or_invalid_configuration(tmp_path):
    class UnsupportedSFTConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    with pytest.raises(TrainingError, match="does not support RouteTune arguments"):
        build_sft_config(UnsupportedSFTConfig, tmp_path)

    class InvalidSFTConfig:
        def __init__(self, **_kwargs):
            raise TypeError("invalid training configuration")

    with pytest.raises(
        TrainingError,
        match="SFTConfig preflight failed: TypeError: invalid training configuration",
    ):
        build_sft_config(InvalidSFTConfig, tmp_path)


def test_sft_config_rejects_non_fractional_warmup_interpretation(tmp_path):
    class InvalidWarmupSFTConfig(RecordingSFTConfig):
        def get_warmup_steps(self, _total_steps):
            return 0

    with pytest.raises(TrainingError, match="warmup_steps=0.1.*10% fraction"):
        build_sft_config(InvalidWarmupSFTConfig, tmp_path)


def test_pinned_real_sft_config_preflight_when_available(tmp_path):
    trl = pytest.importorskip("trl")
    pytest.importorskip("transformers")
    if any(
        importlib.metadata.version(package) != PINNED_GPU_PACKAGE_VERSIONS[package]
        for package in ("trl", "transformers")
    ):
        pytest.skip("real-library preflight requires the pinned TRL/Transformers stack")

    config = build_sft_config(trl.SFTConfig, tmp_path / "trainer")
    assert config.warmup_steps == 0.1
    assert config.get_warmup_steps(100) == 10


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


class FakeDType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"torch.{self.name}"


FLOAT16 = FakeDType("float16")
BFLOAT16 = FakeDType("bfloat16")
FLOAT32 = FakeDType("float32")


class FakeTensorData:
    def __init__(self, dtype):
        self.dtype = dtype

    def to(self, dtype):
        return FakeTensorData(dtype)


class PrecisionParameter:
    def __init__(self, size, requires_grad, dtype, *, device="cuda", index=0):
        self.size = size
        self.requires_grad = requires_grad
        self.data = FakeTensorData(dtype)
        self.device = SimpleNamespace(type=device, index=index)

    @property
    def dtype(self):
        return self.data.dtype

    def numel(self):
        return self.size


class PrecisionModel:
    is_loaded_in_4bit = True

    def __init__(self, trainable_dtype=FLOAT32, *, base_trainable=False):
        self.parameters = [
            ("base.weight", PrecisionParameter(1000, base_trainable, FLOAT16)),
            (
                "block.q_proj.lora_A.default.weight",
                PrecisionParameter(10, True, trainable_dtype),
            ),
        ]
        self.config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=FLOAT16,
            )
        )

    def named_parameters(self):
        yield from self.parameters

    def named_modules(self):
        for target in FIXED_QLORA_CONFIG.target_modules:
            yield f"block.{target}", Module()


def _fake_precision_torch(monkeypatch, *, bf16_supported=False):
    torch = ModuleType("torch")
    torch.float16 = FLOAT16
    torch.bfloat16 = BFLOAT16
    torch.float32 = FLOAT32
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_name=lambda _index: "Tesla T4",
        is_bf16_supported=lambda: bf16_supported,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_trainable_lora_parameters_are_promoted_to_fp32_only(monkeypatch):
    _fake_precision_torch(monkeypatch)
    model = PrecisionModel(BFLOAT16)

    _promote_trainable_adapters_to_fp32(model)

    assert model.parameters[0][1].dtype is FLOAT16
    assert model.parameters[0][1].requires_grad is False
    assert model.parameters[1][1].dtype is FLOAT32
    assert model.parameters[1][1].requires_grad is True


def test_precision_histogram_and_gate_record_effective_contract(monkeypatch):
    _fake_precision_torch(monkeypatch, bf16_supported=False)
    model = PrecisionModel()
    histogram = parameter_dtype_histogram(model)
    assert histogram == {
        "trainable": {"float32": {"tensors": 1, "elements": 10}},
        "frozen": {"float16": {"tensors": 1, "elements": 1000}},
    }

    evidence = verify_training_precision(
        model,
        SimpleNamespace(fp16=True, bf16=False),
        {"gradient_dtypes": {"float32": 1}},
    )
    assert evidence["cuda_device_name"] == "Tesla T4"
    assert evidence["bf16_supported"] is False
    assert evidence["training_fp16"] is True
    assert evidence["training_bf16"] is False
    assert evidence["quantization"] == {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "float16",
    }
    assert evidence["parameter_dtypes"] == histogram
    assert evidence["gradient_dtypes"] == {"float32": 1}
    assert evidence["base_parameters_frozen"] is True


def test_precision_gate_rejects_bf16_trainables_and_unfrozen_base(monkeypatch):
    _fake_precision_torch(monkeypatch)
    arguments = SimpleNamespace(fp16=True, bf16=False)
    gradient = {"gradient_dtypes": {"bfloat16": 1}}
    with pytest.raises(TrainingError, match="non-FP32 trainable adapters"):
        verify_training_precision(PrecisionModel(BFLOAT16), arguments, gradient)

    with pytest.raises(TrainingError, match="not restricted to LoRA"):
        verify_training_precision(PrecisionModel(base_trainable=True), arguments, gradient)


def test_precision_gate_rejects_bf16_training_configuration(monkeypatch):
    _fake_precision_torch(monkeypatch)
    with pytest.raises(TrainingError, match="fp16=True and bf16=False"):
        verify_training_precision(
            PrecisionModel(),
            SimpleNamespace(fp16=False, bf16=True),
            {"gradient_dtypes": {"float32": 1}},
        )


class MutatingTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = None
        for _, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(BFLOAT16)

    def create_optimizer(self):
        trainable = [parameter for _, parameter in self.model.named_parameters() if parameter.requires_grad]
        self.optimizer = SimpleNamespace(param_groups=[{"params": trainable}], state={})
        return self.optimizer

    def train(self):
        assert all(
            parameter.dtype is FLOAT32
            for _, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )


def test_trainer_precision_repairs_quantized_lora_before_optimizer(monkeypatch):
    _fake_precision_torch(monkeypatch)
    model = PrecisionModel()
    base_parameter = model.parameters[0][1]
    adapter_parameter = model.parameters[1][1]
    trainer = MutatingTrainer(model)

    assert parameter_dtype_histogram(trainer.model)["trainable"] == {
        "bfloat16": {"tensors": 1, "elements": 10}
    }
    evidence = prepare_trainer_precision(
        trainer,
        model,
        SimpleNamespace(fp16=True, bf16=False),
        {"gradient_dtypes": {"float32": 1}},
    )

    assert trainer.model is model
    assert model.parameters[0][1] is base_parameter
    assert base_parameter.dtype is FLOAT16
    assert base_parameter.requires_grad is False
    assert model.parameters[1][1] is adapter_parameter
    assert adapter_parameter.dtype is FLOAT32
    assert trainer.optimizer.param_groups[0]["params"] == [adapter_parameter]
    assert evidence["pretraining_model_dtypes"]["trainable"] == {
        "float32": {"tensors": 1, "elements": 10}
    }
    assert all(evidence["trainer_parameter_consistency"].values())
    trainer.train()


def test_trainer_precision_rejects_different_model_or_existing_optimizer(monkeypatch):
    _fake_precision_torch(monkeypatch)
    model = PrecisionModel()
    trainer = MutatingTrainer(model)
    with pytest.raises(TrainingError, match="retain the supplied model"):
        prepare_trainer_precision(
            trainer,
            PrecisionModel(),
            SimpleNamespace(fp16=True, bf16=False),
            {"gradient_dtypes": {"float32": 1}},
        )

    trainer.optimizer = SimpleNamespace()
    with pytest.raises(TrainingError, match="optimizer before the precision correction"):
        prepare_trainer_precision(
            trainer,
            model,
            SimpleNamespace(fp16=True, bf16=False),
            {"gradient_dtypes": {"float32": 1}},
        )


def test_sft_config_preflight_blocks_model_loading(monkeypatch, tmp_path):
    run = _mock_phase3_until_smoke(monkeypatch, tmp_path)
    model_loaded = False

    def fail_preflight(*_args):
        raise TrainingError("incompatible SFTConfig")

    def load_model(_revision):
        nonlocal model_loaded
        model_loaded = True

    monkeypatch.setattr("routetune.training.build_sft_config", fail_preflight)
    monkeypatch.setattr("routetune.training._load_quantized_base", load_model)

    with pytest.raises(TrainingError, match="incompatible SFTConfig"):
        execute_phase3_training(run, tmp_path / "cache")

    assert model_loaded is False
    assert load_manifest(run)["state"] == "PREPARED"


def test_precision_gate_failure_marks_running_experiment_failed(monkeypatch, tmp_path):
    smoke = GenerationResult("card_arrival", "card_arrival", True, 0.1, 0.2)
    run = _mock_phase3_until_smoke(monkeypatch, tmp_path, smoke=smoke)
    fingerprints = {"selection": "fixture"}
    monkeypatch.setattr(
        "routetune.training.load_locked_splits",
        lambda *_args, **_kwargs: (SimpleNamespace(train=(), validation=()), fingerprints),
    )
    monkeypatch.setattr(
        "routetune.training._load_quantized_base",
        lambda _revision: (SimpleNamespace(pad_token_id=0), SimpleNamespace(), MODEL_SHA),
    )
    monkeypatch.setattr("routetune.training.parameter_dtype_histogram", lambda _model: {})
    monkeypatch.setattr("routetune.training._evaluate_qwen", lambda *_args, **_kwargs: ())
    monkeypatch.setattr("routetune.training._write_evaluation", lambda *_args: None)
    monkeypatch.setattr("routetune.training.verify_evaluation_artifact", lambda *_args: {})
    monkeypatch.setattr("routetune.training.tokenize_records", lambda *_args: [{"fixture": True}])
    monkeypatch.setattr(
        "routetune.training.CompletionOnlyCollator", lambda _pad: lambda _rows: {"fixture": True}
    )
    monkeypatch.setattr(
        "routetune.training._attach_lora",
        lambda _model: (SimpleNamespace(), {"base_parameters_frozen": True}),
    )
    monkeypatch.setattr("routetune.training.verify_masked_batch", lambda _batch: None)
    monkeypatch.setattr(
        "routetune.training._gradient_smoke",
        lambda *_args: {"gradient_dtypes": {"float32": 1}},
    )
    monkeypatch.setattr(
        "routetune.training.prepare_trainer_precision",
        lambda *_args: (_ for _ in ()).throw(TrainingError("BF16 adapter detected")),
    )

    class Trainer:
        def __init__(self, model, **_kwargs):
            self.model = model
            self.optimizer = None

    class Dataset:
        @staticmethod
        def from_list(rows):
            return rows

    sys.modules["datasets"].Dataset = Dataset
    sys.modules["trl"].SFTTrainer = Trainer

    with pytest.raises(TrainingError, match="BF16 adapter detected"):
        execute_phase3_training(run, tmp_path / "cache")

    manifest = load_manifest(run)
    assert manifest["state"] == "FAILED"
    assert manifest["failure"]["message"] == "BF16 adapter detected"
    assert manifest["evidence"]["gradient_step_verified"] is True
    assert "training_complete" not in manifest["evidence"]


def test_trainer_dtype_repair_is_recorded_without_weakening_failure_state(monkeypatch, tmp_path):
    smoke = GenerationResult("card_arrival", "card_arrival", True, 0.1, 0.2)
    run = _mock_phase3_until_smoke(monkeypatch, tmp_path, smoke=smoke)
    fingerprints = {"selection": "fixture"}
    model = PrecisionModel()
    model.config.dtype = BFLOAT16
    monkeypatch.setattr("routetune.training.parameter_dtype_histogram", parameter_dtype_histogram)
    monkeypatch.setattr(
        "routetune.training.load_locked_splits",
        lambda *_args, **_kwargs: (SimpleNamespace(train=(), validation=()), fingerprints),
    )
    monkeypatch.setattr(
        "routetune.training._load_quantized_base",
        lambda _revision: (SimpleNamespace(pad_token_id=0), model, MODEL_SHA),
    )
    monkeypatch.setattr("routetune.training._evaluate_qwen", lambda *_args, **_kwargs: ())
    monkeypatch.setattr("routetune.training._write_evaluation", lambda *_args: None)
    monkeypatch.setattr("routetune.training.verify_evaluation_artifact", lambda *_args: {})
    monkeypatch.setattr("routetune.training.tokenize_records", lambda *_args: [{"fixture": True}])
    monkeypatch.setattr(
        "routetune.training.CompletionOnlyCollator", lambda _pad: lambda _rows: {"fixture": True}
    )
    monkeypatch.setattr(
        "routetune.training._attach_lora",
        lambda supplied: (supplied, {"base_parameters_frozen": True}),
    )
    monkeypatch.setattr("routetune.training.verify_masked_batch", lambda _batch: None)
    monkeypatch.setattr(
        "routetune.training._gradient_smoke",
        lambda *_args: {"gradient_dtypes": {"float32": 1}},
    )
    torch = _fake_precision_torch(monkeypatch)
    monkeypatch.setattr(
        "routetune.training.build_sft_config",
        lambda *_args: SimpleNamespace(fp16=True, bf16=False),
    )
    torch.manual_seed = lambda _seed: None
    torch.cuda.manual_seed_all = lambda _seed: None
    torch.cuda.synchronize = lambda: None

    class Trainer(MutatingTrainer):
        def __init__(self, model, **_kwargs):
            super().__init__(model)

        def get_train_dataloader(self):
            raise TrainingError("stop after precision evidence")

    class Dataset:
        @staticmethod
        def from_list(rows):
            return rows

    sys.modules["datasets"].Dataset = Dataset
    sys.modules["trl"].SFTTrainer = Trainer

    with pytest.raises(TrainingError, match="stop after precision evidence"):
        execute_phase3_training(run, tmp_path / "cache")

    manifest = load_manifest(run)
    evidence = manifest["evidence"]
    assert manifest["state"] == "FAILED"
    assert evidence["base_model_config_dtype"] == "bfloat16"
    assert evidence["adapter_model_dtypes"]["trainable"] == {
        "float32": {"tensors": 1, "elements": 10}
    }
    assert evidence["post_gradient_model_dtypes"] == evidence["adapter_model_dtypes"]
    assert evidence["trainer_initialized_model_dtypes"]["trainable"] == {
        "bfloat16": {"tensors": 1, "elements": 10}
    }
    assert evidence["pretraining_model_dtypes"]["trainable"] == {
        "float32": {"tensors": 1, "elements": 10}
    }
    assert evidence["precision_contract_verified"] is True
    assert all(evidence["trainer_parameter_consistency"].values())
    assert "training_complete" not in evidence


@pytest.mark.parametrize(
    ("raw_output", "predicted_class", "output_valid"),
    [
        ("card_arrival", "card_arrival", True),
        ('no\nThe message is classified as "pending_card_payment" because ...', None, False),
        ("", None, False),
    ],
)
def test_smoke_execution_is_recorded_independently_of_output_validity(
    monkeypatch, tmp_path, raw_output, predicted_class, output_valid
):
    smoke = GenerationResult(raw_output, predicted_class, output_valid, 0.25, 0.75)
    run = _mock_phase3_until_smoke(monkeypatch, tmp_path, smoke=smoke)

    with pytest.raises(TrainingError, match="reached validation"):
        execute_phase3_training(run, tmp_path / "cache")

    manifest = load_manifest(run)
    assert manifest["state"] == "FAILED"
    assert manifest["evidence"]["gpu_smoke_verified"] is True
    assert manifest["evidence"]["gpu_smoke_execution_succeeded"] is True
    assert manifest["evidence"]["gpu_smoke"] == {
        "raw_output": raw_output,
        "predicted_class": predicted_class,
        "output_valid": output_valid,
        "tokenization_seconds": 0.25,
        "generation_seconds": 0.75,
    }


@pytest.mark.parametrize(
    ("smoke_error", "load_error", "message"),
    [
        (RuntimeError("CUDA generation failed"), None, "CUDA generation failed"),
        (None, RuntimeError("quantized model load failed"), "quantized model load failed"),
        (None, None, "dataclass instances"),
    ],
)
def test_smoke_generation_and_model_load_failures_still_block(
    monkeypatch, tmp_path, smoke_error, load_error, message
):
    run = _mock_phase3_until_smoke(
        monkeypatch,
        tmp_path,
        smoke_error=smoke_error,
        load_error=load_error,
    )

    with pytest.raises(TrainingError, match=message):
        execute_phase3_training(run, tmp_path / "cache")

    manifest = load_manifest(run)
    assert manifest["state"] == "FAILED"
    assert "gpu_smoke_execution_succeeded" not in manifest["evidence"]


def test_untuned_validation_keeps_all_invalid_outputs_in_metrics(monkeypatch):
    records = tuple(
        Record(
            f"message {index}",
            TARGET_LABELS[index % len(TARGET_LABELS)],
            "official_train",
        )
        for index in range(80)
    )
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    class InvalidAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def classify(self, message):
            return GenerationResult(f"no\nExplanation for {message}", None, False, 0.1, 0.2)

    monkeypatch.setattr("routetune.training.QwenAdapter", InvalidAdapter)
    model = SimpleNamespace(eval=lambda: None)
    predictions = _evaluate_qwen(object(), model, records, "validation")
    report = evaluate_predictions(predictions, TARGET_LABELS)

    assert len(predictions) == 80
    assert predictions[-1].example_id == "validation-079"
    assert all(row.raw_output.startswith("no\nExplanation") for row in predictions)
    assert all(
        row.predicted_class is None and not row.output_valid and not row.correct
        for row in predictions
    )
    assert report.prediction_count == 80
    assert report.invalid_prediction_count == 80
    assert report.validity_rate == 0.0
    assert report.accuracy == 0.0


def test_dependency_failure_is_clear_without_touching_gpu(monkeypatch):
    monkeypatch.setattr(
        "routetune.training.verify_training_apis",
        lambda: (_ for _ in ()).throw(TrainingError("incompatible TRL")),
    )
    with pytest.raises(TrainingError, match="incompatible TRL"):
        verify_gpu_environment()


def test_pinned_gpu_dependency_versions_are_enforced(monkeypatch):
    installed = dict(PINNED_GPU_PACKAGE_VERSIONS)
    monkeypatch.setattr(
        "routetune.training.importlib.metadata.version", lambda package: installed[package]
    )
    verify_pinned_gpu_dependencies()

    installed["transformers"] = "5.18.0"
    with pytest.raises(TrainingError, match="transformers=5.18.0.*expected 5.19.0"):
        verify_pinned_gpu_dependencies()


def test_pinned_gpu_dependency_versions_accept_cuda_local_suffix(monkeypatch):
    monkeypatch.setattr(
        "routetune.training.importlib.metadata.version",
        lambda package: f"{PINNED_GPU_PACKAGE_VERSIONS[package]}+cu130"
        if package in {"torch", "torchvision"}
        else PINNED_GPU_PACKAGE_VERSIONS[package],
    )
    verify_pinned_gpu_dependencies()


def test_torch_and_torchvision_runtime_versions_and_cuda_match(monkeypatch):
    installed = {"torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130"}
    monkeypatch.setattr(
        "routetune.training.importlib.metadata.version", lambda package: installed[package]
    )
    torch = SimpleNamespace(__version__="2.14.1+cu130", version=SimpleNamespace(cuda="13.0"))
    torchvision = SimpleNamespace(__version__="0.29.1+cu130")
    verify_torch_runtime(torch, torchvision)

    torchvision.__version__ = "0.26.0+cu130"
    with pytest.raises(TrainingError, match="mixed torchvision runtime"):
        verify_torch_runtime(torch, torchvision)


def test_torch_runtime_requires_cuda_13_build(monkeypatch):
    installed = {"torch": "2.14.1", "torchvision": "0.29.1"}
    monkeypatch.setattr(
        "routetune.training.importlib.metadata.version", lambda package: installed[package]
    )
    torch = SimpleNamespace(__version__="2.14.1", version=SimpleNamespace(cuda="12.8"))
    torchvision = SimpleNamespace(__version__="0.29.1")
    with pytest.raises(TrainingError, match="CUDA 13.0"):
        verify_torch_runtime(torch, torchvision)


def test_pip_check_allows_but_records_unrelated_preinstalled_conflicts():
    output = "bigframes 2.0 has requirement pyarrow<20, but you have pyarrow 21.0.0."
    result = assess_pip_check(
        output,
        1,
        route_packages={"routetune", "torchvision"},
        operational_packages={"google-colab"},
    )
    assert result == {
        "ready": True,
        "pip_check_clean": False,
        "route_conflicts": [],
        "operational_conflicts": [],
        "unrelated_conflicts": [output],
        "unrelated_conflicts_reason": (
            "requesting packages are outside RouteTune and Colab/GPU dependency closures"
        ),
        "pip_check_output": output,
    }


def test_pip_check_fails_closed_on_unparseable_failure():
    output = "unexpected resolver failure"
    result = assess_pip_check(
        output,
        2,
        route_packages=set(),
        operational_packages=set(),
    )
    assert result["ready"] is False
    assert result["route_conflicts"] == [output]


@pytest.mark.parametrize(
    ("package", "category"),
    [("torchvision", "route_conflicts"), ("google-colab", "operational_conflicts"), ("gcsfs", "operational_conflicts")],
)
def test_pip_check_blocks_required_or_colab_operational_conflicts(package, category):
    output = f"{package} 1.0 has requirement dependency==1, but you have dependency 2."
    result = assess_pip_check(
        output,
        1,
        route_packages={"torchvision"},
        operational_packages={"google-colab", "gcsfs"},
    )
    assert result["ready"] is False
    assert result["pip_check_clean"] is False
    assert result[category] == [output]


def test_transformers_runtime_rejects_mixed_install(monkeypatch):
    transformers = SimpleNamespace(__version__="5.18.0")
    monkeypatch.setattr("routetune.training.importlib.metadata.version", lambda _: "5.19.0")
    with pytest.raises(TrainingError, match="mixed Transformers runtime"):
        verify_transformers_runtime(transformers)


def test_transformers_runtime_rejects_mutable_revision_patch(monkeypatch):
    class AutoConfig:
        from_pretrained = staticmethod(lambda *_args, **_kwargs: None)

    class PreTrainedConfig:
        get_config_dict = staticmethod(lambda *_args, **_kwargs: ({}, {}))

        def __init__(self):
            self._commit_hash = "main"

    transformers = SimpleNamespace(
        __version__="5.19.0", AutoConfig=AutoConfig, PreTrainedConfig=PreTrainedConfig
    )
    monkeypatch.setattr("routetune.training.importlib.metadata.version", lambda _: "5.19.0")
    with pytest.raises(TrainingError, match="injects _commit_hash"):
        verify_transformers_runtime(transformers)


class FakeHubApi:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def model_info(self, model_id, *, revision):
        self.calls.append((model_id, revision))
        if self.error is not None:
            raise self.error
        return self.result


def test_model_revision_resolution_uses_hugging_face_api():
    api = FakeHubApi(SimpleNamespace(sha=MODEL_SHA))
    assert resolve_model_revision(api=api) == MODEL_SHA
    assert api.calls == [(MODEL_ID, "main")]


@pytest.mark.parametrize("metadata", [SimpleNamespace(), SimpleNamespace(sha=None), SimpleNamespace(sha="")])
def test_model_revision_resolution_rejects_missing_sha(metadata):
    with pytest.raises(TrainingError, match="has no commit SHA"):
        resolve_model_revision(api=FakeHubApi(metadata))


@pytest.mark.parametrize("revision", ["main", "a" * 39, "g" * 40, 123])
def test_model_revision_resolution_rejects_invalid_sha(revision):
    with pytest.raises(TrainingError, match="malformed commit SHA"):
        resolve_model_revision(api=FakeHubApi(SimpleNamespace(sha=revision)))


def test_model_revision_resolution_reports_api_failure():
    with pytest.raises(
        TrainingError,
        match="model revision lookup failed.*ConnectionError: network unavailable",
    ):
        resolve_model_revision(api=FakeHubApi(error=ConnectionError("network unavailable")))


def test_model_revision_resolution_reports_unavailable_model():
    class RepositoryNotFoundError(Exception):
        pass

    with pytest.raises(
        TrainingError,
        match="model revision lookup failed.*RepositoryNotFoundError: model unavailable",
    ):
        resolve_model_revision(
            api=FakeHubApi(error=RepositoryNotFoundError("model unavailable"))
        )


def test_quantized_model_and_tokenizer_use_same_immutable_revision(monkeypatch):
    calls = []
    torch = ModuleType("torch")
    torch.float16 = object()
    transformers = ModuleType("transformers")
    config = SimpleNamespace(model_type="qwen2")

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("config", model_id, kwargs))
            return config

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return SimpleNamespace(pad_token_id=0, padding_side=None)

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            quantization = SimpleNamespace(
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            return SimpleNamespace(
                is_loaded_in_4bit=True,
                config=SimpleNamespace(quantization_config=quantization),
            )

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoConfig = AutoConfig
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.BitsAndBytesConfig = BitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    _, _, revision = _load_quantized_base(MODEL_SHA)
    assert revision == MODEL_SHA
    assert [call[2]["revision"] for call in calls] == [MODEL_SHA, MODEL_SHA, MODEL_SHA]
    assert calls[-1][2]["config"] is config
    assert calls[-1][2]["dtype"] is torch.float16
    assert "torch_dtype" not in calls[-1][2]


def test_quantized_model_load_rejects_mutable_revision_before_dependencies():
    with pytest.raises(TrainingError, match="full 40-character hexadecimal SHA"):
        _load_quantized_base("main")


def test_saved_adapter_reload_uses_manifest_revision(monkeypatch, tmp_path):
    run = create_run(
        tmp_path,
        configuration={"fixture": True},
        fingerprints={"selection": "fixture"},
        environment={},
    )
    update_run(run, model_revision=MODEL_SHA)
    captured = []

    def fake_load(revision):
        captured.append(revision)
        return "tokenizer", "model", revision

    monkeypatch.setattr("routetune.training._load_quantized_base", fake_load)
    assert _load_recorded_quantized_base(run) == ("tokenizer", "model", MODEL_SHA)
    assert captured == [MODEL_SHA]


def test_colab_workflow_installs_minimally_then_forces_clean_restart():
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "RouteTune_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "--force-reinstall" not in source
    assert '"--upgrade"' not in source
    assert '"uninstall", "-y", "torchaudio"' in source
    assert "https://download.pytorch.org/whl/cu130" in source
    assert "torch==2.14.1" in source
    assert "torchvision==0.29.1" in source
    assert ".colab-dependencies-installed" in source
    assert "os.kill(os.getpid(), signal.SIGKILL)" in source
    assert "capture_output=True" in source
    assert "assess_pip_check" in source
    assert "dependency-readiness.json" in source
    assert "verify_gpu_environment" in source
    assert "FP32 trainable LoRA adapters" in source
    assert "PreTrainedConfig.__init__" not in source
    assert '_commit_hash = "main"' not in source


def test_gpu_extra_preserves_compatible_runtime_pins():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = set(project["project"]["optional-dependencies"]["gpu"])
    assert {
        "torch==2.14.1",
        "torchvision==0.29.1",
        "numpy==2.0.2",
        "pandas==2.2.3",
        "requests==2.32.4",
        "fsspec==2025.12.0",
        "pyarrow==21.0.0",
    }.issubset(requirements)
    assert set(PINNED_GPU_PACKAGE_VERSIONS).issubset(GPU_PACKAGES)
    assert {"huggingface-hub", "safetensors", "tokenizers"}.issubset(GPU_PACKAGES)


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
