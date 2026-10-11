from dataclasses import replace
import json
from pathlib import Path
import sys
import tomllib
from types import ModuleType, SimpleNamespace

import pytest

from routetune.config import MODEL_ID, TARGET_LABELS
from routetune.data import Record
from routetune.evaluation import prediction_record
from routetune.experiment import create_run, update_run
from routetune.training import (
    FIXED_QLORA_CONFIG,
    GPU_PACKAGES,
    PINNED_GPU_PACKAGE_VERSIONS,
    TrainingError,
    _load_quantized_base,
    _load_recorded_quantized_base,
    assess_pip_check,
    adapter_parameter_evidence,
    assert_identical_order,
    pad_tokenized_batch,
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
    verify_transformers_runtime,
)


MODEL_SHA = "0123456789abcdef0123456789abcdef01234567"


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
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
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
