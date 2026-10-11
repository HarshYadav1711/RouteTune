"""Verified QLoRA preparation, training, reload, and controlled evaluation."""

from __future__ import annotations

import gc
import importlib.metadata
import inspect
import json
import math
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .baseline import fit_baseline, predict_records
from .challenge import EXPECTED_CHALLENGE_FINGERPRINT, challenge_fingerprint
from .config import (
    DEFAULT_CONFIG,
    EXPECTED_SELECTION_FINGERPRINT,
    EXPECTED_TEST_SHA256,
    EXPECTED_TRAIN_SHA256,
    MAX_NEW_TOKENS,
    MODEL_ID,
    TARGET_LABELS,
)
from .data import DatasetSplits, OFFICIAL_SOURCES, Record, build_splits, fetch_source, selection_fingerprint
from .evaluation import PredictionRecord, evaluate_predictions, prediction_record
from .experiment import (
    ExperimentError,
    adapter_hashes,
    atomic_json_write,
    claim_heldout,
    create_run,
    is_immutable_model_revision,
    load_manifest,
    mark_failure,
    update_run,
    verify_adapter,
)
from .inference import QwenAdapter, build_messages


class TrainingError(RuntimeError):
    """Raised when a Phase 3 execution gate cannot be proven."""


@dataclass(frozen=True)
class QLoRAConfig:
    model_id: str = MODEL_ID
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    epochs: int = 2
    learning_rate: float = 1.5e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_batch_size: int = 8
    max_sequence_length: int = 256
    gradient_checkpointing: bool = True
    fp16: bool = True
    logging_steps: int = 10
    seed: int = 42


FIXED_QLORA_CONFIG = QLoRAConfig()
GPU_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "fsspec",
    "numpy",
    "pandas",
    "pyarrow",
    "requests",
    "huggingface-hub",
    "safetensors",
    "tokenizers",
    "scikit-learn",
)
PINNED_GPU_PACKAGE_VERSIONS = {
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
    "datasets": "4.7.0",
    "fsspec": "2025.12.0",
    "numpy": "2.0.2",
    "pandas": "2.2.3",
    "peft": "0.21.2",
    "pyarrow": "21.0.0",
    "requests": "2.32.4",
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "transformers": "5.19.0",
    "trl": "1.15.0",
}
ROUTETUNE_DEPENDENCY_ROOTS = (*PINNED_GPU_PACKAGE_VERSIONS, "routetune")
COLAB_OPERATIONAL_ROOTS = ("google-colab", "gcsfs")


def validate_qlora_config(config: QLoRAConfig) -> None:
    if config != FIXED_QLORA_CONFIG:
        raise TrainingError("Phase 3 QLoRA settings differ from the frozen configuration")


def prompt_completion(record: Record) -> dict[str, list[dict[str, str]]]:
    if record.category not in TARGET_LABELS:
        raise TrainingError(f"training record has an unknown label: {record.category!r}")
    return {
        "prompt": list(build_messages(record.text)),
        "completion": [{"role": "assistant", "content": record.category}],
    }


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def tokenize_training_record(
    record: Record, tokenizer: Any, *, max_length: int = 256
) -> dict[str, list[int]]:
    example = prompt_completion(record)
    prompt = example["prompt"]
    full_messages = prompt + example["completion"]
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    )
    input_ids = _token_ids(
        tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
    )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise TrainingError("chat template did not preserve the inference prompt prefix")
    if len(input_ids) > max_length:
        raise TrainingError(
            f"tokenized example has {len(input_ids)} tokens; refusing to truncate its target"
        )
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    supervised_ids = [token for token, label in zip(input_ids, labels) if label != -100]
    if not supervised_ids:
        raise TrainingError("training example has zero supervised target tokens")
    decoded = tokenizer.decode(supervised_ids, skip_special_tokens=True).strip()
    if decoded != record.category:
        raise TrainingError(
            f"completion tokens decode to {decoded!r}, expected {record.category!r}"
        )
    result = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
    verify_tokenized_sample(result)
    return result


def verify_tokenized_sample(sample: dict[str, list[int]]) -> None:
    lengths = {len(sample[name]) for name in ("input_ids", "attention_mask", "labels")}
    if len(lengths) != 1 or not lengths.pop():
        raise TrainingError("tokenized sample arrays must have one non-zero length")
    supervised = [index for index, label in enumerate(sample["labels"]) if label != -100]
    if not supervised:
        raise TrainingError("tokenized sample has zero supervised tokens")
    first = supervised[0]
    if any(label != -100 for label in sample["labels"][:first]):
        raise TrainingError("prompt tokens are not fully masked")
    if any(mask != 1 for mask in sample["attention_mask"]):
        raise TrainingError("un-padded sample has an invalid attention mask")
    if sample["labels"][first:] != sample["input_ids"][first:]:
        raise TrainingError("supervised labels differ from completion input tokens")


def tokenize_records(records: tuple[Record, ...], tokenizer: Any) -> list[dict[str, list[int]]]:
    if not records or any(record.source != "official_train" for record in records):
        raise TrainingError("training and validation records must come only from official_train")
    examples = [
        tokenize_training_record(record, tokenizer, max_length=FIXED_QLORA_CONFIG.max_sequence_length)
        for record in records
    ]
    representatives = {record.category for record in records}
    if representatives != set(TARGET_LABELS):
        raise TrainingError("loss masks must be verified on at least one example per class")
    return examples


def pad_tokenized_batch(
    samples: list[dict[str, list[int]]], pad_token_id: int
) -> dict[str, list[list[int]]]:
    if not samples:
        raise TrainingError("cannot collate an empty batch")
    width = max(len(sample["input_ids"]) for sample in samples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for sample in samples:
        verify_tokenized_sample(sample)
        padding = width - len(sample["input_ids"])
        batch["input_ids"].append(sample["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(sample["attention_mask"] + [0] * padding)
        batch["labels"].append(sample["labels"] + [-100] * padding)
    verify_masked_batch(batch)
    return batch


def verify_masked_batch(batch: dict[str, Any]) -> None:
    rows = {name: value.tolist() if hasattr(value, "tolist") else value for name, value in batch.items()}
    for input_ids, attention, labels in zip(rows["input_ids"], rows["attention_mask"], rows["labels"]):
        if not any(label != -100 for label in labels):
            raise TrainingError("processed batch contains a sample with zero supervised tokens")
        for token, mask, label in zip(input_ids, attention, labels):
            if mask == 0 and label != -100:
                raise TrainingError("padding token contributes to training loss")
            if label != -100 and label != token:
                raise TrainingError("completion label does not match its input token")


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        padded = pad_tokenized_batch(samples, self.pad_token_id)
        return {name: torch.tensor(value, dtype=torch.long) for name, value in padded.items()}


def environment_report() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in GPU_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report: dict[str, Any] = {
        "python": sys.version,
        "operating_system": platform.platform(),
        "packages": versions,
        "cuda_available": False,
        "cuda_runtime": None,
        "gpu_model": None,
        "gpu_total_vram_bytes": None,
    }
    try:
        import torch

        report.update(
            cuda_available=torch.cuda.is_available(),
            cuda_runtime=torch.version.cuda,
        )
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            report.update(
                gpu_model=properties.name,
                gpu_total_vram_bytes=properties.total_memory,
                gpu_compute_capability=list(torch.cuda.get_device_capability(0)),
            )
    except ImportError:
        pass
    return report


def verify_pinned_gpu_dependencies() -> None:
    mismatches = []
    for package, expected in PINNED_GPU_PACKAGE_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual.split("+", 1)[0] != expected:
            mismatches.append(f"{package}={actual} (expected {expected})")
    if mismatches:
        raise TrainingError(f"installed GPU dependency lock differs: {', '.join(mismatches)}")


def dependency_packages(roots: Iterable[str]) -> set[str]:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    required = {canonicalize_name(package) for package in roots}
    pending = list(required)
    while pending:
        package = pending.pop()
        try:
            requirements = importlib.metadata.requires(package) or ()
        except importlib.metadata.PackageNotFoundError:
            continue
        for value in requirements:
            requirement = Requirement(value)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            name = canonicalize_name(requirement.name)
            if name not in required:
                required.add(name)
                pending.append(name)
    return required


def assess_pip_check(
    output: str,
    returncode: int,
    *,
    route_packages: set[str] | None = None,
    operational_packages: set[str] | None = None,
) -> dict[str, Any]:
    from packaging.utils import canonicalize_name

    route = dependency_packages(ROUTETUNE_DEPENDENCY_ROOTS) if route_packages is None else {
        canonicalize_name(package) for package in route_packages
    }
    operational = dependency_packages(COLAB_OPERATIONAL_ROOTS) if operational_packages is None else {
        canonicalize_name(package) for package in operational_packages
    }
    issues = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
        and line.strip() != "No broken requirements found."
        and not line.lstrip().startswith("WARNING:")
    ]
    route_conflicts: list[str] = []
    operational_conflicts: list[str] = []
    unrelated: list[str] = []
    for line in issues:
        parts = line.split()
        if len(parts) < 3 or parts[2] not in {"has", "requires", "is"}:
            route_conflicts.append(line)
            continue
        requester = canonicalize_name(parts[0])
        if requester in route:
            route_conflicts.append(line)
        elif requester in operational:
            operational_conflicts.append(line)
        else:
            unrelated.append(line)
    if returncode != 0 and not issues:
        route_conflicts.append(f"pip check exited with status {returncode} without a parseable conflict")
    return {
        "ready": not route_conflicts and not operational_conflicts,
        "pip_check_clean": returncode == 0,
        "route_conflicts": route_conflicts,
        "operational_conflicts": operational_conflicts,
        "unrelated_conflicts": unrelated,
        "unrelated_conflicts_reason": (
            "requesting packages are outside RouteTune and Colab/GPU dependency closures"
            if unrelated
            else None
        ),
        "pip_check_output": output,
    }


def verify_imported_package_version(package: str, module: Any) -> None:
    installed = importlib.metadata.version(package)
    imported = getattr(module, "__version__", None)
    if imported != installed:
        label = "Transformers" if package == "transformers" else package
        raise TrainingError(
            f"mixed {label} runtime: imported {imported!r}, installed {installed!r}; "
            "restart the runtime after installation"
        )


def verify_torch_runtime(torch: Any, torchvision: Any) -> None:
    verify_imported_package_version("torch", torch)
    verify_imported_package_version("torchvision", torchvision)
    if getattr(getattr(torch, "version", None), "cuda", None) != "13.0":
        raise TrainingError("installed PyTorch is not the required CUDA 13.0 build")


def verify_transformers_runtime(transformers: Any) -> None:
    verify_imported_package_version("transformers", transformers)
    auto_config = getattr(transformers, "AutoConfig", None)
    pretrained_config = getattr(transformers, "PreTrainedConfig", None)
    if not callable(getattr(auto_config, "from_pretrained", None)) or not callable(
        getattr(pretrained_config, "get_config_dict", None)
    ):
        raise TrainingError("installed Transformers configuration APIs are incompatible")
    try:
        injected_revision = getattr(pretrained_config(), "_commit_hash", None)
    except Exception as exc:
        raise TrainingError(f"Transformers configuration initialization is incompatible: {exc}") from exc
    if injected_revision is not None:
        raise TrainingError(
            "Transformers PreTrainedConfig injects _commit_hash; use an unmodified runtime"
        )


def verify_training_apis() -> None:
    verify_pinned_gpu_dependencies()
    try:
        import torch
        import torchvision
        import transformers
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise TrainingError(f"GPU dependency is unavailable: {exc}") from exc
    verify_torch_runtime(torch, torchvision)
    verify_transformers_runtime(transformers)
    required_sft_config = {
        "completion_only_loss",
        "dataset_kwargs",
        "eval_strategy",
        "gradient_checkpointing",
        "max_length",
    }
    config_parameters = set(inspect.signature(SFTConfig).parameters)
    trainer_parameters = set(inspect.signature(SFTTrainer).parameters)
    if not required_sft_config.issubset(config_parameters):
        missing = sorted(required_sft_config - config_parameters)
        raise TrainingError(f"installed TRL SFTConfig lacks required APIs: {missing}")
    if not {"processing_class", "data_collator", "train_dataset", "eval_dataset"}.issubset(trainer_parameters):
        raise TrainingError("installed TRL SFTTrainer lacks required dataset/collator APIs")
    if not all(
        callable(value)
        for value in (
            LoraConfig,
            PeftModel.from_pretrained,
            get_peft_model,
            prepare_model_for_kbit_training,
            AutoConfig.from_pretrained,
            AutoTokenizer.from_pretrained,
            AutoModelForCausalLM.from_pretrained,
            BitsAndBytesConfig,
        )
    ):
        raise TrainingError("installed Transformers/PEFT APIs are incompatible")


def verify_gpu_environment(*, minimum_vram_gib: int = 12) -> dict[str, Any]:
    verify_training_apis()
    report = environment_report()
    if not report["cuda_available"]:
        raise TrainingError("CUDA is unavailable; real QLoRA execution is blocked")
    minimum = minimum_vram_gib * 1024**3
    if not isinstance(report["gpu_total_vram_bytes"], int) or report["gpu_total_vram_bytes"] < minimum:
        raise TrainingError(f"fixed training run requires at least {minimum_vram_gib} GiB GPU VRAM")
    return report


def load_locked_splits(cache_dir: Path, *, offline: bool) -> tuple[DatasetSplits, dict[str, str]]:
    sources = {spec.name: fetch_source(spec, cache_dir, offline=offline) for spec in OFFICIAL_SOURCES}
    fingerprints = {
        "official_train_sha256": sources["official_train"].provenance.sha256,
        "official_test_sha256": sources["official_test"].provenance.sha256,
    }
    expected = {
        "official_train_sha256": EXPECTED_TRAIN_SHA256,
        "official_test_sha256": EXPECTED_TEST_SHA256,
    }
    if fingerprints != expected:
        raise TrainingError(f"official source fingerprints changed: {fingerprints}")
    splits = build_splits(sources["official_train"], sources["official_test"])
    fingerprints["selection_sha256"] = selection_fingerprint(splits)
    fingerprints["challenge_sha256"] = challenge_fingerprint()
    if fingerprints["selection_sha256"] != EXPECTED_SELECTION_FINGERPRINT:
        raise TrainingError("locked dataset selection fingerprint changed")
    if fingerprints["challenge_sha256"] != EXPECTED_CHALLENGE_FINGERPRINT:
        raise TrainingError("frozen challenge fingerprint changed")
    return splits, fingerprints


def prepare_phase3_run(
    cache_dir: Path, runs_dir: Path, *, offline: bool = True
) -> Path:
    _, fingerprints = load_locked_splits(cache_dir, offline=offline)
    return create_run(
        runs_dir,
        configuration={"qlora": asdict(FIXED_QLORA_CONFIG), "data": asdict(DEFAULT_CONFIG)},
        fingerprints=fingerprints,
        environment=environment_report(),
    )


def resolve_model_revision(model_id: str = MODEL_ID, *, api: Any | None = None) -> str:
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise TrainingError("huggingface_hub is unavailable; model revision cannot be resolved") from exc
        api = HfApi()
    try:
        model_info = api.model_info(model_id, revision="main")
    except Exception as exc:
        raise TrainingError(
            f"Hugging Face model revision lookup failed for {model_id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    revision = getattr(model_info, "sha", None)
    if revision is None or revision == "":
        raise TrainingError(f"Hugging Face model metadata for {model_id!r} has no commit SHA")
    if not is_immutable_model_revision(revision):
        raise TrainingError(
            f"Hugging Face model metadata for {model_id!r} returned a malformed commit SHA: "
            f"{revision!r}"
        )
    return revision


def _load_quantized_base(revision: str) -> tuple[Any, Any, str]:
    if not is_immutable_model_revision(revision):
        raise TrainingError("model revision must be a full 40-character hexadecimal SHA")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        config = AutoConfig.from_pretrained(MODEL_ID, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    except Exception as exc:
        raise TrainingError(
            f"base-model configuration/tokenizer load failed at immutable revision {revision}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        config=config,
        device_map={"": 0},
        quantization_config=quantization,
        torch_dtype=torch.float16,
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise TrainingError("model did not load with real four-bit quantization")
    actual = model.config.quantization_config
    read = lambda name: actual.get(name) if isinstance(actual, dict) else getattr(actual, name, None)
    if read("bnb_4bit_quant_type") != "nf4" or read("bnb_4bit_use_double_quant") is not True:
        raise TrainingError("loaded quantization is not NF4 with double quantization")
    return tokenizer, model, revision


def _load_recorded_quantized_base(run_dir: Path) -> tuple[Any, Any, str]:
    revision = load_manifest(run_dir)["model_revision"]
    if not is_immutable_model_revision(revision):
        raise TrainingError("run manifest does not contain an immutable model revision")
    return _load_quantized_base(revision)


def _attach_lora(model: Any) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(FIXED_QLORA_CONFIG.target_modules),
        ),
    )
    return model, adapter_parameter_evidence(model)


def adapter_parameter_evidence(model: Any) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise TrainingError("trainable parameters are not restricted to LoRA adapters")
    matched = {
        target
        for target in FIXED_QLORA_CONFIG.target_modules
        if any(name.endswith(target) and hasattr(module, "lora_A") for name, module in model.named_modules())
    }
    if matched != set(FIXED_QLORA_CONFIG.target_modules):
        raise TrainingError(f"LoRA target modules were not all matched: {sorted(matched)}")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    total_count = trainable_count + sum(parameter.numel() for _, parameter in frozen)
    if trainable_count >= total_count / 5:
        raise TrainingError("adapter parameter count is implausibly large")
    return {
        "total_parameters": total_count,
        "trainable_parameters": trainable_count,
        "trainable_fraction": trainable_count / total_count,
        "matched_target_modules": sorted(matched),
        "base_parameters_frozen": True,
    }


def _gradient_smoke(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    import torch

    model.train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    snapshots = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1.5e-4)
    device = next(parameter for _, parameter in trainable).device
    moved = {name: value.to(device) for name, value in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    loss = model(**moved).loss
    if not torch.isfinite(loss):
        raise TrainingError("GPU smoke loss is not finite")
    loss.backward()
    gradient_names = [
        name
        for name, parameter in trainable
        if parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
    ]
    if not gradient_names:
        raise TrainingError("no LoRA parameter received a finite non-zero gradient")
    if any(parameter.grad is not None for _, parameter in model.named_parameters() if not parameter.requires_grad):
        raise TrainingError("a frozen base parameter received a gradient")
    optimizer.step()
    changed = [name for name, parameter in trainable if not torch.equal(parameter.detach().cpu(), snapshots[name])]
    if not changed:
        raise TrainingError("optimizer step did not update an adapter parameter")
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(snapshots[name].to(parameter.device))
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().cpu()),
        "parameters_with_nonzero_gradients": len(gradient_names),
        "parameters_updated": len(changed),
        "diagnostic_update_restored": True,
    }


def _evaluate_qwen(
    tokenizer: Any,
    model: Any,
    records: tuple[Record, ...],
    split_name: str,
    *,
    progress_path: Path | None = None,
) -> tuple[PredictionRecord, ...]:
    import torch

    model.eval()
    adapter = QwenAdapter(tokenizer, model, synchronize=torch.cuda.synchronize)
    predictions = []
    for index, record in enumerate(records):
        result = adapter.classify(record.text)
        predictions.append(
            prediction_record(
                example_id=f"{split_name}-{index:03d}",
                input_message=record.text,
                expected_class=record.category,
                raw_output=result.raw_output,
                labels=TARGET_LABELS,
                latency_seconds=result.generation_seconds,
            )
        )
        if progress_path is not None:
            from dataclasses import asdict as convert

            atomic_json_write(
                progress_path,
                {
                    "complete": False,
                    "prediction_count": len(predictions),
                    "predictions": [convert(row) for row in predictions],
                },
            )
    result = tuple(predictions)
    if progress_path is not None:
        _write_evaluation(progress_path, result)
    return result


def _write_evaluation(path: Path, predictions: tuple[PredictionRecord, ...]) -> None:
    from dataclasses import asdict as convert

    report = evaluate_predictions(predictions, TARGET_LABELS)
    latencies = [row.latency_seconds for row in predictions if row.latency_seconds is not None]
    atomic_json_write(
        path,
        {
            "complete": True,
            "metrics": report.to_dict(),
            "latency_seconds": {
                "count": len(latencies),
                "minimum": min(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "mean": statistics.fmean(latencies) if latencies else None,
                "maximum": max(latencies) if latencies else None,
            },
            "predictions": [convert(row) for row in predictions],
        },
    )


def verify_evaluation_artifact(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        predictions = tuple(PredictionRecord(**row) for row in value["predictions"])
        if value["complete"] is not True:
            raise TrainingError(f"evaluation artifact is incomplete: {path}")
        if [row.example_id for row in predictions] != expected_ids:
            raise TrainingError(f"evaluation example order is incomplete or changed: {path}")
        if evaluate_predictions(predictions, TARGET_LABELS).to_dict() != value["metrics"]:
            raise TrainingError(f"evaluation metrics do not match predictions: {path}")
        latencies = [row.latency_seconds for row in predictions if row.latency_seconds is not None]
        expected_latency = {
            "count": len(latencies),
            "minimum": min(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        }
        if value["latency_seconds"] != expected_latency:
            raise TrainingError(f"evaluation latency summary does not match predictions: {path}")
    except TrainingError:
        raise
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot verify evaluation artifact {path}: {exc}") from exc
    return value


def assert_identical_order(*groups: Iterable[PredictionRecord]) -> None:
    identifiers = [[row.example_id for row in group] for group in groups]
    if not identifiers or any(current != identifiers[0] for current in identifiers[1:]):
        raise TrainingError("evaluation conditions do not contain identical ordered examples")


def execute_phase3_training(run_dir: Path, cache_dir: Path) -> Path:
    """Execute Gates B-D. Official held-out data is not evaluated here."""

    validate_qlora_config(FIXED_QLORA_CONFIG)
    manifest = load_manifest(run_dir)
    if manifest["state"] != "PREPARED":
        raise TrainingError("training requires a fresh PREPARED run")
    splits, fingerprints = load_locked_splits(cache_dir, offline=True)
    if manifest["fingerprints"] != fingerprints:
        raise TrainingError("run fingerprints do not match the current locked data")
    environment = verify_gpu_environment()
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(FIXED_QLORA_CONFIG.seed)
    torch.cuda.manual_seed_all(FIXED_QLORA_CONFIG.seed)
    update_run(run_dir, state="RUNNING", environment=environment)
    try:
        revision = resolve_model_revision()
        update_run(run_dir, model_revision=revision, evidence={"fingerprints_verified": True})
        tokenizer, base_model, revision = _load_quantized_base(revision)
        smoke = QwenAdapter(tokenizer, base_model, synchronize=torch.cuda.synchronize).classify(
            "My new card has not arrived yet."
        )
        update_run(
            run_dir,
            evidence={
                "gpu_smoke_verified": True,
                "gpu_smoke_execution_succeeded": True,
                "gpu_smoke": asdict(smoke),
            },
        )

        untuned_predictions = _evaluate_qwen(tokenizer, base_model, splits.validation, "validation")
        untuned_path = run_dir / "untuned-validation.json"
        _write_evaluation(untuned_path, untuned_predictions)
        verify_evaluation_artifact(
            untuned_path, [f"validation-{index:03d}" for index in range(len(splits.validation))]
        )

        train_rows = tokenize_records(splits.train, tokenizer)
        validation_rows = tokenize_records(splits.validation, tokenizer)
        collator = CompletionOnlyCollator(tokenizer.pad_token_id)
        model, parameters = _attach_lora(base_model)
        diagnostic_batch = collator(train_rows[:2])
        verify_masked_batch(diagnostic_batch)
        gradient = _gradient_smoke(model, diagnostic_batch)
        update_run(
            run_dir,
            evidence={
                "loss_masks_verified": True,
                "gradient_step_verified": True,
                "gradient_smoke": gradient,
                "parameters": parameters,
            },
        )

        arguments = SFTConfig(
            output_dir=str(run_dir / "trainer-output"),
            num_train_epochs=2,
            learning_rate=1.5e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=8,
            max_length=256,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=True,
            logging_steps=10,
            logging_strategy="steps",
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
            completion_only_loss=True,
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        trainer = SFTTrainer(
            model=model,
            args=arguments,
            train_dataset=Dataset.from_list(train_rows),
            eval_dataset=Dataset.from_list(validation_rows),
            data_collator=collator,
            processing_class=tokenizer,
        )
        verify_masked_batch(next(iter(trainer.get_train_dataloader())))
        started = perf_counter()
        trainer.train()
        elapsed = perf_counter() - started
        logs = trainer.state.log_history
        losses = [row[key] for row in logs for key in ("loss", "eval_loss") if key in row]
        if not losses or any(not math.isfinite(float(loss)) for loss in losses):
            raise TrainingError("training log contains no finite training/evaluation loss")
        atomic_json_write(run_dir / "training-log.json", {"elapsed_seconds": elapsed, "log_history": logs})

        before_reload = _evaluate_qwen(tokenizer, model, splits.validation[:3], "reload-check")
        adapter_dir = run_dir / "adapter"
        if adapter_dir.exists():
            raise TrainingError("adapter output already exists")
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        identity = {
            "run_id": manifest["run_id"],
            "configuration_sha256": manifest["configuration_sha256"],
            "model_revision": revision,
        }
        atomic_json_write(adapter_dir / "adapter_identity.json", identity)
        hashes = adapter_hashes(adapter_dir)
        update_run(
            run_dir,
            state="TRAINED",
            evidence={"training_complete": True, "training_elapsed_seconds": elapsed},
            artifacts={
                "adapter": {"path": "adapter", "sha256": hashes},
                "training_log": "training-log.json",
                "untuned_validation": untuned_path.name,
            },
        )

        del trainer, model, base_model
        gc.collect()
        torch.cuda.empty_cache()
        tokenizer, reloaded_base, reloaded_revision = _load_recorded_quantized_base(run_dir)
        if reloaded_revision != revision:
            raise TrainingError("base-model revision changed during adapter reload")
        from peft import PeftModel
        from transformers import AutoTokenizer

        verified_adapter = verify_adapter(run_dir)
        tokenizer = AutoTokenizer.from_pretrained(verified_adapter, local_files_only=True)
        reloaded = PeftModel.from_pretrained(reloaded_base, verified_adapter)
        if not getattr(reloaded, "peft_config", None):
            raise TrainingError("reloaded model has no active PEFT adapter")
        after_reload = _evaluate_qwen(tokenizer, reloaded, splits.validation[:3], "reload-check")
        if [row.raw_output for row in before_reload] != [row.raw_output for row in after_reload]:
            raise TrainingError("saved/reloaded adapter predictions are not serialization-consistent")
        tuned_predictions = _evaluate_qwen(tokenizer, reloaded, splits.validation, "validation")
        tuned_path = run_dir / "adapted-validation.json"
        _write_evaluation(tuned_path, tuned_predictions)
        verify_evaluation_artifact(
            tuned_path, [f"validation-{index:03d}" for index in range(len(splits.validation))]
        )
        update_run(
            run_dir,
            state="VERIFIED",
            evidence={
                "adapter_reloaded": True,
                "validation_complete": True,
                "reload_outputs": [row.raw_output for row in after_reload],
            },
            artifacts={"adapted_validation": tuned_path.name},
        )
        del reloaded, reloaded_base
        gc.collect()
        torch.cuda.empty_cache()
        return run_dir
    except KeyboardInterrupt as exc:
        mark_failure(run_dir, exc, interrupted=True)
        raise
    except BaseException as exc:
        mark_failure(run_dir, exc)
        if isinstance(exc, (ExperimentError, TrainingError)):
            raise
        raise TrainingError(f"Phase 3 GPU execution failed: {exc}") from exc


def execute_final_heldout(run_dir: Path, cache_dir: Path) -> Path:
    """Claim and run the one authorized held-out comparison after Gates A-D."""

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    claim_heldout(run_dir)
    try:
        partial_paths = {
            "tfidf": run_dir / "heldout-tfidf.json",
            "untuned_qwen": run_dir / "heldout-untuned-qwen.json",
            "qlora_qwen": run_dir / "heldout-qlora-qwen.json",
        }
        update_run(
            run_dir,
            artifacts={"heldout_partial_or_complete": {key: path.name for key, path in partial_paths.items()}},
        )
        splits, fingerprints = load_locked_splits(cache_dir, offline=True)
        manifest = load_manifest(run_dir)
        if manifest["fingerprints"] != fingerprints:
            raise TrainingError("held-out fingerprints differ from the verified run")
        pipeline, fit_seconds = fit_baseline(splits.train, DEFAULT_CONFIG.seed)
        baseline = predict_records(pipeline, splits.test, TARGET_LABELS, "test")
        _write_evaluation(partial_paths["tfidf"], baseline)
        tokenizer, base, revision = _load_recorded_quantized_base(run_dir)
        if revision != manifest["model_revision"]:
            raise TrainingError("held-out base-model revision differs from training")
        untuned = _evaluate_qwen(
            tokenizer, base, splits.test, "test", progress_path=partial_paths["untuned_qwen"]
        )
        del base
        gc.collect()
        torch.cuda.empty_cache()
        tokenizer, base, revision = _load_recorded_quantized_base(run_dir)
        verified_adapter = verify_adapter(run_dir)
        tokenizer = AutoTokenizer.from_pretrained(verified_adapter, local_files_only=True)
        adapted_model = PeftModel.from_pretrained(base, verified_adapter)
        adapted = _evaluate_qwen(
            tokenizer, adapted_model, splits.test, "test", progress_path=partial_paths["qlora_qwen"]
        )
        assert_identical_order(baseline, untuned, adapted)
        expected_ids = [f"test-{index:03d}" for index in range(len(splits.test))]
        for path in partial_paths.values():
            verify_evaluation_artifact(path, expected_ids)
        result_path = run_dir / "heldout-comparison.json"
        conditions = {
            name: verify_evaluation_artifact(partial_paths[name], expected_ids)
            for name in ("tfidf", "untuned_qwen", "qlora_qwen")
        }
        atomic_json_write(
            result_path,
            {
                "evaluation_split": "official_test",
                "prediction_count_per_condition": len(splits.test),
                "baseline_fit_seconds": fit_seconds,
                "timing_scope": {
                    "tfidf": "CPU Pipeline.predict including TF-IDF transform and classification",
                    "untuned_qwen": "synchronized GPU generation only; tokenization excluded",
                    "qlora_qwen": "synchronized GPU generation only; tokenization excluded",
                },
                "conditions": conditions,
            },
        )
        update_run(
            run_dir,
            evidence={"heldout_complete": True},
            artifacts={"heldout_comparison": result_path.name},
        )
        del adapted_model, base
        gc.collect()
        torch.cuda.empty_cache()
        return result_path
    except BaseException as exc:
        update_run(
            run_dir,
            evidence={
                "heldout_complete": False,
                "heldout_failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
