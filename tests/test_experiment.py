import json

import pytest

from routetune.experiment import (
    ExperimentError,
    adapter_hashes,
    atomic_json_write,
    canonical_sha256,
    create_run,
    load_manifest,
    mark_failure,
    require_heldout_gate,
    update_run,
    verify_adapter,
)

MODEL_SHA = "a" * 40


def new_run(tmp_path):
    return create_run(
        tmp_path,
        configuration={"model": "fixture", "seed": 42},
        fingerprints={"selection": "abc"},
        environment={"cuda_available": False},
    )


def test_manifest_is_atomic_and_state_transitions_are_ordered(tmp_path):
    run = new_run(tmp_path)
    assert load_manifest(run)["state"] == "PREPARED"
    assert not list(run.glob("*.part"))

    update_run(run, state="RUNNING")
    update_run(run, state="TRAINED")
    update_run(run, state="VERIFIED")
    with pytest.raises(ExperimentError, match="invalid run transition"):
        update_run(run, state="RUNNING")


def test_stale_configuration_is_rejected(tmp_path):
    run = new_run(tmp_path)
    path = run / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["configuration"]["seed"] = 99
    atomic_json_write(path, manifest)

    with pytest.raises(ExperimentError, match="configuration fingerprint is stale"):
        load_manifest(run)


def test_failure_and_interruption_preserve_evidence(tmp_path):
    failed = new_run(tmp_path / "failed")
    update_run(failed, state="RUNNING")
    mark_failure(failed, RuntimeError("CUDA OOM"))
    assert load_manifest(failed)["failure"] == {"type": "RuntimeError", "message": "CUDA OOM"}
    assert load_manifest(failed)["state"] == "FAILED"

    interrupted = new_run(tmp_path / "interrupted")
    update_run(interrupted, state="RUNNING")
    mark_failure(interrupted, KeyboardInterrupt(), interrupted=True)
    assert load_manifest(interrupted)["state"] == "INTERRUPTED"


def complete_adapter(run):
    update_run(run, state="RUNNING", model_revision=MODEL_SHA)
    adapter = run / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    atomic_json_write(
        adapter / "adapter_identity.json",
        {
            "run_id": run.name,
            "configuration_sha256": canonical_sha256({"model": "fixture", "seed": 42}),
            "model_revision": MODEL_SHA,
        },
    )
    hashes = adapter_hashes(adapter)
    update_run(
        run,
        state="TRAINED",
        artifacts={"adapter": {"path": "adapter", "sha256": hashes}},
    )
    return adapter


def test_adapter_identity_and_hashes_reject_stale_artifacts(tmp_path):
    run = new_run(tmp_path)
    adapter = complete_adapter(run)
    assert verify_adapter(run) == adapter.resolve()

    (adapter / "adapter_model.safetensors").write_bytes(b"stale")
    with pytest.raises(ExperimentError, match="recorded hashes"):
        verify_adapter(run)


def test_adapter_without_immutable_model_revision_is_rejected(tmp_path):
    run = new_run(tmp_path)
    update_run(run, state="RUNNING")
    adapter = run / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    atomic_json_write(
        adapter / "adapter_identity.json",
        {
            "run_id": run.name,
            "configuration_sha256": canonical_sha256({"model": "fixture", "seed": 42}),
            "model_revision": None,
        },
    )
    update_run(
        run,
        state="TRAINED",
        artifacts={"adapter": {"path": "adapter", "sha256": adapter_hashes(adapter)}},
    )
    with pytest.raises(ExperimentError, match="no immutable model revision"):
        verify_adapter(run)


def test_manifest_model_revision_is_immutable_and_well_formed(tmp_path):
    run = new_run(tmp_path)
    update_run(run, model_revision=MODEL_SHA)
    assert load_manifest(run)["model_revision"] == MODEL_SHA

    with pytest.raises(ExperimentError, match="cannot change"):
        update_run(run, model_revision="b" * 40)
    with pytest.raises(ExperimentError, match="40-character"):
        update_run(run, model_revision="main")

    path = run / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model_revision"] = "malformed"
    atomic_json_write(path, manifest)
    with pytest.raises(ExperimentError, match="not a full commit SHA"):
        load_manifest(run)


def test_heldout_gate_requires_all_verified_evidence_and_is_single_use(tmp_path):
    run = new_run(tmp_path)
    complete_adapter(run)
    update_run(run, state="VERIFIED")
    with pytest.raises(ExperimentError, match="missing evidence"):
        require_heldout_gate(run)

    required = {
        "fingerprints_verified": True,
        "loss_masks_verified": True,
        "gpu_smoke_verified": True,
        "gradient_step_verified": True,
        "training_complete": True,
        "adapter_reloaded": True,
        "validation_complete": True,
    }
    update_run(run, evidence=required)
    assert require_heldout_gate(run)["state"] == "VERIFIED"
    update_run(run, evidence={"heldout_started": True})
    with pytest.raises(ExperimentError, match="already claimed"):
        require_heldout_gate(run)

