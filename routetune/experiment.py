"""Small, fail-closed filesystem ledger for Phase 3 experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import file_sha256


class ExperimentError(ValueError):
    """Raised when run evidence is missing, stale, or inconsistent."""


STATES = frozenset({"PREPARED", "RUNNING", "TRAINED", "VERIFIED", "FAILED", "INTERRUPTED"})
TRANSITIONS = {
    "PREPARED": frozenset({"RUNNING", "FAILED", "INTERRUPTED"}),
    "RUNNING": frozenset({"TRAINED", "FAILED", "INTERRUPTED"}),
    "TRAINED": frozenset({"VERIFIED", "FAILED", "INTERRUPTED"}),
    "VERIFIED": frozenset(),
    "FAILED": frozenset(),
    "INTERRUPTED": frozenset(),
}
REQUIRED_HELDOUT_EVIDENCE = (
    "fingerprints_verified",
    "loss_masks_verified",
    "gpu_smoke_verified",
    "gradient_step_verified",
    "precision_contract_verified",
    "training_complete",
    "adapter_reloaded",
    "validation_complete",
    "configuration_frozen",
)
MODEL_REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_immutable_model_revision(value: object) -> bool:
    return isinstance(value, str) and MODEL_REVISION_PATTERN.fullmatch(value) is not None


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def create_run(
    runs_dir: Path,
    *,
    configuration: dict[str, Any],
    fingerprints: dict[str, str],
    environment: dict[str, Any],
) -> Path:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:10]}"
    run_dir = runs_dir / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ExperimentError(f"run already exists: {run_dir}") from exc
    manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "state": "PREPARED",
        "configuration": configuration,
        "configuration_sha256": canonical_sha256(configuration),
        "fingerprints": fingerprints,
        "environment": environment,
        "model_revision": None,
        "evidence": {"configuration_frozen": True, "heldout_started": False},
        "artifacts": {},
        "failure": None,
    }
    atomic_json_write(run_dir / "manifest.json", manifest)
    return run_dir


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != "1" or manifest["state"] not in STATES:
            raise ExperimentError(f"invalid run manifest: {path}")
        if manifest["run_id"] != run_dir.name:
            raise ExperimentError("run directory and manifest identity differ")
        if canonical_sha256(manifest["configuration"]) != manifest["configuration_sha256"]:
            raise ExperimentError("run configuration fingerprint is stale")
        if manifest["model_revision"] is not None and not is_immutable_model_revision(
            manifest["model_revision"]
        ):
            raise ExperimentError("run model revision is not a full commit SHA")
        if not isinstance(manifest["evidence"], dict) or not isinstance(manifest["artifacts"], dict):
            raise ExperimentError(f"invalid run manifest: {path}")
    except ExperimentError:
        raise
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"cannot load run manifest {path}: {exc}") from exc
    return manifest


def update_run(
    run_dir: Path,
    *,
    state: str | None = None,
    evidence: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    model_revision: str | None = None,
    failure: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    if state is not None:
        if state not in TRANSITIONS[manifest["state"]]:
            raise ExperimentError(f"invalid run transition: {manifest['state']} -> {state}")
        manifest["state"] = state
    if evidence:
        manifest["evidence"].update(evidence)
    if artifacts:
        manifest["artifacts"].update(artifacts)
    if environment is not None:
        manifest["environment"] = environment
    if model_revision is not None:
        if not is_immutable_model_revision(model_revision):
            raise ExperimentError("model revision must be a full 40-character hexadecimal SHA")
        if manifest["model_revision"] is not None and manifest["model_revision"] != model_revision:
            raise ExperimentError("run model revision cannot change")
        manifest["model_revision"] = model_revision
    if failure is not None:
        manifest["failure"] = failure
    manifest["updated_at_utc"] = utc_now()
    atomic_json_write(run_dir / "manifest.json", manifest)
    return manifest


def mark_failure(run_dir: Path, exc: BaseException, *, interrupted: bool = False) -> None:
    manifest = load_manifest(run_dir)
    if manifest["state"] in {"FAILED", "INTERRUPTED", "VERIFIED"}:
        return
    update_run(
        run_dir,
        state="INTERRUPTED" if interrupted else "FAILED",
        failure={"type": type(exc).__name__, "message": str(exc)},
    )


def adapter_hashes(adapter_dir: Path) -> dict[str, str]:
    required = adapter_dir / "adapter_config.json"
    weights = [adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"]
    if not required.is_file() or not any(path.is_file() for path in weights):
        raise ExperimentError("adapter is incomplete")
    files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
    return {path.relative_to(adapter_dir).as_posix(): file_sha256(path) for path in files}


def verify_adapter(run_dir: Path) -> Path:
    manifest = load_manifest(run_dir)
    if manifest["state"] not in {"TRAINED", "VERIFIED"}:
        raise ExperimentError("adapter requires TRAINED or VERIFIED run evidence")
    if not is_immutable_model_revision(manifest["model_revision"]):
        raise ExperimentError("adapter run has no immutable model revision")
    try:
        record = manifest["artifacts"]["adapter"]
        relative_path = Path(record["path"])
        expected_hashes = record["sha256"]
    except (KeyError, TypeError) as exc:
        raise ExperimentError("adapter artifact is not recorded") from exc
    adapter_dir = (run_dir / relative_path).resolve()
    if not adapter_dir.is_relative_to(run_dir.resolve()):
        raise ExperimentError("adapter artifact escapes the run directory")
    identity_path = adapter_dir / "adapter_identity.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError("adapter identity is missing or invalid") from exc
    if identity != {
        "run_id": manifest["run_id"],
        "configuration_sha256": manifest["configuration_sha256"],
        "model_revision": manifest["model_revision"],
    }:
        raise ExperimentError("adapter identity does not match its run")
    if adapter_hashes(adapter_dir) != expected_hashes:
        raise ExperimentError("adapter files do not match recorded hashes")
    return adapter_dir


def require_heldout_gate(run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    if manifest["state"] != "VERIFIED":
        raise ExperimentError("official held-out evaluation requires a VERIFIED run")
    missing = [name for name in REQUIRED_HELDOUT_EVIDENCE if manifest["evidence"].get(name) is not True]
    if missing:
        raise ExperimentError(f"official held-out gate is missing evidence: {', '.join(missing)}")
    if manifest["evidence"].get("heldout_started") is not False:
        raise ExperimentError("official held-out evaluation was already claimed")
    verify_adapter(run_dir)
    return manifest


def claim_heldout(run_dir: Path) -> None:
    require_heldout_gate(run_dir)
    update_run(run_dir, evidence={"heldout_started": True, "heldout_started_at_utc": utc_now()})
