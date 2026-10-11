# RouteTune engineering contract

The experiment contract in `docs/EXPERIMENT_PROTOCOL.md` is authoritative. Keep changes inside the active phase and prefer small, typed modules and pure functions. Do not add dependencies without a concrete need.

## Phase boundaries

- Phase 1: package/config, validated data ingestion and cache, deterministic balanced splits, provenance, diagnostics, CPU-only tests, and documentation.
- Phase 2: strict evaluation engine, validation-only classical baseline, frozen Qwen inference contract, and frozen robustness diagnostics.
- Phase 3: QLoRA training and the controlled GPU experiment.
- Phase 4: diagnostics and portfolio release.

Never add model downloads, training, inference, GPU packages, services, deployment, authentication, or a database during Phase 1.

During Phase 2, never fit on validation data or evaluate the official held-out test split. The challenge suite is diagnostic only: do not tune against it or copy failures into training. Qwen dependencies remain optional and model imports must stay lazy so ordinary tests remain CPU-only.

## Evidence and testing

Use these exact states: **NOT IMPLEMENTED**, **IMPLEMENTED, UNVERIFIED**, **VERIFIED**, **FAILED**, or **BLOCKED**. A unit test verifies only the behavior it asserts; it does not verify official data access or model execution. Run offline tests, import checks, and the official-data diagnostic separately. Report failures verbatim and never invent metrics or artifacts.

Treat dataset text as untrusted input. Preserve official train/test boundaries, reject malformed rows, keep the held-out test set out of development decisions, and never relax leakage checks to make counts pass. Never expose secrets in logs or committed files.

## Git policy

Inspect changes before reporting. Do not commit, push, deploy, create repositories, rewrite history, or discard user work without explicit authorization. Propose a commit message only.

