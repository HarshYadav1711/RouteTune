# RouteTune 2.0

RouteTune is an evidence-first experiment that will test whether QLoRA adaptation of `Qwen/Qwen2.5-0.5B-Instruct` improves eight-class BANKING77 intent classification against meaningful baselines.

## Current status

Phase 1 is **VERIFIED** on the available Python 3.13.3 environment. It implements the Python package, fixed experiment configuration, validated official CSV ingestion, atomic local caching, SHA-256 provenance, deterministic balanced splits, leakage checks, a data diagnostic, and CPU-only tests. Training, inference, evaluation metrics, classifiers, GPU integration, and application infrastructure are **NOT IMPLEMENTED**.

## Structure

- `routetune/config.py`: fixed labels, seed, and split sizes
- `routetune/data.py`: ingestion, cache, sampling, validation, provenance, and CLI
- `tests/test_data.py`: synthetic offline behavior tests
- `docs/EXPERIMENT_PROTOCOL.md`: scientific contract for all phases
- `docs/PHASES.md`: phase gates and evidence-backed status

## Setup (PowerShell)

Python 3.11 or newer is required. Confirm `python --version` selects the intended interpreter. Python 3.13.3 was available for the initial verification; 3.11 was not installed on that machine.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
```

These commands were executed successfully during Phase 1. The managed verification environment required `TEMP` and `TMP` to point to a writable project-local directory; ordinary PowerShell environments should not need that sandbox-specific override.

Run the offline suite and import check:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -c "import routetune; import routetune.data"
```

Inspect official data (downloads two CSVs into the ignored `data/cache` directory on first use):

```powershell
.\.venv\Scripts\python -m routetune.data --cache-dir data/cache
```

Require an already validated cache and make no network request:

```powershell
.\.venv\Scripts\python -m routetune.data --cache-dir data/cache --offline
```

The regular test suite uses synthetic files and performs no external requests. A successful official-data command is the separate integration check.

## Data attribution

The CSVs come from the official [PolyAI-LDN task-specific datasets repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data). A SHA-256 value identifies downloaded content; it does not independently establish that an upstream source is trustworthy.

