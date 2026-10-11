# RouteTune 2.0

RouteTune is an evidence-first experiment that will test whether QLoRA adaptation of `Qwen/Qwen2.5-0.5B-Instruct` improves eight-class BANKING77 intent classification against meaningful baselines.

## Current status

Phases 1 and 2 are **VERIFIED** on the available Python 3.13.3 environment. The project now has validated data ingestion, deterministic splits, a shared strict evaluator, a real TF-IDF/logistic-regression validation baseline, an injected and CPU-tested Qwen inference contract, and a frozen 20-case robustness suite. QLoRA training, adapter loading, official held-out model evaluation, and production infrastructure are **NOT IMPLEMENTED**. Real quantized Qwen execution is **NOT VERIFIED**.

## Structure

- `routetune/config.py`: fixed labels, seed, and split sizes
- `routetune/data.py`: ingestion, cache, sampling, validation, provenance, and CLI
- `routetune/evaluation.py`: strict label parsing, prediction records, metrics, and timing
- `routetune/baseline.py`: validation-only TF-IDF/logistic-regression runner and JSON artifacts
- `routetune/inference.py`: chat-template and completion-only Qwen adapter
- `routetune/challenge.py`: frozen handcrafted diagnostic cases
- `routetune/cli.py`: cohesive Phase 2 commands
- `tests/`: synthetic CPU tests plus a cache-backed, network-free integration check
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

The regular suite uses synthetic files plus one network-free integration check against the validated local cache; that check skips when the cache is absent. Tests never make external requests. The official-data command remains the separate live integration check.

## Phase 2 commands

Each baseline output directory must be new; artifacts are never overwritten.

```powershell
.\.venv\Scripts\routetune baseline --cache-dir data/cache --offline --output-dir artifacts/phase2/my-validation-run
.\.venv\Scripts\routetune show-metrics artifacts/phase2/my-validation-run/result.json
.\.venv\Scripts\routetune show-errors artifacts/phase2/my-validation-run/result.json --limit 5
.\.venv\Scripts\routetune prompt-diagnostics
.\.venv\Scripts\routetune qwen-smoke --message "My replacement card has not arrived yet."
```

The verified baseline fitted 360 training examples and evaluated 80 validation examples with scikit-learn 1.9.1. It produced accuracy `0.8875`, macro F1 `0.8868696362117415`, 71 correct predictions, and 100% vocabulary-valid output. This is a validation result, not an official held-out test result. The complete 80-record artifact is at `artifacts/phase2/tfidf-validation-20261011-final/result.json` and is intentionally ignored by Git.

The local environment resolved NumPy 2.5.4 and SciPy 1.18.1. Optional Qwen packages are separated behind `.[qwen]`; that extra was not installed or verified during Phase 2. The visible GTX 1650 Ti and driver alone do not establish that the exact 4-bit stack works.

## Data attribution

The CSVs come from the official [PolyAI-LDN task-specific datasets repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data). A SHA-256 value identifies downloaded content; it does not independently establish that an upstream source is trustworthy.

