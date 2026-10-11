# RouteTune 2.0

RouteTune is an evidence-first experiment that will test whether QLoRA adaptation of `Qwen/Qwen2.5-0.5B-Instruct` improves eight-class BANKING77 intent classification against meaningful baselines.

## Current status

Phases 1 and 2 are **VERIFIED**. Phase 3 is **PREPARED / GPU EXECUTION BLOCKED**: its CPU contract, evidence ledger, genuine QLoRA path, and Colab notebook are implemented and 63 offline tests pass, but no real GPU training or held-out evaluation has been run. The local environment lacks the optional GPU packages and its 4 GiB GTX 1650 Ti is below the fixed-run 12 GiB gate. Phase 4 has not started.

## Structure

- `routetune/config.py`: fixed labels, seed, and split sizes
- `routetune/data.py`: ingestion, cache, sampling, validation, provenance, and CLI
- `routetune/evaluation.py`: strict label parsing, prediction records, metrics, and timing
- `routetune/baseline.py`: validation-only TF-IDF/logistic-regression runner and JSON artifacts
- `routetune/inference.py`: chat-template and completion-only Qwen adapter
- `routetune/challenge.py`: frozen handcrafted diagnostic cases
- `routetune/training.py`: masked QLoRA data, GPU gates, training, reload, and controlled evaluation
- `routetune/experiment.py`: atomic run manifests, artifact identity, and held-out gate
- `routetune/cli.py`: cohesive Phase 2 commands
- `notebooks/RouteTune_Colab.ipynb`: T4-or-better Phase 3 execution and artifact export
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

The local environment resolved NumPy 2.5.4 and SciPy 1.18.1. Optional GPU packages are isolated behind `.[gpu]`; they are deliberately absent from the Windows CPU environment. A visible GPU and driver alone do not establish that the exact four-bit stack works.

## Phase 3 Colab handoff

1. Create a ZIP of the current project worktree without `.venv`, `data`, or `artifacts`.
2. Open `notebooks/RouteTune_Colab.ipynb` in Google Colab and select a T4-or-better GPU runtime.
3. Run each cell in order, upload that ZIP, permit the Google Drive mount, and retain the printed run path.
4. The notebook installs the exact `.[gpu,test]` lock, checks installed APIs and CUDA, runs all CPU tests, downloads and fingerprints BANKING77, creates a fresh `PREPARED` run, then executes Gates B-D.
5. Return the exported run ZIP. Do not enable `RUN_FINAL_HELDOUT` until the manifest is `VERIFIED` and the Gate B-D evidence has been reviewed.

The current lock is `torch 2.14.1`, `transformers 5.19.0`, `trl 1.15.0`, `peft 0.21.2`, `accelerate 1.15.0`, `bitsandbytes 0.50.2`, and `datasets 4.7.0`. These are a reproducible target, not locally verified GPU evidence; the notebook checks the required installed APIs before model loading.

QLoRA keeps the quantized base weights frozen and trains small LoRA matrices, unlike full fine-tuning. NF4 represents normally distributed pretrained weights efficiently, while double quantization reduces quantization-metadata cost. Gradient accumulation produces an effective batch of 16 from four-example device batches, and checkpointing trades extra recomputation for lower activation memory. Completion-only labels mask prompts and padding with `-100`, so only the exact assistant label/control suffix contributes to loss. Lower loss is not classification accuracy: generated labels are separately parsed and scored. Saving is insufficient by itself, so adapter identity, hashes, reload, and deterministic reload outputs are verified before validation or held-out authorization. Test contamination can still occur without fitting on test rows if prompts, settings, or rerun decisions are changed after seeing held-out results; the ledger therefore makes Gate E single-use.

## Data attribution

The CSVs come from the official [PolyAI-LDN task-specific datasets repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data). A SHA-256 value identifies downloaded content; it does not independently establish that an upstream source is trustworthy.

