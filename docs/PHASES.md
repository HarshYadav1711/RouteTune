# Development phases

Status vocabulary: **NOT IMPLEMENTED**, **IMPLEMENTED, UNVERIFIED**, **VERIFIED**, **FAILED**, **BLOCKED**. Gates require direct evidence; later work does not begin early.

## 1. Foundation and Data Integrity

Status: **VERIFIED** on 2026-10-11 using the available Python 3.13.3 interpreter (Python 3.11 was not installed).

Gate: package imports; all offline tests pass; the official CSV integration diagnostic reports 360/80/160 balanced rows, zero selected duplicates/leakage, provenance, and valid fingerprints; exclusions and scope are inspected.

Evidence: an isolated editable install with declared test dependencies succeeded; `python -m pytest --basetemp=.pytest-tmp-final` reported `15 passed in 0.40s`; compile and import checks passed; the online official-data diagnostic downloaded and validated both CSVs; and an offline cache-only rerun reproduced 360/80/160 rows with per-class counts 45/10/20 and zero selected duplicates or cross-split leakage. Source SHA-256 values were `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` (train) and `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` (test). The selected ordered dataset fingerprint was `a4fc6b6b55100d7996170686e5c1493f4d890fcd9b6d21d7ae35a4ec60dfa129`.

## 2. Evaluation Engine and Baselines

Status: **VERIFIED** on 2026-10-11 for CPU evaluation and inference contracts. Real quantized Qwen execution is **NOT VERIFIED**.

Gate: frozen evaluation contract, strict output parser, shared examples, TF-IDF/logistic-regression and untuned-Qwen baselines, metrics and artifact integrity, with CPU/GPU evidence correctly separated.

Evidence: 48 offline/network-free tests passed, including the retained 15 Phase 1 tests and a validated-cache integration check. The locked selection fingerprint remains `a4fc6b6b55100d7996170686e5c1493f4d890fcd9b6d21d7ae35a4ec60dfa129`. Scikit-learn 1.9.1 fitted 360 training rows and evaluated only 80 validation rows: accuracy `0.8875`, macro F1 `0.8868696362117415`, 71/80 correct, validity rate `1.0`, and zero invalid predictions. Two executions produced identical 80 predicted labels. The complete final artifact is `artifacts/phase2/tfidf-validation-20261011-final/result.json`; it records `official_test_evaluated: false` and is excluded from Git.

The 20-case challenge fingerprint is `85ebd0e590b90c352370bb4d4772a3f886c4e4055ec0c2509c9d01f9a17049a5`, with no normalized overlap against selected BANKING77 records. Prompt construction, chat-template use, completion slicing, greedy settings, invalid-output preservation, and mocked CUDA synchronization are CPU-tested. A GTX 1650 Ti is visible, but optional Qwen dependencies were deliberately not installed; no model weights were downloaded.

## 3. QLoRA Training and Controlled GPU Experiment

Status: **NOT IMPLEMENTED**.

Gate: fixed training contract executes on real compatible hardware; isolated run state and artifacts prove success or failure; adapter identity and configuration are verified.

## 4. Model Diagnostics and Portfolio Release

Status: **NOT IMPLEMENTED**.

Gate: per-class and confusion analysis, frozen robustness diagnostics, reproducible release instructions, limitations, and no fabricated claims or unsafe banking actions.

