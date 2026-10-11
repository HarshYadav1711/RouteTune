# Experiment protocol

## Question and invariant configuration

RouteTune asks whether QLoRA adaptation improves eight-class banking-intent classification. The fixed intents are `card_arrival`, `card_not_working`, `declined_card_payment`, `pending_card_payment`, `transaction_charged_twice`, `cash_withdrawal_not_recognised`, `lost_or_stolen_card`, and `change_pin`.

Official BANKING77 `train.csv` supplies development data only: 45 train and 10 validation examples per class. Official `test.csv` supplies 20 held-out examples per class. Seed 42 and the normalized source files fully determine selection. Text normalization for leakage checks is `" ".join(text.casefold().split())`; model input retains the validated original text. Any normalized duplicate within or across selected splits invalidates construction.

For example, `" Card   LOST "` and `"card lost"` normalize to the same key. If one appeared in training and the other in test, evaluation would reward recall of an exposed example rather than generalization, so `validate_splits` raises `DataError`.

Sampling derives a SHA-256 seed from the global seed and each source/label namespace, then ranks candidates deterministically. A fixed seed matters because changing examples can change measured performance; without control, a code change and a sample change cannot be separated. Source SHA-256 fingerprints make exact input bytes identifiable and changes detectable, but do not prove upstream trust.

## Phase 3 model experiment (prepared; GPU verification pending)

The fixed base model is `Qwen/Qwen2.5-0.5B-Instruct`. Phase 3 will use NF4 4-bit quantization with double quantization; LoRA rank 8, alpha 16, dropout 0.05; targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`; two epochs; learning rate `1.5e-4`; gradient accumulation 4; maximum sequence length 256; and completion-only supervised training.

The prepared comparison uses the same examples and documented conditions:

1. TF-IDF plus logistic regression (feature-based prediction)
2. Untuned 4-bit Qwen (generative inference)
3. QLoRA-adapted Qwen (generative inference)

Latency reporting must disclose those methodological differences. Generated labels must pass exact deterministic validation; invalid output remains invalid. Runs will use isolated directories, configuration snapshots, explicit status, and verifiable artifacts. A stale adapter directory never establishes a successful run.

## Evaluation policy

Development validation may guide prompts and hyperparameters. Held-out test results may be produced only for independent evaluation and must not guide fixes: doing so converts test data into development data and makes the final estimate optimistic. A separate frozen robustness challenge suite will support qualitative diagnostics and must not become a repeatedly optimized target.

The shared evaluator records accuracy, macro precision/recall/F1, per-class precision/recall/F1, confusion matrices, invalid-output rate, and clearly conditioned latency. Balanced classes make accuracy easier to interpret, but do not reveal whether one intent systematically fails; per-class results and confusion pairs remain necessary.

## Phase 3 execution controls

Training examples are rendered with the same `build_messages` prompt used for inference. The tokenizer first renders that prompt with an assistant-generation prefix, then renders the same messages plus the exact target label. Prompt positions receive label `-100`; completion control/label tokens remain supervised; right-padding also receives `-100`. Any non-prefix template result, empty target, malformed decoded completion, or sequence over 256 tokens fails before training rather than truncating a target.

The GPU path resolves one Hugging Face model revision, loads NF4 four-bit weights with nested quantization and FP16 compute, freezes the base through PEFT k-bit preparation, and attaches LoRA only to the seven frozen target-module names. Trainable LoRA parameters and their gradients must be FP32 under FP16 AMP; compact dtype histograms and GPU BF16 capability are recorded before training. A diagnostic real batch must produce finite loss, non-zero finite LoRA gradients, no frozen-base gradients, and a real optimizer update; the diagnostic adapter snapshot is restored before the two-epoch run.

The untuned smoke generation proves that real GPU inference executed; strict label validity is recorded separately and does not gate training. Its raw completion, parsed label or null, validity, and timing remain evidence, while generation or infrastructure exceptions still block progress.

Each run owns an atomic JSON manifest and moves only through `PREPARED`, `RUNNING`, `TRAINED`, and `VERIFIED`, or into terminal `FAILED`/`INTERRUPTED`. Adapter configuration, weights, tokenizer files, a run identity record, and hashes are checked before reload. Gate E is claimed once and only after all prior evidence is true; a claimed or partial held-out run cannot be silently repeated.

## Threats to validity

- Small per-class samples create variance even with deterministic selection.
- BANKING77 phrasing may not represent production language or distribution shift.
- Exact normalized matching does not detect semantic paraphrase leakage.
- Quantization, prompting, hardware, and feature-vs-generative latency differences can confound comparisons.
- Repeated held-out inspection or challenge-suite tuning would bias conclusions.

Synthetic unit tests verify parsing, sampling, cache, and rejection behavior in isolation. They cannot prove the remote files are reachable, unchanged, large enough, or compatible end to end; only the separate official-data integration command can establish those facts for a specific fingerprint and time.

## Phase 2 evaluation contract

Every evaluated row retains a deterministic identifier, input, expected class, raw output, parsed class or null, strict validity, correctness, and optional per-example latency. A generation is valid only when trimming surrounding whitespace produces exactly one configured label. Casing changes, prefixes, explanations, punctuation, multiple labels, unknown labels, and empty output are invalid and are never repaired.

Accuracy is correct predictions divided by all predictions. For each official class, precision is `TP / (TP + FP)`, recall is `TP / support`, and F1 is their harmonic mean; a zero denominator produces zero. Macro values average the eight official classes equally. Invalid output is an `__invalid__` confusion-matrix column, counts as incorrect and as a false negative for its expected class, and remains in every total. It is not a ninth intent class.

The classical baseline is a scikit-learn pipeline with lowercase word unigrams/bigrams and logistic regression (`C=1.0`, `solver=lbfgs`, `max_iter=1000`, seed 42). The vectorizer and classifier fit only the 360 training rows. Validation has 80 rows; the official test split is neither fitted nor evaluated. Per-example CPU latency covers `Pipeline.predict([text])`, including TF-IDF transformation and classification. Fit time is separate, and these measurements are not comparable to future GPU generation without matched hardware and boundaries.

Qwen uses the tokenizer's chat template with separate system and user messages, greedy decoding, `do_sample=False`, at most 32 new tokens, EOS/padding IDs, and completion-only decoding. Tokenization and generation are timed separately; CUDA synchronization surrounds generation when available. The customer message is untrusted task data, and no expected label is passed into prompt construction.

The frozen robustness suite contains 20 handcrafted clear, paraphrased, boundary, prompt-injection, out-of-domain, and ambiguous cases. Ambiguous, boundary, and out-of-domain cases have null expectations and produce diagnostics rather than invented accuracy. Its SHA-256 fingerprint is `85ebd0e590b90c352370bb4d4772a3f886c4e4055ec0c2509c9d01f9a17049a5`. It has no normalized exact-text overlap with the selected 600 BANKING77 records and must not become training or repeated optimization data.

