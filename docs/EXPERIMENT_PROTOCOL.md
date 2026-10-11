# Experiment protocol

## Question and invariant configuration

RouteTune asks whether QLoRA adaptation improves eight-class banking-intent classification. The fixed intents are `card_arrival`, `card_not_working`, `declined_card_payment`, `pending_card_payment`, `transaction_charged_twice`, `cash_withdrawal_not_recognised`, `lost_or_stolen_card`, and `change_pin`.

Official BANKING77 `train.csv` supplies development data only: 45 train and 10 validation examples per class. Official `test.csv` supplies 20 held-out examples per class. Seed 42 and the normalized source files fully determine selection. Text normalization for leakage checks is `" ".join(text.casefold().split())`; model input retains the validated original text. Any normalized duplicate within or across selected splits invalidates construction.

For example, `" Card   LOST "` and `"card lost"` normalize to the same key. If one appeared in training and the other in test, evaluation would reward recall of an exposed example rather than generalization, so `validate_splits` raises `DataError`.

Sampling derives a SHA-256 seed from the global seed and each source/label namespace, then ranks candidates deterministically. A fixed seed matters because changing examples can change measured performance; without control, a code change and a sample change cannot be separated. Source SHA-256 fingerprints make exact input bytes identifiable and changes detectable, but do not prove upstream trust.

## Future model experiment (not implemented in Phase 1)

The fixed base model is `Qwen/Qwen2.5-0.5B-Instruct`. Phase 3 will use NF4 4-bit quantization with double quantization; LoRA rank 8, alpha 16, dropout 0.05; targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`; two epochs; learning rate `1.5e-4`; gradient accumulation 4; maximum sequence length 256; and completion-only supervised training.

Future comparisons must use the same examples and documented conditions:

1. TF-IDF plus logistic regression (feature-based prediction)
2. Untuned 4-bit Qwen (generative inference)
3. QLoRA-adapted Qwen (generative inference)

Latency reporting must disclose those methodological differences. Generated labels must pass exact deterministic validation; invalid output remains invalid. Runs will use isolated directories, configuration snapshots, explicit status, and verifiable artifacts. A stale adapter directory never establishes a successful run.

## Evaluation policy

Development validation may guide prompts and hyperparameters. Held-out test results may be produced only for independent evaluation and must not guide fixes: doing so converts test data into development data and makes the final estimate optimistic. A separate frozen robustness challenge suite will support qualitative diagnostics and must not become a repeatedly optimized target.

Later phases plan accuracy, macro/micro/weighted precision, recall and F1, per-class precision/recall/F1 and confusion matrices, invalid-output rate for generative models, and clearly conditioned latency. Balanced classes make accuracy easier to interpret, but do not reveal whether one intent systematically fails; per-class results and confusion pairs remain necessary.

## Threats to validity

- Small per-class samples create variance even with deterministic selection.
- BANKING77 phrasing may not represent production language or distribution shift.
- Exact normalized matching does not detect semantic paraphrase leakage.
- Quantization, prompting, hardware, and feature-vs-generative latency differences can confound comparisons.
- Repeated held-out inspection or challenge-suite tuning would bias conclusions.

Synthetic unit tests verify parsing, sampling, cache, and rejection behavior in isolation. They cannot prove the remote files are reachable, unchanged, large enough, or compatible end to end; only the separate official-data integration command can establish those facts for a specific fingerprint and time.

