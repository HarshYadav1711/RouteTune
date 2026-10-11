"""Fixed experiment configuration."""

from dataclasses import dataclass


TARGET_LABELS = (
    "card_arrival",
    "card_not_working",
    "declined_card_payment",
    "pending_card_payment",
    "transaction_charged_twice",
    "cash_withdrawal_not_recognised",
    "lost_or_stolen_card",
    "change_pin",
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
EXPECTED_SELECTION_FINGERPRINT = (
    "a4fc6b6b55100d7996170686e5c1493f4d890fcd9b6d21d7ae35a4ec60dfa129"
)
EXPECTED_TRAIN_SHA256 = (
    "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
)
EXPECTED_TEST_SHA256 = (
    "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
)


@dataclass(frozen=True)
class ExperimentConfig:
    labels: tuple[str, ...] = TARGET_LABELS
    seed: int = 42
    train_per_class: int = 45
    validation_per_class: int = 10
    test_per_class: int = 20


DEFAULT_CONFIG = ExperimentConfig()

