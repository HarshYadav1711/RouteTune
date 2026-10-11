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


@dataclass(frozen=True)
class ExperimentConfig:
    labels: tuple[str, ...] = TARGET_LABELS
    seed: int = 42
    train_per_class: int = 45
    validation_per_class: int = 10
    test_per_class: int = 20


DEFAULT_CONFIG = ExperimentConfig()

