from routetune.config import MAX_NEW_TOKENS, TARGET_LABELS
from routetune.inference import QwenAdapter, build_messages


class FakeIds(list):
    @property
    def shape(self):
        return (1, len(self[0]))


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def __init__(self, decoded="card_arrival"):
        self.decoded = decoded
        self.template_call = None
        self.decoded_tokens = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = (messages, kwargs)
        return {"input_ids": FakeIds([[10, 11, 12]]), "attention_mask": [[1, 1, 1]]}

    def decode(self, tokens, **kwargs):
        self.decoded_tokens = (tokens, kwargs)
        return self.decoded


class FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[10, 11, 12, 99]]


def test_prompt_contains_all_labels_and_customer_text_without_answer_metadata():
    customer_text = "Please help with this customer message."
    messages = build_messages(customer_text)
    assert messages[1] == {"role": "user", "content": customer_text}
    assert all(label in messages[0]["content"] for label in TARGET_LABELS)
    assert "expected" not in messages[0]["content"].casefold()
    assert "ground truth" not in messages[0]["content"].casefold()


def test_adapter_uses_chat_template_slices_prompt_and_greedy_generation():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    times = iter((0.0, 0.2, 1.0, 1.5))
    sync_calls = []
    result = QwenAdapter(
        tokenizer,
        model,
        synchronize=lambda: sync_calls.append("sync"),
        clock=lambda: next(times),
    ).classify("Where is my card?")

    messages, template_kwargs = tokenizer.template_call
    assert messages == build_messages("Where is my card?")
    assert template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert tokenizer.decoded_tokens == ([99], {"skip_special_tokens": True})
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == MAX_NEW_TOKENS
    assert model.generate_kwargs["pad_token_id"] == tokenizer.eos_token_id
    assert model.generate_kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs
    assert result.raw_output == "card_arrival"
    assert result.predicted_class == "card_arrival"
    assert result.output_valid is True
    assert result.tokenization_seconds == 0.2
    assert result.generation_seconds == 0.5
    assert sync_calls == ["sync", "sync"]


def test_invalid_mocked_generation_is_preserved_not_repaired():
    tokenizer = FakeTokenizer("Label: card_arrival")
    result = QwenAdapter(tokenizer, FakeModel()).classify("Where is my card?")
    assert result.raw_output == "Label: card_arrival"
    assert result.predicted_class is None
    assert result.output_valid is False

