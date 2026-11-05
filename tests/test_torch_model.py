from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from server.torch_model import ContiguousKVCache, TorchModel

MODEL_DIR = Path(__file__).parent.parent / "models" / "qwen2.5-0.5b-instruct"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "你好，世界 🙂",
    "word " * 200,
]
GREEDY_TOKENS = 32
LOGIT_TOLERANCE = 0.0

needs_model = pytest.mark.skipif(not MODEL_DIR.exists(), reason="model files not present")


@pytest.fixture(scope="module")
def model():
    return TorchModel.load(MODEL_DIR)


@pytest.fixture(scope="module")
def reference():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def reference_logits(reference, tokens):
    with torch.inference_mode():
        return reference(input_ids=torch.tensor([tokens]), use_cache=False).logits[0]


def max_logit_difference(ours, theirs):
    return (ours.float() - theirs.float()).abs().max().item()


def small_cache(capacity=3):
    return ContiguousKVCache(
        num_layers=2, num_kv_heads=2, head_dim=4, capacity=capacity, dtype=torch.float32
    )


def test_cache_appends_contiguously_per_layer():
    cache = small_cache()
    first = torch.ones(2, 1, 4)
    second = torch.full((2, 1, 4), 2.0)

    cache.append(0, first, first)
    cache.advance(1)
    keys, values = cache.append(0, second, second * 3)

    assert keys.shape == (2, 2, 4)
    assert torch.equal(keys[:, 0], first[:, 0])
    assert torch.equal(keys[:, 1], second[:, 0])
    assert torch.equal(values[:, 1], second[:, 0] * 3)


def test_cache_layers_are_independent():
    cache = small_cache()

    cache.append(0, torch.ones(2, 1, 4), torch.ones(2, 1, 4))
    keys, _ = cache.append(1, torch.zeros(2, 1, 4), torch.zeros(2, 1, 4))

    assert torch.equal(keys, torch.zeros(2, 1, 4))


def test_cache_length_only_changes_on_advance():
    cache = small_cache()
    cache.append(0, torch.ones(2, 2, 4), torch.ones(2, 2, 4))

    assert cache.length == 0

    cache.advance(2)

    assert cache.length == 2


def test_cache_refuses_more_tokens_than_capacity():
    cache = small_cache(capacity=3)
    cache.advance(2)

    cache.check_room(1)
    with pytest.raises(ValueError):
        cache.check_room(2)


@needs_model
def test_loaded_weights_equal_reference_weights(model, reference):
    state = reference.state_dict()

    for name, weight in model.weights.items():
        assert torch.equal(weight, state[name]), name


@needs_model
def test_overflowing_forward_raises_and_leaves_cache_unchanged(model):
    cache = model.new_cache(capacity=2)

    with pytest.raises(ValueError):
        model.forward([1, 2, 3], cache)

    assert cache.length == 0


@needs_model
@pytest.mark.parametrize("prompt", PROMPTS)
def test_prefill_logits_match_reference(model, reference, tokenizer, prompt):
    tokens = tokenizer(prompt).input_ids

    ours = model.forward(tokens, model.new_cache(capacity=len(tokens)))

    theirs = reference_logits(reference, tokens)
    assert ours.shape == theirs.shape
    assert max_logit_difference(ours, theirs) <= LOGIT_TOLERANCE


@needs_model
def test_cached_decode_logits_match_uncached_reference(model, reference, tokenizer):
    tokens = tokenizer(PROMPTS[0]).input_ids
    cache = model.new_cache(capacity=len(tokens) + 8)
    model.forward(tokens, cache)

    for token in [42, 7, 1000, 99, 5, 256, 11, 3]:
        ours = model.forward([token], cache)[-1]
        tokens = tokens + [token]
        theirs = reference_logits(reference, tokens)[-1]
        assert max_logit_difference(ours, theirs) <= LOGIT_TOLERANCE


@needs_model
def test_prefill_in_chunks_matches_prefill_in_one_call(model, tokenizer):
    tokens = tokenizer(PROMPTS[3]).input_ids
    whole = model.forward(tokens, model.new_cache(capacity=len(tokens)))

    cache = model.new_cache(capacity=len(tokens))
    chunks = [
        model.forward(tokens[start : start + 64], cache) for start in range(0, len(tokens), 64)
    ]

    assert cache.length == len(tokens)
    assert max_logit_difference(torch.cat(chunks), whole) <= LOGIT_TOLERANCE


@needs_model
@pytest.mark.parametrize("prompt", PROMPTS[:3])
def test_greedy_tokens_are_identical_to_reference(model, reference, tokenizer, prompt):
    prompt_tokens = tokenizer(prompt).input_ids

    expected = list(prompt_tokens)
    for _ in range(GREEDY_TOKENS):
        expected.append(reference_logits(reference, expected)[-1].float().argmax().item())

    cache = model.new_cache(capacity=len(prompt_tokens) + GREEDY_TOKENS)
    next_token = model.forward(prompt_tokens, cache)[-1].float().argmax().item()
    generated = list(prompt_tokens)
    for _ in range(GREEDY_TOKENS - 1):
        generated.append(next_token)
        next_token = model.forward([next_token], cache)[-1].float().argmax().item()
    generated.append(next_token)

    assert generated == expected
