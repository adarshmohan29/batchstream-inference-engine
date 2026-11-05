from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from server.block_allocator import OutOfBlocks
from server.torch_model import ModelConfig, PagedKVPool, TorchModel, attend, paged_attention

MODEL_DIR = Path(__file__).parent.parent / "models" / "qwen2.5-0.5b-instruct"
BLOCK_SIZE = 16
LOGIT_TOLERANCE = 0.0

needs_model = pytest.mark.skipif(not MODEL_DIR.exists(), reason="model files not present")


@pytest.fixture(scope="module")
def model():
    return TorchModel.load(MODEL_DIR)


@pytest.fixture(scope="module")
def tokens():
    return AutoTokenizer.from_pretrained(MODEL_DIR)("word and number " * 60).input_ids


def difference(ours, theirs):
    return (ours.float() - theirs.float()).abs().max().item()


def scattered_keys_and_values(length, block_size, table):
    torch.manual_seed(0)
    keys = torch.randn(2, length, 4)
    values = torch.randn(2, length, 4)
    pool_keys = torch.full((2, 16, block_size, 4), float("nan"))
    pool_values = torch.full((2, 16, block_size, 4), float("nan"))
    for position in range(length):
        block = table[position // block_size]
        pool_keys[:, block, position % block_size] = keys[:, position]
        pool_values[:, block, position % block_size] = values[:, position]
    return keys, values, pool_keys, pool_values


@pytest.mark.parametrize("length", [1, 3, 4, 5, 8, 9, 15])
def test_paged_attention_matches_contiguous_attend_on_scattered_blocks(length):
    table = [9, 2, 14, 0][: -(-length // 4)]
    keys, values, pool_keys, pool_values = scattered_keys_and_values(length, 4, table)
    query = torch.randn(4, 4)

    paged = paged_attention(query, pool_keys, pool_values, table, length, 4, 2, 0.5)

    expected = attend(query[:, None], keys, values, None, 2, 0.5)[:, 0]
    assert torch.equal(paged, expected)


def test_paged_attention_never_reads_past_length():
    # Unused slots hold NaN, so reading one would poison the whole result.
    table = [3, 7]
    _, _, pool_keys, pool_values = scattered_keys_and_values(5, 4, table)

    output = paged_attention(torch.randn(4, 4), pool_keys, pool_values, table, 5, 4, 2, 0.5)

    assert not output.isnan().any()


@needs_model
@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33, 100])
def test_decode_logits_match_contiguous_at_block_boundaries(model, tokens, length):
    prompt, last = tokens[: length - 1], tokens[length - 1]
    contiguous = model.new_cache(capacity=length)
    pool = PagedKVPool(model.config, num_blocks=16, block_size=BLOCK_SIZE)
    paged = pool.new_cache(seq_id=0)
    if prompt:
        model.forward(prompt, contiguous)
        model.forward(prompt, paged)

    expected = model.forward([last], contiguous)
    actual = model.forward([last], paged)

    assert difference(actual, expected) <= LOGIT_TOLERANCE


@needs_model
def test_multi_token_prefill_across_blocks_matches_contiguous(model, tokens):
    prompt = tokens[:50]
    pool = PagedKVPool(model.config, num_blocks=8, block_size=BLOCK_SIZE)

    actual = model.forward(prompt, pool.new_cache(seq_id=0))

    expected = model.forward(prompt, model.new_cache(capacity=50))
    assert difference(actual, expected) <= LOGIT_TOLERANCE


@needs_model
def test_fragmented_blocks_give_identical_logits(model, tokens):
    pool = PagedKVPool(model.config, num_blocks=16, block_size=BLOCK_SIZE)
    other = pool.new_cache(seq_id=1)
    target = pool.new_cache(seq_id=0)
    contiguous = model.new_cache(capacity=49)
    for start in range(0, 48, BLOCK_SIZE):
        chunk = tokens[start : start + BLOCK_SIZE]
        model.forward(chunk, target)
        model.forward(tokens[100:101], other)
        model.forward(chunk, contiguous)
    table = pool.block_table(0)
    assert table != list(range(table[0], table[0] + len(table)))

    actual = model.forward([tokens[48]], target)

    assert difference(actual, model.forward([tokens[48]], contiguous)) <= LOGIT_TOLERANCE


@needs_model
def test_reused_blocks_do_not_leak_old_data(model, tokens):
    pool = PagedKVPool(model.config, num_blocks=2, block_size=BLOCK_SIZE)
    first = pool.new_cache(seq_id=0)
    model.forward(tokens[100:130], first)
    pool.release(first)

    actual = model.forward(tokens[:20], pool.new_cache(seq_id=1))

    expected = model.forward(tokens[:20], model.new_cache(capacity=20))
    assert difference(actual, expected) <= LOGIT_TOLERANCE


@needs_model
def test_running_out_of_blocks_raises_and_leaves_cache_unchanged(model, tokens):
    pool = PagedKVPool(model.config, num_blocks=1, block_size=BLOCK_SIZE)
    cache = pool.new_cache(seq_id=0)

    with pytest.raises(OutOfBlocks):
        model.forward(tokens[:BLOCK_SIZE + 1], cache)

    assert cache.length == 0


def test_release_returns_every_block():
    config = ModelConfig(1, 8, 2, 2, 10000.0, 1e-6)
    pool = PagedKVPool(config, num_blocks=4, block_size=4)
    cache = pool.new_cache(seq_id=0)
    cache.reserve(9)

    pool.release(cache)

    assert pool.free_blocks == 4


# Measured on real runs: 774 of 7.7M attention values differ, by at most 0.0078, and those
# differences reach the logits as at most 0.37 after 24 layers. Both limits sit just above that.
CPP_TOLERANCE = 0.01
CPP_MODEL_TOLERANCE = 0.5


@pytest.fixture(scope="module")
def cpp_attention():
    from server.paged_attention_cpp import paged_attention as compiled

    return compiled


def bf16_attention_inputs(length, block_size=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    num_blocks = -(-length // block_size) + 3
    table = torch.randperm(num_blocks, generator=generator)[: -(-length // block_size)].tolist()
    shape = (2, num_blocks, block_size, 8)
    keys = torch.full(shape, float("nan"), dtype=torch.bfloat16)
    values = torch.full(shape, float("nan"), dtype=torch.bfloat16)
    for position in range(length):
        block, offset = table[position // block_size], position % block_size
        keys[:, block, offset] = torch.randn(2, 8, generator=generator).to(torch.bfloat16)
        values[:, block, offset] = torch.randn(2, 8, generator=generator).to(torch.bfloat16)
    query = torch.randn(4, 8, generator=generator).to(torch.bfloat16)
    return query, keys, values, table


@pytest.mark.parametrize("length", [1, 3, 4, 5, 8, 9, 15, 16, 17, 100, 1000])
def test_cpp_matches_python_on_scattered_blocks(cpp_attention, length):
    query, keys, values, table = bf16_attention_inputs(length, seed=length)
    scaling = 8**-0.5

    expected = paged_attention(query, keys, values, table, length, 4, 2, scaling)
    actual = cpp_attention(query, keys, values, table, length, 4, 2, scaling)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert difference(actual, expected) <= CPP_TOLERANCE


def test_cpp_accepts_a_non_contiguous_query(cpp_attention):
    query, keys, values, table = bf16_attention_inputs(5)
    strided = torch.zeros(4, 3, 8, dtype=torch.bfloat16)
    strided[:, 1] = query

    actual = cpp_attention(strided[:, 1], keys, values, table, 5, 4, 2, 0.5)

    assert torch.equal(actual, cpp_attention(query, keys, values, table, 5, 4, 2, 0.5))


def test_cpp_never_reads_past_length(cpp_attention):
    query, keys, values, table = bf16_attention_inputs(5)

    assert not cpp_attention(query, keys, values, table, 5, 4, 2, 0.5).isnan().any()


@pytest.mark.parametrize(
    "change",
    [
        {"length": 0},
        {"length": -1},
        {"length": 13},
        {"table": []},
        {"table": [999, 0]},
        {"table": [-1, 0]},
        {"block_size": 0},
        {"block_size": 8},
        {"groups": 0},
        {"groups": 3},
        {"query": torch.zeros(4, 7, dtype=torch.bfloat16)},
        {"query": torch.zeros(4, 8, dtype=torch.float32)},
        {"query": torch.zeros(8, dtype=torch.bfloat16)},
        {"keys": torch.zeros(2, 3, 4, 8, dtype=torch.float32)},
        {"keys": torch.zeros(3, 8, dtype=torch.bfloat16)},
        {"values": torch.zeros(2, 2, 4, 8, dtype=torch.bfloat16)},
    ],
)
def test_cpp_rejects_bad_input_with_an_error_not_a_crash(cpp_attention, change):
    query, keys, values, table = bf16_attention_inputs(5)
    arguments = dict(
        query=query, keys=keys, values=values, table=table, length=5, block_size=4, groups=2
    )
    arguments.update(change)

    with pytest.raises(RuntimeError):
        cpp_attention(
            arguments["query"], arguments["keys"], arguments["values"], arguments["table"],
            arguments["length"], arguments["block_size"], arguments["groups"], 0.5,
        )


@needs_model
@pytest.mark.parametrize("length", [1, 15, 16, 17, 33, 100])
def test_cpp_model_logits_match_python_paged_attention(model, tokens, cpp_attention, length):
    prompt, last = tokens[: length - 1], tokens[length - 1]
    python_pool = PagedKVPool(model.config, 16, BLOCK_SIZE)
    cpp_pool = PagedKVPool(model.config, 16, BLOCK_SIZE, attention=cpp_attention)
    python_cache, cpp_cache = python_pool.new_cache(0), cpp_pool.new_cache(0)
    if prompt:
        model.forward(prompt, python_cache)
        model.forward(prompt, cpp_cache)

    actual = model.forward([last], cpp_cache)

    assert difference(actual, model.forward([last], python_cache)) <= CPP_MODEL_TOLERANCE


@needs_model
def test_cpp_greedy_tokens_match_python(model, tokens, cpp_attention):
    def generate(pool):
        cache = pool.new_cache(0)
        logits = model.forward(tokens[:20], cache)[-1]
        generated = []
        for _ in range(16):
            generated.append(logits.float().argmax().item())
            logits = model.forward([generated[-1]], cache)[-1]
        return generated

    python_tokens = generate(PagedKVPool(model.config, 16, BLOCK_SIZE))
    cpp_tokens = generate(PagedKVPool(model.config, 16, BLOCK_SIZE, attention=cpp_attention))

    assert cpp_tokens == python_tokens
