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
