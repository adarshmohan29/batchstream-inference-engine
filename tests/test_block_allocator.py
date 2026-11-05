import random

import pytest

from server.block_allocator import BlockAllocator, OutOfBlocks


def test_allocate_covers_requested_tokens_in_whole_blocks():
    allocator = BlockAllocator(num_blocks=4, block_size=16)

    allocator.allocate(seq_id=0, total_tokens=17)

    assert len(allocator.block_table(0)) == 2
    assert allocator.free_blocks == 2


def test_allocate_exactly_full_block_uses_one_block():
    allocator = BlockAllocator(num_blocks=4, block_size=16)

    allocator.allocate(seq_id=0, total_tokens=16)

    assert len(allocator.block_table(0)) == 1


def test_sequence_grows_only_when_it_crosses_a_block_boundary():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    allocator.allocate(seq_id=0, total_tokens=1)
    first = allocator.block_table(0)[0]

    allocator.allocate(seq_id=0, total_tokens=16)
    assert allocator.block_table(0) == [first]

    allocator.allocate(seq_id=0, total_tokens=17)
    assert allocator.block_table(0)[0] == first
    assert len(allocator.block_table(0)) == 2


def test_locate_maps_position_to_block_and_offset():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    allocator.allocate(seq_id=0, total_tokens=40)
    table = allocator.block_table(0)

    assert allocator.locate(0, 0) == (table[0], 0)
    assert allocator.locate(0, 15) == (table[0], 15)
    assert allocator.locate(0, 16) == (table[1], 0)
    assert allocator.locate(0, 39) == (table[2], 7)


def test_sequences_never_share_a_block():
    allocator = BlockAllocator(num_blocks=4, block_size=16)

    allocator.allocate(seq_id=0, total_tokens=32)
    allocator.allocate(seq_id=1, total_tokens=32)

    assert not set(allocator.block_table(0)) & set(allocator.block_table(1))


def test_free_returns_every_block():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    allocator.allocate(seq_id=0, total_tokens=48)

    allocator.free(0)

    assert allocator.free_blocks == 4


def test_freed_sequence_has_no_table():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    allocator.allocate(seq_id=0, total_tokens=1)

    allocator.free(0)

    with pytest.raises(KeyError):
        allocator.block_table(0)


def test_freeing_unknown_sequence_raises():
    with pytest.raises(KeyError):
        BlockAllocator(num_blocks=4, block_size=16).free(7)


def test_out_of_blocks_raises():
    allocator = BlockAllocator(num_blocks=2, block_size=16)

    with pytest.raises(OutOfBlocks):
        allocator.allocate(seq_id=0, total_tokens=33)


def test_failed_allocation_changes_nothing():
    allocator = BlockAllocator(num_blocks=3, block_size=16)
    allocator.allocate(seq_id=0, total_tokens=16)
    table = list(allocator.block_table(0))

    with pytest.raises(OutOfBlocks):
        allocator.allocate(seq_id=0, total_tokens=64)

    assert allocator.block_table(0) == table
    assert allocator.free_blocks == 2


def test_blocks_are_reused_after_free():
    allocator = BlockAllocator(num_blocks=2, block_size=16)
    allocator.allocate(seq_id=0, total_tokens=32)
    allocator.free(0)

    allocator.allocate(seq_id=1, total_tokens=32)

    assert sorted(allocator.block_table(1)) == [0, 1]


def test_no_leaked_or_shared_blocks_after_many_cycles():
    rng = random.Random(0)
    allocator = BlockAllocator(num_blocks=64, block_size=8)
    tokens = {}
    next_id = 0

    for _ in range(20_000):
        if tokens and (rng.random() < 0.4 or allocator.free_blocks == 0):
            seq_id = rng.choice(list(tokens))
            del tokens[seq_id]
            allocator.free(seq_id)
        elif tokens and rng.random() < 0.5:
            seq_id = rng.choice(list(tokens))
            try_tokens = tokens[seq_id] + rng.randint(1, 20)
            try:
                allocator.allocate(seq_id, try_tokens)
                tokens[seq_id] = try_tokens
            except OutOfBlocks:
                pass
        else:
            count = rng.randint(1, 40)
            try:
                allocator.allocate(next_id, count)
                tokens[next_id] = count
                next_id += 1
            except OutOfBlocks:
                pass

        used = [block for seq_id in tokens for block in allocator.block_table(seq_id)]
        assert len(used) == len(set(used))
        assert len(used) + allocator.free_blocks == 64

    for seq_id in list(tokens):
        allocator.free(seq_id)
    assert allocator.free_blocks == 64
