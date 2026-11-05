class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self._block_size = block_size
        self._free = list(range(num_blocks))
        self._tables: dict[int, list[int]] = {}

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def allocate(self, seq_id: int, total_tokens: int) -> None:
        """Grow a sequence's block table until it covers total_tokens."""
        table = self._tables.get(seq_id, [])
        needed = -(-total_tokens // self._block_size) - len(table)
        if needed > len(self._free):
            raise OutOfBlocks(f"need {needed} blocks, {len(self._free)} free")
        for _ in range(needed):
            table.append(self._free.pop())
        self._tables[seq_id] = table

    def block_table(self, seq_id: int) -> list[int]:
        return self._tables[seq_id]

    def locate(self, seq_id: int, position: int) -> tuple[int, int]:
        block_index, offset = divmod(position, self._block_size)
        return self._tables[seq_id][block_index], offset

    def free(self, seq_id: int) -> None:
        self._free.extend(self._tables.pop(seq_id))
