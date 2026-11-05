class KVBudget:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self.available = capacity

    def fits_ever(self, tokens: int) -> bool:
        return tokens <= self._capacity

    def try_reserve(self, tokens: int) -> bool:
        if tokens > self.available:
            return False
        self.available -= tokens
        return True

    def release(self, tokens: int) -> None:
        self.available += tokens


class BlockBudget(KVBudget):
    """Reserves a request's worst case in whole blocks, so a running request cannot run out."""

    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks)
        self._block_size = block_size

    def fits_ever(self, tokens: int) -> bool:
        return super().fits_ever(self._blocks(tokens))

    def try_reserve(self, tokens: int) -> bool:
        return super().try_reserve(self._blocks(tokens))

    def release(self, tokens: int) -> None:
        super().release(self._blocks(tokens))

    def _blocks(self, tokens: int) -> int:
        return -(-tokens // self._block_size)
