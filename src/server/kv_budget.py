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
