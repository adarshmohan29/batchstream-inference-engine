import time
from dataclasses import dataclass
from itertools import count

from server.backend.base import Backend, StepOutput


@dataclass
class _Sequence:
    next_token: int
    remaining: int


class FakeBackend(Backend):
    def __init__(self, tokens_per_sequence: int, step_seconds: float):
        self._tokens_per_sequence = tokens_per_sequence
        self._step_seconds = step_seconds
        self._sequences: dict[int, _Sequence] = {}
        self._ids = count()

    def start_sequence(self, prompt_tokens: list[int]) -> int:
        seq_id = next(self._ids)
        self._sequences[seq_id] = _Sequence(sum(prompt_tokens), self._tokens_per_sequence)
        return seq_id

    def step(self) -> dict[int, StepOutput]:
        time.sleep(self._step_seconds)
        return {
            seq_id: self._advance(seq)
            for seq_id, seq in self._sequences.items()
            if seq.remaining > 0
        }

    def release_sequence(self, seq_id: int) -> None:
        del self._sequences[seq_id]

    @staticmethod
    def _advance(seq: _Sequence) -> StepOutput:
        token = seq.next_token
        seq.next_token += 1
        seq.remaining -= 1
        return StepOutput(token, finished=seq.remaining == 0)
