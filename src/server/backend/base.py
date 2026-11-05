from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StepOutput:
    token: int
    finished: bool


class Backend(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Convert text to prompt tokens."""

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> bytes:
        """Return raw bytes, because one character can span several tokens."""

    @abstractmethod
    def start_sequence(self, prompt_tokens: list[int]) -> int:
        """Start a sequence from prompt tokens and return its id."""

    @abstractmethod
    def step(self) -> dict[int, StepOutput]:
        """Generate one token for each unfinished sequence, keyed by sequence id."""

    @abstractmethod
    def release_sequence(self, seq_id: int) -> None:
        """Free a sequence, finished or not."""
