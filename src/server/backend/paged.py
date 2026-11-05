from collections import deque
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import llama_cpp as lc

from server.backend.base import Backend, StepOutput
from server.backend.llamacpp import LlamaVocab
from server.torch_model import PagedKVCache, PagedKVPool, TorchModel


@dataclass
class _Sequence:
    cache: PagedKVCache
    pending: deque[int]
    next_token: int | None = None
    finished: bool = False


class PagedBackend(Backend):
    def __init__(
        self, model_dir: Path, gguf_path: Path, num_blocks: int, block_size: int, prefill_chunk: int
    ):
        self._model = TorchModel.load(model_dir)
        self._pool = PagedKVPool(self._model.config, num_blocks, block_size)
        self._num_blocks = num_blocks
        self._prefill_chunk = prefill_chunk
        # The GGUF holds the same vocabulary, so both backends tokenize identically.
        vocab_params = lc.llama_model_default_params()
        vocab_params.vocab_only = True
        self._vocab_model = lc.llama_model_load_from_file(str(gguf_path).encode(), vocab_params)
        self._vocab = LlamaVocab(lc.llama_model_get_vocab(self._vocab_model))
        self._ids = count()
        self._sequences: dict[int, _Sequence] = {}

    @property
    def blocks_in_use(self) -> int:
        return self._num_blocks - self._pool.free_blocks

    def tokenize(self, text: str) -> list[int]:
        return self._vocab.tokenize(text)

    def detokenize(self, tokens: list[int]) -> bytes:
        return self._vocab.detokenize(tokens)

    def start_sequence(self, prompt_tokens: list[int]) -> int:
        seq_id = next(self._ids)
        cache = self._pool.new_cache(seq_id)
        cache.reserve(0)
        self._sequences[seq_id] = _Sequence(cache, deque(prompt_tokens))
        return seq_id

    def step(self) -> dict[int, StepOutput]:
        batch = self._generating_inputs() + self._prefill_inputs()
        if not batch:
            return {}
        logits = self._model.forward_batch([(tokens, self._sequences[s].cache) for s, tokens in batch])
        outputs = {}
        for row, (seq_id, _) in enumerate(batch):
            sequence = self._sequences[seq_id]
            if not sequence.pending:
                outputs[seq_id] = self._next_output(sequence, logits[row])
        return outputs

    def release_sequence(self, seq_id: int) -> None:
        self._pool.release(self._sequences.pop(seq_id).cache)

    def close(self) -> None:
        lc.llama_model_free(self._vocab_model)

    def _generating_inputs(self) -> list[tuple[int, list[int]]]:
        return [
            (seq_id, [sequence.next_token])
            for seq_id, sequence in self._sequences.items()
            if not sequence.pending and not sequence.finished
        ]

    def _prefill_inputs(self) -> list[tuple[int, list[int]]]:
        inputs = []
        remaining = self._prefill_chunk
        for seq_id, sequence in self._sequences.items():
            take = min(remaining, len(sequence.pending))
            if take:
                inputs.append((seq_id, [sequence.pending.popleft() for _ in range(take)]))
                remaining -= take
        return inputs

    def _next_output(self, sequence: _Sequence, logits) -> StepOutput:
        token = logits.float().argmax().item()
        sequence.next_token = token
        sequence.finished = self._vocab.is_end_of_generation(token)
        return StepOutput(token, sequence.finished)
