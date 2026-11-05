import ctypes
from collections import deque
from dataclasses import dataclass

import llama_cpp as lc

from server.backend.base import Backend, StepOutput

_MAX_PIECE_BYTES = 256

lc.llama_backend_init()


class LlamaVocab:
    def __init__(self, vocab):
        self._vocab = vocab

    def tokenize(self, text: str) -> list[int]:
        encoded = text.encode()
        tokens = (lc.llama_token * (len(encoded) + 2))()
        # parse_special is off so that user text cannot forge control tokens.
        count = lc.llama_tokenize(self._vocab, encoded, len(encoded), tokens, len(tokens), True, False)
        return list(tokens[:count])

    def detokenize(self, tokens: list[int]) -> bytes:
        return b"".join(self._piece(token) for token in tokens)

    def is_end_of_generation(self, token: int) -> bool:
        return lc.llama_vocab_is_eog(self._vocab, token)

    def _piece(self, token: int) -> bytes:
        buffer = ctypes.create_string_buffer(_MAX_PIECE_BYTES)
        count = lc.llama_token_to_piece(self._vocab, token, buffer, _MAX_PIECE_BYTES, 0, False)
        return buffer.raw[:count]


@dataclass
class _Sequence:
    pending: deque[int]
    position: int = 0
    next_token: int | None = None
    finished: bool = False


class LlamaCppBackend(Backend):
    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        max_sequences: int,
        prefill_chunk: int,
        n_ubatch: int | None = None,
    ):
        n_batch = max_sequences + prefill_chunk
        if n_ctx < n_batch:
            # llama_decode aborts the process instead of returning an error in this case.
            raise ValueError(f"n_ctx {n_ctx} is smaller than the batch size {n_batch}")
        model_params = lc.llama_model_default_params()
        model_params.n_gpu_layers = -1
        self._model = lc.llama_model_load_from_file(model_path.encode(), model_params)
        if self._model is None:
            raise ValueError(f"could not load model: {model_path}")
        context_params = lc.llama_context_default_params()
        context_params.n_ctx = n_ctx
        context_params.n_batch = n_batch
        context_params.n_ubatch = n_ubatch or n_batch
        context_params.n_seq_max = max_sequences
        # One shared pool, so KVBudget capacity maps directly to n_ctx instead of n_ctx per sequence.
        context_params.kv_unified = True
        self._context = lc.llama_init_from_model(self._model, context_params)
        self._vocab = LlamaVocab(lc.llama_model_get_vocab(self._model))
        self._memory = lc.llama_get_memory(self._context)
        self._batch = lc.llama_batch_init(n_batch, 0, 1)
        self._sampler = lc.llama_sampler_init_greedy()
        self._prefill_chunk = prefill_chunk
        self._free_ids = list(range(max_sequences))
        self._sequences: dict[int, _Sequence] = {}

    def tokenize(self, text: str) -> list[int]:
        return self._vocab.tokenize(text)

    def detokenize(self, tokens: list[int]) -> bytes:
        return self._vocab.detokenize(tokens)

    def start_sequence(self, prompt_tokens: list[int]) -> int:
        seq_id = self._free_ids.pop()
        self._sequences[seq_id] = _Sequence(pending=deque(prompt_tokens))
        return seq_id

    def step(self) -> dict[int, StepOutput]:
        generating = self._add_generating_tokens()
        prefilled = self._add_prefill_tokens()
        outputs = generating | prefilled
        if self._batch.n_tokens == 0:
            return {}
        return_code = lc.llama_decode(self._context, self._batch)
        self._batch.n_tokens = 0
        if return_code != 0:
            raise RuntimeError(f"llama_decode returned {return_code}")
        return {seq_id: self._sample(seq_id, index) for seq_id, index in outputs.items()}

    def release_sequence(self, seq_id: int) -> None:
        del self._sequences[seq_id]
        lc.llama_memory_seq_rm(self._memory, seq_id, -1, -1)
        self._free_ids.append(seq_id)

    def close(self) -> None:
        lc.llama_sampler_free(self._sampler)
        lc.llama_batch_free(self._batch)
        lc.llama_free(self._context)
        lc.llama_model_free(self._model)

    def _add_generating_tokens(self) -> dict[int, int]:
        outputs = {}
        for seq_id, sequence in self._sequences.items():
            if not sequence.pending and not sequence.finished:
                outputs[seq_id] = self._add_token(seq_id, sequence, sequence.next_token, True)
        return outputs

    def _add_prefill_tokens(self) -> dict[int, int]:
        outputs = {}
        remaining = self._prefill_chunk
        for seq_id, sequence in self._sequences.items():
            while sequence.pending and remaining > 0:
                token = sequence.pending.popleft()
                last = not sequence.pending
                index = self._add_token(seq_id, sequence, token, last)
                remaining -= 1
                if last:
                    outputs[seq_id] = index
        return outputs

    def _add_token(self, seq_id: int, sequence: _Sequence, token: int, wants_logits: bool) -> int:
        index = self._batch.n_tokens
        self._batch.token[index] = token
        self._batch.pos[index] = sequence.position
        self._batch.n_seq_id[index] = 1
        self._batch.seq_id[index][0] = seq_id
        self._batch.logits[index] = wants_logits
        self._batch.n_tokens = index + 1
        sequence.position += 1
        return index

    def _sample(self, seq_id: int, index: int) -> StepOutput:
        token = lc.llama_sampler_sample(self._sampler, self._context, index)
        sequence = self._sequences[seq_id]
        sequence.next_token = token
        sequence.finished = self._vocab.is_end_of_generation(token)
        return StepOutput(token, sequence.finished)
