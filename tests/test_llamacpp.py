import asyncio
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from server.backend.llamacpp import LlamaCppBackend
from server.kv_budget import KVBudget
from server.scheduler import BatchingScheduler, NaiveScheduler

MODEL = Path(__file__).parent.parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea:\n",
    "def fibonacci(n):\n",
    "List three primary colors:\n",
    "Explain gravity in one sentence.",
    "Translate 'good morning' to Spanish:",
    "2 + 2 =",
    "Name a large river:",
]
MAX_TOKENS = 24
LARGE = 1_000_000

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model file not present")


@contextmanager
def open_backend(n_ctx=4096, max_sequences=8, prefill_chunk=512, n_ubatch=None):
    backend = LlamaCppBackend(str(MODEL), n_ctx, max_sequences, prefill_chunk, n_ubatch)
    try:
        yield backend
    finally:
        backend.close()


def generate(backend, prompt_tokens, count):
    seq_id = backend.start_sequence(prompt_tokens)
    tokens = []
    while len(tokens) < count:
        output = backend.step().get(seq_id)
        if output is not None:
            tokens.append(output.token)
    backend.release_sequence(seq_id)
    return tokens


def serve(scheduler, specs, stagger_seconds=0):
    async def run():
        loop_task = asyncio.create_task(scheduler.run())
        requests = []
        for prompt_tokens, max_tokens in specs:
            requests.append(scheduler.submit(prompt_tokens, max_tokens))
            await asyncio.sleep(stagger_seconds)
        await asyncio.wait_for(asyncio.gather(*(r.done.wait() for r in requests)), timeout=120)
        loop_task.cancel()
        return requests

    return asyncio.run(run())


def specs_for(backend):
    return [(backend.tokenize(prompt), MAX_TOKENS) for prompt in PROMPTS]


def naive(backend):
    return NaiveScheduler(backend, KVBudget(LARGE), LARGE)


def batching(backend, max_concurrent=8):
    return BatchingScheduler(backend, max_concurrent, KVBudget(LARGE), LARGE)


def test_missing_model_file_raises():
    with pytest.raises(ValueError):
        LlamaCppBackend("models/does-not-exist.gguf", 256, 1, 8)


def test_text_survives_tokenize_and_detokenize():
    with open_backend() as backend:
        text = "Hello, world! 你好 🙂"
        pieces = [backend.detokenize([token]) for token in backend.tokenize(text)]

    assert b"".join(pieces).decode() == text


def test_step_returns_one_token_per_ready_sequence():
    with open_backend() as backend:
        first = backend.start_sequence(backend.tokenize(PROMPTS[0]))
        second = backend.start_sequence(backend.tokenize(PROMPTS[1]))

        assert backend.step().keys() == {first, second}
        assert backend.step().keys() == {first, second}


def test_generation_is_repeatable():
    with open_backend() as backend:
        tokens = backend.tokenize(PROMPTS[0])

        assert generate(backend, tokens, 8) == generate(backend, tokens, 8)


def test_long_prompt_does_not_stall_active_sequence():
    with open_backend(prefill_chunk=8) as backend:
        active = backend.start_sequence(backend.tokenize(PROMPTS[0]))
        backend.step()
        long_prompt = backend.tokenize("word " * 40)
        assert len(long_prompt) > 32
        waiting = backend.start_sequence(long_prompt)

        outputs = [backend.step() for _ in range(3)]

        assert all(active in step_output for step_output in outputs)
        assert all(waiting not in step_output for step_output in outputs)
        while waiting not in (step_output := backend.step()):
            assert active in step_output


def test_chunked_prefill_gives_same_first_token_as_full_prefill():
    prompt = "word " * 40
    with open_backend(n_ubatch=1) as whole:
        expected = generate(whole, whole.tokenize(prompt), 4)
    with open_backend(prefill_chunk=8, n_ubatch=1) as chunked:
        assert generate(chunked, chunked.tokenize(prompt), 4) == expected


def test_released_sequence_id_is_reused_cleanly():
    with open_backend(max_sequences=1, n_ubatch=1) as backend:
        tokens = backend.tokenize(PROMPTS[1])
        expected = generate(backend, tokens, 6)
        interrupted = backend.start_sequence(backend.tokenize(PROMPTS[0]))
        backend.step()
        backend.release_sequence(interrupted)

        assert generate(backend, tokens, 6) == expected


def test_release_during_prefill_is_allowed():
    with open_backend(max_sequences=1, prefill_chunk=4) as backend:
        seq_id = backend.start_sequence(backend.tokenize("word " * 40))
        backend.step()
        backend.release_sequence(seq_id)

        assert backend.step() == {}


def test_releasing_unknown_sequence_raises():
    with open_backend() as backend:
        with pytest.raises(KeyError):
            backend.release_sequence(99)


def test_end_of_generation_finishes_sequence(monkeypatch):
    with open_backend() as backend:
        seq_id = backend.start_sequence(backend.tokenize(PROMPTS[0]))
        monkeypatch.setattr("server.backend.llamacpp.lc.llama_vocab_is_eog", lambda *_: True)

        assert backend.step()[seq_id].finished is True
        assert backend.step() == {}


def test_context_smaller_than_batch_is_refused():
    # llama.cpp aborts the whole process on this configuration, so it must fail at startup.
    with pytest.raises(ValueError):
        LlamaCppBackend(str(MODEL), 32, 8, 64)


def test_kv_cache_overflow_raises():
    # llama.cpp rounds n_ctx up to a multiple of 256, so 256 is the smallest exact capacity.
    with open_backend(n_ctx=256, max_sequences=1, prefill_chunk=8) as backend:
        backend.start_sequence(backend.tokenize("word " * 400))

        with pytest.raises(RuntimeError):
            for _ in range(40):
                backend.step()


def test_single_request_matches_between_naive_and_batched():
    with open_backend() as backend:
        spec = [(backend.tokenize(PROMPTS[0]), MAX_TOKENS)]
        [one_at_a_time] = serve(naive(backend), spec)
        [batched] = serve(batching(backend), spec)

    assert one_at_a_time.tokens == batched.tokens


def test_batched_output_is_identical_with_single_token_ubatches():
    with open_backend(n_ubatch=1) as backend:
        specs = specs_for(backend)
        one_at_a_time = serve(naive(backend), specs)
        batched = serve(batching(backend), specs)

    assert [r.tokens for r in batched] == [r.tokens for r in one_at_a_time]


def test_mid_flight_joins_and_id_reuse_keep_output_identical():
    with open_backend(max_sequences=3, prefill_chunk=4, n_ubatch=1) as backend:
        specs = specs_for(backend)
        one_at_a_time = serve(naive(backend), specs)
        batched = serve(batching(backend, max_concurrent=3), specs, stagger_seconds=0.02)

    assert [r.tokens for r in batched] == [r.tokens for r in one_at_a_time]


def test_budget_of_exactly_n_ctx_never_overflows_the_kv_cache():
    with open_backend(n_ctx=256, prefill_chunk=64) as backend:
        specs = specs_for(backend)
        capacity = sum(len(tokens) + max_tokens for tokens, max_tokens in specs)
        assert capacity <= 256
        scheduler = BatchingScheduler(backend, 8, KVBudget(256), LARGE)

        requests = serve(scheduler, specs)

    assert all(len(r.tokens) <= MAX_TOKENS for r in requests)


def test_batched_is_faster_under_concurrent_load():
    with open_backend() as backend:
        specs = specs_for(backend)
        serve(naive(backend), specs[:1])

        start = time.monotonic()
        serve(naive(backend), specs)
        naive_seconds = time.monotonic() - start

        start = time.monotonic()
        serve(batching(backend), specs)
        batched_seconds = time.monotonic() - start

    assert batched_seconds < naive_seconds


def test_real_kv_overflow_fails_request_and_backend_keeps_working():
    with open_backend(n_ctx=256, max_sequences=1, prefill_chunk=8) as backend:
        tokens = backend.tokenize(PROMPTS[0])
        scheduler = naive(backend)
        expected = serve(scheduler, [(tokens, 6)])[0].tokens

        oversized, following = serve(
            batching(backend, max_concurrent=1),
            [(backend.tokenize("word " * 400), 6), (tokens, 6)],
        )

    assert oversized.failed
    assert not following.failed
    assert following.tokens == expected
