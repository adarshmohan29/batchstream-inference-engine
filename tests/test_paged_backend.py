import asyncio
from contextlib import contextmanager
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from server.backend.paged import PagedBackend
from server.kv_budget import BlockBudget, KVBudget
from server.scheduler import BatchingScheduler, NaiveScheduler

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
GGUF = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "List three primary colors:\n",
    "2 + 2 =",
    "Name a large river:",
    "word " * 40,
]
MAX_TOKENS = 12
LARGE = 1_000_000
BLOCK_SIZE = 16

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR.exists() and GGUF.exists()), reason="model files not present"
)


@contextmanager
def open_backend(num_blocks=64, prefill_chunk=64):
    backend = PagedBackend(MODEL_DIR, GGUF, num_blocks, BLOCK_SIZE, prefill_chunk)
    try:
        yield backend
    finally:
        backend.close()


def serve(scheduler, specs, stagger_seconds=0):
    async def run():
        loop_task = asyncio.create_task(scheduler.run())
        requests = []
        for prompt_tokens, max_tokens in specs:
            requests.append(scheduler.submit(prompt_tokens, max_tokens))
            await asyncio.sleep(stagger_seconds)
        await asyncio.wait_for(asyncio.gather(*(r.done.wait() for r in requests)), timeout=300)
        loop_task.cancel()
        return requests

    return asyncio.run(run())


def specs_for(backend):
    return [(backend.tokenize(prompt), MAX_TOKENS) for prompt in PROMPTS]


def test_tokens_match_the_reference_tokenizer():
    reference = AutoTokenizer.from_pretrained(MODEL_DIR)

    with open_backend(num_blocks=1) as backend:
        for prompt in PROMPTS + ["你好，世界 🙂"]:
            assert backend.tokenize(prompt) == reference(prompt).input_ids


def test_text_survives_tokenize_and_detokenize():
    with open_backend(num_blocks=1) as backend:
        text = "Hello 你好 🙂"
        pieces = [backend.detokenize([token]) for token in backend.tokenize(text)]

    assert b"".join(pieces).decode() == text


def test_step_returns_one_token_per_ready_sequence():
    with open_backend() as backend:
        first = backend.start_sequence(backend.tokenize(PROMPTS[0]))
        second = backend.start_sequence(backend.tokenize(PROMPTS[3]))

        assert backend.step().keys() == {first, second}
        assert backend.step().keys() == {first, second}


def test_long_prompt_does_not_stall_active_sequence():
    with open_backend(prefill_chunk=8) as backend:
        active = backend.start_sequence(backend.tokenize(PROMPTS[0]))
        backend.step()
        waiting = backend.start_sequence(backend.tokenize("word " * 40))

        outputs = [backend.step() for _ in range(3)]

        assert all(active in output for output in outputs)
        assert all(waiting not in output for output in outputs)


def test_blocks_are_allocated_on_demand_not_up_front():
    with open_backend(num_blocks=64) as backend:
        backend.start_sequence(backend.tokenize(PROMPTS[0]))
        assert backend.blocks_in_use == 0

        backend.step()
        assert backend.blocks_in_use == 1


def test_release_returns_all_blocks_and_allows_prefill_release():
    with open_backend() as backend:
        seq_id = backend.start_sequence(backend.tokenize(PROMPTS[0]))
        never_stepped = backend.start_sequence(backend.tokenize(PROMPTS[3]))
        backend.release_sequence(never_stepped)
        backend.step()

        backend.release_sequence(seq_id)

        assert backend.blocks_in_use == 0
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


def test_batched_greedy_output_equals_one_at_a_time():
    with open_backend() as backend:
        specs = specs_for(backend)
        one_at_a_time = serve(NaiveScheduler(backend, KVBudget(LARGE), LARGE), specs)
        batched = serve(BatchingScheduler(backend, 6, KVBudget(LARGE), LARGE), specs)

    assert [r.tokens for r in batched] == [r.tokens for r in one_at_a_time]


def test_mid_flight_joins_keep_output_identical():
    with open_backend(prefill_chunk=8) as backend:
        specs = specs_for(backend)
        one_at_a_time = serve(NaiveScheduler(backend, KVBudget(LARGE), LARGE), specs)
        batched = serve(
            BatchingScheduler(backend, 3, KVBudget(LARGE), LARGE), specs, stagger_seconds=0.05
        )

    assert [r.tokens for r in batched] == [r.tokens for r in one_at_a_time]


def test_scheduler_runs_end_to_end_with_block_budget():
    budget = BlockBudget(num_blocks=8, block_size=BLOCK_SIZE)
    with open_backend(num_blocks=8) as backend:
        specs = specs_for(backend)
        requests = serve(BatchingScheduler(backend, 3, budget, LARGE), specs)

        assert backend.blocks_in_use == 0

    assert all(len(r.tokens) == MAX_TOKENS and not r.failed for r in requests)
    assert budget.available == 8


def test_running_out_of_blocks_fails_request_and_backend_recovers():
    with open_backend(num_blocks=2) as backend:
        small = backend.tokenize(PROMPTS[0])
        expected = serve(NaiveScheduler(backend, KVBudget(LARGE), LARGE), [(small, 4)])[0].tokens

        oversized, following = serve(
            BatchingScheduler(backend, 1, KVBudget(LARGE), LARGE),
            [(backend.tokenize("word " * 60), 4), (small, 4)],
        )

    assert oversized.failed
    assert not following.failed
    assert following.tokens == expected
