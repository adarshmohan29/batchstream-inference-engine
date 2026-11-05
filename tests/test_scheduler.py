import asyncio

from server.backend.fake import FakeBackend
from server.kv_budget import KVBudget
from server.scheduler import BatchingScheduler, NaiveScheduler


LARGE = 1_000_000


class RecordingBackend(FakeBackend):
    def __init__(self, tokens_per_sequence, step_seconds=0):
        super().__init__(tokens_per_sequence, step_seconds)
        self.events = []
        self.active = 0
        self.peak_active = 0

    def start_sequence(self, prompt_tokens):
        seq_id = super().start_sequence(prompt_tokens)
        self.events.append(("start", seq_id))
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        return seq_id

    def release_sequence(self, seq_id):
        super().release_sequence(seq_id)
        self.events.append(("release", seq_id))
        self.active -= 1


def naive_scheduler(backend):
    return NaiveScheduler(backend, KVBudget(LARGE), LARGE)


def batching_scheduler(backend, max_concurrent):
    return BatchingScheduler(backend, max_concurrent, KVBudget(LARGE), LARGE)


def run_requests(scheduler, *specs):
    async def run():
        loop_task = asyncio.create_task(scheduler.run())
        requests = [scheduler.submit(prompt, max_tokens) for prompt, max_tokens in specs]
        await asyncio.wait_for(asyncio.gather(*(r.done.wait() for r in requests)), timeout=5)
        loop_task.cancel()
        return requests

    return asyncio.run(run())


def serve(backend, *specs):
    return run_requests(naive_scheduler(backend), *specs)


def serve_batched(backend, max_concurrent, *specs):
    return run_requests(batching_scheduler(backend, max_concurrent), *specs)


def total_time(requests):
    return max(r.metrics.finish for r in requests) - min(r.metrics.arrival for r in requests)


def test_request_receives_backend_tokens_in_order():
    [request] = serve(FakeBackend(tokens_per_sequence=3, step_seconds=0), ([1, 2], 10))

    assert request.tokens == [3, 4, 5]


def test_each_sequence_is_released_before_the_next_starts():
    backend = RecordingBackend(tokens_per_sequence=2)

    serve(backend, ([1], 10), ([2], 10), ([3], 10))

    assert backend.events == [
        ("start", 0), ("release", 0),
        ("start", 1), ("release", 1),
        ("start", 2), ("release", 2),
    ]


def test_short_request_behind_long_one_waits_for_it():
    backend = FakeBackend(tokens_per_sequence=5, step_seconds=0)

    long_request, short_request = serve(backend, ([1], 5), ([2], 1))

    assert short_request.metrics.first_token >= long_request.metrics.finish
    assert short_request.metrics.finish >= long_request.metrics.finish


def test_max_tokens_stops_request_before_backend_finishes():
    backend = RecordingBackend(tokens_per_sequence=5)

    [request] = serve(backend, ([1], 2))

    assert request.tokens == [1, 2]
    assert backend.events == [("start", 0), ("release", 0)]


def test_backend_finish_stops_request_before_max_tokens():
    [request] = serve(FakeBackend(tokens_per_sequence=2, step_seconds=0), ([1], 10))

    assert request.tokens == [1, 2]


def test_request_submitted_after_queue_drains_is_served():
    async def run():
        scheduler = naive_scheduler(FakeBackend(tokens_per_sequence=1, step_seconds=0))
        loop_task = asyncio.create_task(scheduler.run())
        first = scheduler.submit([1], 10)
        await asyncio.wait_for(first.done.wait(), timeout=5)
        second = scheduler.submit([2], 10)
        await asyncio.wait_for(second.done.wait(), timeout=5)
        loop_task.cancel()
        return second

    assert asyncio.run(run()).tokens == [2]


def test_metrics_include_time_spent_waiting_in_queue():
    step_seconds = 0.01
    backend = FakeBackend(tokens_per_sequence=3, step_seconds=step_seconds)

    first, second = serve(backend, ([1], 3), ([2], 3))

    assert first.metrics.end_to_end >= 3 * step_seconds
    assert second.metrics.ttft >= 3 * step_seconds
    assert second.metrics.end_to_end >= second.metrics.ttft


def test_batched_short_request_behind_long_one_finishes_first():
    backend = FakeBackend(tokens_per_sequence=5, step_seconds=0.001)

    long_request, short_request = serve_batched(backend, 2, ([1], 5), ([2], 1))

    assert short_request.metrics.finish < long_request.metrics.finish


def test_batched_total_time_is_less_than_naive():
    specs = [([i], 5) for i in range(4)]

    naive = serve(FakeBackend(tokens_per_sequence=5, step_seconds=0.01), *specs)
    batched = serve_batched(FakeBackend(tokens_per_sequence=5, step_seconds=0.01), 4, *specs)

    assert total_time(batched) < total_time(naive)


def test_active_sequences_never_exceed_max_concurrent():
    backend = RecordingBackend(tokens_per_sequence=3)

    serve_batched(backend, 2, *[([i], 3) for i in range(6)])

    assert backend.peak_active == 2


def test_waiting_request_joins_when_a_slot_frees():
    backend = FakeBackend(tokens_per_sequence=5, step_seconds=0.001)

    long_request, _, waiting = serve_batched(backend, 2, ([1], 5), ([2], 1), ([3], 1))

    assert waiting.metrics.finish < long_request.metrics.finish


def test_requests_are_admitted_in_fifo_order():
    backend = RecordingBackend(tokens_per_sequence=1)

    requests = serve_batched(backend, 1, ([10], 1), ([20], 1), ([30], 1))

    assert [r.tokens for r in requests] == [[10], [20], [30]]
    assert backend.events == [
        ("start", 0), ("release", 0),
        ("start", 1), ("release", 1),
        ("start", 2), ("release", 2),
    ]


def test_batched_tokens_match_naive_tokens():
    specs = [([1, 2], 4), ([5], 2), ([7, 8, 9], 3)]

    naive = serve(FakeBackend(tokens_per_sequence=4, step_seconds=0), *specs)
    batched = serve_batched(FakeBackend(tokens_per_sequence=4, step_seconds=0), 2, *specs)

    assert [r.tokens for r in batched] == [r.tokens for r in naive]


def test_max_tokens_stops_one_request_without_affecting_others():
    backend = RecordingBackend(tokens_per_sequence=5)

    stopped, other = serve_batched(backend, 2, ([1], 2), ([2], 5))

    assert stopped.tokens == [1, 2]
    assert other.tokens == [2, 3, 4, 5, 6]
    assert backend.events == [("start", 0), ("start", 1), ("release", 0), ("release", 1)]


def test_every_sequence_is_released_exactly_once():
    backend = RecordingBackend(tokens_per_sequence=2)

    serve_batched(backend, 2, *[([i], 2) for i in range(4)])

    released = [seq_id for event, seq_id in backend.events if event == "release"]
    assert sorted(released) == [0, 1, 2, 3]


def test_batched_request_submitted_after_idle_is_served():
    async def run():
        scheduler = batching_scheduler(FakeBackend(tokens_per_sequence=1, step_seconds=0), 2)
        loop_task = asyncio.create_task(scheduler.run())
        first = scheduler.submit([1], 10)
        await asyncio.wait_for(first.done.wait(), timeout=5)
        second = scheduler.submit([2], 10)
        await asyncio.wait_for(second.done.wait(), timeout=5)
        loop_task.cancel()
        return second

    assert asyncio.run(run()).tokens == [2]
