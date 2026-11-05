import asyncio

import pytest
from test_scheduler import RecordingBackend, run_requests

from server.kv_budget import KVBudget
from server.scheduler import (
    BatchingScheduler,
    NaiveScheduler,
    QueueFull,
    RequestRejected,
    RequestTooLarge,
)

LARGE = 1_000_000


def batching(backend, capacity=LARGE, max_queue=LARGE, max_concurrent=4):
    return BatchingScheduler(backend, max_concurrent, KVBudget(capacity), max_queue)


def naive(backend, capacity=LARGE, max_queue=LARGE):
    return NaiveScheduler(backend, KVBudget(capacity), max_queue)


@pytest.mark.parametrize("make", [batching, naive])
def test_submit_rejects_when_queue_is_full(make):
    scheduler = make(RecordingBackend(tokens_per_sequence=1), max_queue=2)
    scheduler.submit([1], 1)
    scheduler.submit([2], 1)

    with pytest.raises(RequestRejected):
        scheduler.submit([3], 1)


@pytest.mark.parametrize("make", [batching, naive])
def test_request_that_can_never_fit_is_rejected_even_when_idle(make):
    backend = RecordingBackend(tokens_per_sequence=1)
    scheduler = make(backend, capacity=10)

    with pytest.raises(RequestRejected):
        scheduler.submit([1, 2], 9)

    assert backend.events == []


@pytest.mark.parametrize("make", [batching, naive])
def test_request_of_exactly_total_capacity_is_served(make):
    scheduler = make(RecordingBackend(tokens_per_sequence=1), capacity=10)

    [request] = run_requests(scheduler, ([1, 2], 8))

    assert request.tokens == [3]


def test_request_that_fits_is_admitted():
    backend = RecordingBackend(tokens_per_sequence=2)

    [request] = run_requests(batching(backend, capacity=10), ([1], 5))

    assert request.tokens == [1, 2]
    assert backend.events == [("start", 0), ("release", 0)]


def test_request_that_does_not_fit_yet_waits_for_budget():
    backend = RecordingBackend(tokens_per_sequence=3)

    first, second = run_requests(batching(backend, capacity=10), ([1], 9), ([2], 4))

    assert backend.peak_active == 1
    assert second.metrics.first_token >= first.metrics.finish
    assert second.tokens == [2, 3, 4]


def test_smaller_request_does_not_overtake_waiting_head():
    backend = RecordingBackend(tokens_per_sequence=2)

    run_requests(batching(backend, capacity=10), ([1], 4), ([1], 9), ([1], 2))

    assert backend.events == [
        ("start", 0), ("release", 0),
        ("start", 1), ("release", 1),
        ("start", 2), ("release", 2),
    ]


def test_budget_is_held_while_request_is_active():
    budget = KVBudget(20)
    seen = []

    class ObservingBackend(RecordingBackend):
        def step(self):
            seen.append(budget.available)
            return super().step()

    backend = ObservingBackend(tokens_per_sequence=1)

    run_requests(BatchingScheduler(backend, 4, budget, LARGE), ([1, 2], 3))

    assert seen == [15]


def test_full_reservation_is_released_after_finish():
    budget = KVBudget(10)
    backend = RecordingBackend(tokens_per_sequence=2)

    run_requests(BatchingScheduler(backend, 4, budget, LARGE), ([1], 2), ([2], 3))

    assert budget.available == 10


def test_full_reservation_is_released_when_request_ends_early():
    budget = KVBudget(10)
    backend = RecordingBackend(tokens_per_sequence=1)

    [request] = run_requests(BatchingScheduler(backend, 4, budget, LARGE), ([1], 9))

    assert request.tokens == [1]
    assert budget.available == 10


def test_rejected_request_leaves_budget_unchanged():
    budget = KVBudget(10)
    scheduler = BatchingScheduler(RecordingBackend(tokens_per_sequence=1), 4, budget, LARGE)

    with pytest.raises(RequestRejected):
        scheduler.submit([1], 10)

    assert budget.available == 10


def test_admitted_request_frees_its_queue_slot():
    async def run():
        scheduler = batching(RecordingBackend(tokens_per_sequence=1), max_queue=1)
        loop_task = asyncio.create_task(scheduler.run())
        first = scheduler.submit([1], 1)
        await asyncio.wait_for(first.done.wait(), timeout=5)
        second = scheduler.submit([2], 1)
        await asyncio.wait_for(second.done.wait(), timeout=5)
        loop_task.cancel()
        return second

    assert asyncio.run(run()).tokens == [2]


def test_rejection_types_distinguish_too_large_from_queue_full():
    scheduler = batching(RecordingBackend(tokens_per_sequence=1), capacity=10, max_queue=1)

    with pytest.raises(RequestTooLarge):
        scheduler.submit([1], 10)

    scheduler.submit([1], 1)
    with pytest.raises(QueueFull):
        scheduler.submit([2], 1)
