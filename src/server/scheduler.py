import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from server.backend.base import Backend, StepOutput
from server.kv_budget import KVBudget
from server.metrics import RequestMetrics


class RequestRejected(Exception):
    pass


@dataclass
class Request:
    prompt_tokens: list[int]
    max_tokens: int
    metrics: RequestMetrics
    tokens: list[int] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def kv_tokens(self) -> int:
        return len(self.prompt_tokens) + self.max_tokens


class _Scheduler:
    def __init__(self, backend: Backend, budget: KVBudget, max_queue: int):
        self._backend = backend
        self._budget = budget
        self._max_queue = max_queue
        self._waiting: deque[Request] = deque()
        self._arrived = asyncio.Event()

    def submit(self, prompt_tokens: list[int], max_tokens: int) -> Request:
        request = Request(prompt_tokens, max_tokens, RequestMetrics(arrival=time.monotonic()))
        if not self._budget.fits_ever(request.kv_tokens):
            raise RequestRejected("request can never fit in the KV budget")
        if len(self._waiting) >= self._max_queue:
            raise RequestRejected("waiting queue is full")
        self._waiting.append(request)
        self._arrived.set()
        return request

    async def _wait_for_request(self) -> None:
        while not self._waiting:
            self._arrived.clear()
            await self._arrived.wait()

    def _record_token(self, request: Request, output: StepOutput) -> bool:
        """Record one token and return whether the request is finished."""
        request.tokens.append(output.token)
        request.metrics.record_token(time.monotonic())
        return output.finished or len(request.tokens) >= request.max_tokens

    def _finish(self, seq_id: int, request: Request) -> None:
        self._backend.release_sequence(seq_id)
        request.metrics.record_finish(time.monotonic())
        request.done.set()


class NaiveScheduler(_Scheduler):
    async def run(self) -> None:
        while True:
            await self._wait_for_request()
            await self._serve(self._waiting.popleft())

    async def _serve(self, request: Request) -> None:
        # Backend calls block on real models, so they must not run on the event loop.
        seq_id = await asyncio.to_thread(self._backend.start_sequence, request.prompt_tokens)
        finished = False
        while not finished:
            output = (await asyncio.to_thread(self._backend.step))[seq_id]
            finished = self._record_token(request, output)
        self._finish(seq_id, request)


class BatchingScheduler(_Scheduler):
    def __init__(self, backend: Backend, max_concurrent: int, budget: KVBudget, max_queue: int):
        super().__init__(backend, budget, max_queue)
        self._max_concurrent = max_concurrent
        self._active: dict[int, Request] = {}

    async def run(self) -> None:
        while True:
            if not self._active:
                await self._wait_for_request()
            await self._admit()
            await self._step()

    async def _admit(self) -> None:
        # Only the head is tried, so smaller requests cannot starve a large one by overtaking it.
        while (
            self._waiting
            and len(self._active) < self._max_concurrent
            and self._budget.try_reserve(self._waiting[0].kv_tokens)
        ):
            request = self._waiting.popleft()
            seq_id = await asyncio.to_thread(self._backend.start_sequence, request.prompt_tokens)
            self._active[seq_id] = request

    async def _step(self) -> None:
        outputs = await asyncio.to_thread(self._backend.step)
        for seq_id, output in outputs.items():
            request = self._active[seq_id]
            if self._record_token(request, output):
                del self._active[seq_id]
                self._budget.release(request.kv_tokens)
                self._finish(seq_id, request)
