import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from server.backend.base import Backend, StepOutput
from server.kv_budget import KVBudget
from server.metrics import RequestMetrics

logger = logging.getLogger(__name__)


class RequestRejected(Exception):
    pass


class RequestTooLarge(RequestRejected):
    pass


class QueueFull(RequestRejected):
    pass


@dataclass(eq=False)
class Request:
    prompt_tokens: list[int]
    max_tokens: int
    metrics: RequestMetrics
    tokens: list[int] = field(default_factory=list)
    stream: asyncio.Queue[int | None] = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    failed: bool = False

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
            raise RequestTooLarge("request can never fit in the KV budget")
        if len(self._waiting) >= self._max_queue:
            raise QueueFull("waiting queue is full")
        self._waiting.append(request)
        self._arrived.set()
        return request

    def cancel(self, request: Request) -> None:
        if request in self._waiting:
            self._waiting.remove(request)
            self._end(request)
        else:
            # step() may be using the sequence in its worker thread, so the loop releases it later.
            request.cancelled = True

    async def _wait_for_request(self) -> None:
        while not self._waiting:
            self._arrived.clear()
            await self._arrived.wait()

    def _record_token(self, request: Request, output: StepOutput) -> bool:
        """Record one token and return whether the request is finished."""
        request.tokens.append(output.token)
        request.stream.put_nowait(output.token)
        request.metrics.record_token(time.monotonic())
        return output.finished or len(request.tokens) >= request.max_tokens

    def _finish(self, seq_id: int, request: Request) -> None:
        self._backend.release_sequence(seq_id)
        self._end(request)

    def _end(self, request: Request) -> None:
        request.metrics.record_finish(time.monotonic())
        request.stream.put_nowait(None)
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
        while not finished and not request.cancelled:
            try:
                outputs = await asyncio.to_thread(self._backend.step)
            except RuntimeError:
                logger.exception("backend step failed")
                request.failed = True
                break
            if seq_id in outputs:
                finished = self._record_token(request, outputs[seq_id])
        self._finish(seq_id, request)


class BatchingScheduler(_Scheduler):
    def __init__(self, backend: Backend, max_concurrent: int, budget: KVBudget, max_queue: int):
        super().__init__(backend, budget, max_queue)
        self._max_concurrent = max_concurrent
        self._active: dict[int, Request] = {}

    async def run(self) -> None:
        while True:
            self._remove_cancelled()
            if not self._active:
                await self._wait_for_request()
            await self._admit()
            await self._step()

    def _remove_cancelled(self) -> None:
        for seq_id, request in list(self._active.items()):
            if request.cancelled:
                self._complete(seq_id, request)

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
        try:
            outputs = await asyncio.to_thread(self._backend.step)
        except RuntimeError:
            logger.exception("backend step failed")
            self._fail_active()
            return
        for seq_id, output in outputs.items():
            request = self._active[seq_id]
            if self._record_token(request, output):
                self._complete(seq_id, request)

    def _fail_active(self) -> None:
        # One decode covers every active sequence, so none of them can be trusted to continue.
        for seq_id, request in list(self._active.items()):
            request.failed = True
            self._complete(seq_id, request)

    def _complete(self, seq_id: int, request: Request) -> None:
        del self._active[seq_id]
        self._budget.release(request.kv_tokens)
        self._finish(seq_id, request)
