import asyncio
import codecs
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from server.backend.base import Backend
from server.backend.fake import FakeBackend
from server.kv_budget import KVBudget
from server.scheduler import BatchingScheduler, QueueFull, Request, RequestTooLarge


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(ge=1)


class _CancelOnExitResponse(StreamingResponse):
    def __init__(self, content: AsyncIterator[str], on_exit: Callable[[], None]):
        super().__init__(content, media_type="text/event-stream")
        self._on_exit = on_exit

    async def __call__(self, scope, receive, send) -> None:
        # Cleanup lives here because a disconnect during send never resumes the generator.
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._on_exit()


async def _events(backend: Backend, request: Request) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while (token := await request.stream.get()) is not None:
        text = decoder.decode(backend.detokenize([token]))
        if text:
            yield f"data: {json.dumps({'text': text})}\n\n"
    yield "data: [DONE]\n\n"


def create_app(scheduler, backend: Backend) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        loop_task = asyncio.create_task(scheduler.run())
        yield
        loop_task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.post("/generate")
    async def generate(body: GenerateBody) -> StreamingResponse:
        try:
            request = scheduler.submit(backend.tokenize(body.prompt), body.max_tokens)
        except RequestTooLarge as rejection:
            raise HTTPException(status_code=413, detail=str(rejection))
        except QueueFull as rejection:
            raise HTTPException(status_code=429, detail=str(rejection))
        return _CancelOnExitResponse(_events(backend, request), lambda: scheduler.cancel(request))

    return app


_backend = FakeBackend(tokens_per_sequence=20, step_seconds=0.05)
app = create_app(BatchingScheduler(_backend, 4, KVBudget(4096), 32), _backend)
