import http.client
import json
import socket
import threading
import time
from contextlib import contextmanager

import pytest
import uvicorn

from server.api import create_app
from server.backend.fake import FakeBackend
from server.kv_budget import KVBudget
from server.scheduler import BatchingScheduler

LARGE = 1_000_000
TIMEOUT = 5


@contextmanager
def running_server(scheduler, backend):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(
        create_app(scheduler, backend), log_level="warning", timeout_graceful_shutdown=1
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + TIMEOUT
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.005)
    assert server.started
    try:
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(TIMEOUT)
        sock.close()


def post(port, prompt, max_tokens):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT)
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens})
    connection.request("POST", "/generate", body, {"Content-Type": "application/json"})
    return connection, connection.getresponse()


def read_events(response):
    return [
        line.decode().removeprefix("data: ").strip()
        for line in response.read().splitlines()
        if line.startswith(b"data: ")
    ]


def texts(events):
    return [json.loads(event)["text"] for event in events[:-1]]


def wait_until(condition):
    deadline = time.monotonic() + TIMEOUT
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    return condition()


def make_scheduler(backend, capacity=LARGE, max_queue=LARGE, max_concurrent=4):
    return BatchingScheduler(backend, max_concurrent, KVBudget(capacity), max_queue)


def test_generate_streams_tokens_in_order_then_end_marker():
    backend = FakeBackend(tokens_per_sequence=3, step_seconds=0)

    with running_server(make_scheduler(backend), backend) as port:
        connection, response = post(port, "a", 10)
        events = read_events(response)
        connection.close()

    assert response.status == 200
    assert response.getheader("content-type").startswith("text/event-stream")
    expected = [backend.detokenize([t + backend.tokenize("a")[0]]).decode() for t in range(3)]
    assert texts(events) == expected
    assert events[-1] == "[DONE]"


def test_character_split_across_tokens_arrives_intact():
    class SplitCharBackend(FakeBackend):
        def tokenize(self, text):
            return [0]

        def detokenize(self, tokens):
            return "é".encode()[tokens[0]:tokens[0] + 1]

    backend = SplitCharBackend(tokens_per_sequence=2, step_seconds=0)

    with running_server(make_scheduler(backend), backend) as port:
        connection, response = post(port, "a", 10)
        events = read_events(response)
        connection.close()

    assert texts(events) == ["é"]


def test_too_large_request_returns_413():
    backend = FakeBackend(tokens_per_sequence=1, step_seconds=0)

    with running_server(make_scheduler(backend, capacity=5), backend) as port:
        connection, response = post(port, "a", 100)
        response.read()
        connection.close()

    assert response.status == 413


def test_full_queue_returns_429():
    backend = FakeBackend(tokens_per_sequence=100_000, step_seconds=0.001)
    scheduler = make_scheduler(backend, max_queue=1, max_concurrent=1)

    with running_server(scheduler, backend) as port:
        running_connection, running = post(port, "a", 100_000)
        running.readline()
        queued_connection, queued = post(port, "b", 100_000)
        rejected_connection, rejected = post(port, "c", 100_000)
        rejected.read()
        for connection, response in [(running_connection, running), (queued_connection, queued)]:
            response.close()
            connection.close()
        rejected_connection.close()

    assert rejected.status == 429


@pytest.mark.parametrize("prompt, max_tokens", [("a", 0), ("a", -1), ("", 5)])
def test_invalid_input_returns_422(prompt, max_tokens):
    backend = FakeBackend(tokens_per_sequence=1, step_seconds=0)

    with running_server(make_scheduler(backend), backend) as port:
        connection, response = post(port, prompt, max_tokens)
        response.read()
        connection.close()

    assert response.status == 422


def test_client_disconnect_releases_sequence_and_full_budget():
    budget = KVBudget(LARGE)
    backend = FakeBackend(tokens_per_sequence=1_000_000, step_seconds=0.001)
    scheduler = BatchingScheduler(backend, 4, budget, LARGE)
    body = json.dumps({"prompt": "a", "max_tokens": 500_000}).encode()
    request = (
        b"POST /generate HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )

    with running_server(scheduler, backend) as port:
        client = socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT)
        client.sendall(request)
        received = b""
        while b"data: " not in received:
            received += client.recv(4096)
        held = budget.available
        client.close()

        released = wait_until(lambda: budget.available == LARGE)

    assert held < LARGE
    assert released


def test_backend_error_sends_error_event_and_server_keeps_serving():
    class FailsOnSecondStep(FakeBackend):
        steps = 0

        def step(self):
            self.steps += 1
            if self.steps == 2:
                raise RuntimeError("decode failed")
            return super().step()

    backend = FailsOnSecondStep(tokens_per_sequence=2, step_seconds=0)

    with running_server(make_scheduler(backend), backend) as port:
        connection, response = post(port, "a", 2)
        failed_events = read_events(response)
        connection.close()
        connection, response = post(port, "a", 2)
        next_events = read_events(response)
        connection.close()

    assert "error" in json.loads(failed_events[-1])
    assert "[DONE]" not in failed_events
    assert next_events[-1] == "[DONE]"
