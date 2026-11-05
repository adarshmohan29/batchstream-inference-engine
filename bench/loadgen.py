import asyncio
import csv
import hashlib
import json
import multiprocessing
import platform
import random
import re
import subprocess
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).parent.parent
TORCH_MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
GGUF = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
RESULTS = ROOT / "results" / "phase8"
SEED = 0
RATES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
DURATION_SECONDS = 30.0
REPEATS = 3
SETTLE_POLL_SECONDS = 15.0
SETTLE_STABLE_MB = 50.0
SETTLE_STABLE_READS = 3
SETTLE_MAX_POLLS = 20
MAX_TOKENS_CHOICES = [16, 32, 64]
MAX_CONCURRENT = 8
MAX_QUEUE = 32
KV_TOKENS = 4096
BLOCK_SIZE = 16
PREFILL_CHUNK = 64
TORCH_THREADS = 4
PROMPTS = [
    "2 + 2 =",
    "The capital of France is",
    "def fibonacci(n):\n",
    "List three primary colors:\n",
    "Explain gravity in one sentence.",
    "Summarize this text: " + "the quick brown fox jumps over the lazy dog " * 4,
    "Summarize this text: " + "the quick brown fox jumps over the lazy dog " * 12,
    "Summarize this text: " + "the quick brown fox jumps over the lazy dog " * 30,
]
FIELDS = [
    "backend", "mode", "rate", "repeat", "request", "prompt_tokens", "max_tokens", "scheduled_s",
    "submit_lateness_s", "status", "ttft_s", "end_to_end_s", "tokens",
]


@dataclass(frozen=True)
class Arrival:
    time: float
    prompt: int
    max_tokens: int


def make_workload(rate: float, duration: float, seed: int, num_prompts: int) -> list[Arrival]:
    # Separate generators keep the request sequence the same at every rate.
    timing = random.Random(seed)
    requests = random.Random(seed + 1)
    workload = []
    now = timing.expovariate(rate)
    while now < duration:
        prompt = requests.randrange(num_prompts)
        workload.append(Arrival(now, prompt, requests.choice(MAX_TOKENS_CHOICES)))
        now += timing.expovariate(rate)
    return workload


async def run_load(scheduler, workload: list[Arrival], prompts: list[list[int]]) -> list[dict]:
    from server.scheduler import QueueFull, RequestTooLarge

    loop_task = asyncio.create_task(scheduler.run())
    start = time.monotonic()
    rows, submitted = [], []
    for index, arrival in enumerate(workload):
        delay = start + arrival.time - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        tokens = prompts[arrival.prompt]
        row = {
            "request": index,
            "prompt_tokens": len(tokens),
            "max_tokens": arrival.max_tokens,
            "scheduled_s": arrival.time,
            "submit_lateness_s": time.monotonic() - start - arrival.time,
            "status": "",
            "ttft_s": "",
            "end_to_end_s": "",
            "tokens": 0,
        }
        rows.append(row)
        try:
            submitted.append((row, scheduler.submit(tokens, arrival.max_tokens), start + arrival.time))
        except RequestTooLarge:
            row["status"] = "rejected_too_large"
        except QueueFull:
            row["status"] = "rejected_queue_full"
    await asyncio.gather(*(request.done.wait() for _, request, _ in submitted))
    loop_task.cancel()
    for row, request, scheduled in submitted:
        _record_outcome(row, request, scheduled)
    return rows


def _record_outcome(row: dict, request, scheduled: float) -> None:
    row["tokens"] = len(request.tokens)
    if request.failed:
        row["status"] = "failed"
        return
    row["status"] = "ok"
    # Measured from the scheduled time, so a late load generator cannot hide queueing delay.
    row["ttft_s"] = request.metrics.first_token - scheduled
    row["end_to_end_s"] = request.metrics.finish - scheduled


def _sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def _environment() -> dict:
    return {
        "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(_sysctl("hw.memsize")),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "llama_cpp_python": version("llama-cpp-python"),
        "torch": version("torch"),
        "llama_cpp_model": GGUF.name,
        "llama_cpp_model_sha256": hashlib.sha256(GGUF.read_bytes()).hexdigest(),
        "llama_cpp_quantization": "Q4_K_M",
        "llama_cpp_device": "Metal",
        "paged_model": str(TORCH_MODEL_DIR.relative_to(ROOT)),
        "paged_model_sha256": hashlib.sha256(
            (TORCH_MODEL_DIR / "model.safetensors").read_bytes()
        ).hexdigest(),
        "paged_dtype": "bfloat16",
        "paged_device": "CPU",
        "paged_attention": "C++",
        "torch_threads": TORCH_THREADS,
        "sampling": "greedy",
        "load": "open-loop Poisson arrivals, measured in process at the scheduler, without HTTP",
        "isolation": "fresh process per backend per repeat, after swap settles",
        "seed": SEED,
        "rates_per_s": RATES,
        "duration_s": DURATION_SECONDS,
        "max_concurrent": MAX_CONCURRENT,
        "max_queue": MAX_QUEUE,
        "kv_tokens": KV_TOKENS,
        "block_size": BLOCK_SIZE,
        "prefill_chunk": PREFILL_CHUNK,
    }


def _open_llamacpp():
    from server.backend.llamacpp import LlamaCppBackend
    from server.kv_budget import KVBudget

    backend = LlamaCppBackend(str(GGUF), KV_TOKENS, MAX_CONCURRENT, PREFILL_CHUNK)
    return backend, lambda: KVBudget(KV_TOKENS)


def _open_paged():
    import torch

    from server.backend.paged import PagedBackend
    from server.kv_budget import BlockBudget
    from server.paged_attention_cpp import paged_attention

    torch.set_num_threads(TORCH_THREADS)
    num_blocks = KV_TOKENS // BLOCK_SIZE
    backend = PagedBackend(
        TORCH_MODEL_DIR, GGUF, num_blocks, BLOCK_SIZE, PREFILL_CHUNK, attention=paged_attention
    )
    return backend, lambda: BlockBudget(num_blocks, BLOCK_SIZE)


def _make_scheduler(mode: str, backend, budget):
    from server.scheduler import BatchingScheduler, NaiveScheduler

    if mode == "naive":
        return NaiveScheduler(backend, budget, MAX_QUEUE)
    return BatchingScheduler(backend, MAX_CONCURRENT, budget, MAX_QUEUE)


OPENERS = {"llama.cpp": _open_llamacpp, "paged": _open_paged}


def backend_order(repeat: int) -> list[str]:
    # Flipping the order keeps whichever backend runs first from always getting a calmer machine.
    names = list(OPENERS)
    return names if repeat % 2 == 0 else names[::-1]


def parse_swap_used(usage: str) -> float:
    return float(re.search(r"used = ([0-9.]+)M", usage).group(1))


def wait_for_swap_to_settle(read_swap_used, sleep) -> float:
    """Wait until swap stops changing, because a machine still swapping distorts latency."""
    readings = [read_swap_used()]
    for _ in range(SETTLE_MAX_POLLS):
        recent = readings[-SETTLE_STABLE_READS:]
        if len(recent) == SETTLE_STABLE_READS and max(recent) - min(recent) < SETTLE_STABLE_MB:
            break
        sleep(SETTLE_POLL_SECONDS)
        readings.append(read_swap_used())
    return readings[-1]


def _swap_used() -> float:
    return parse_swap_used(_sysctl("vm.swapusage"))


def _run_backend(results: Path, name: str, repeat: int, rates: list[float], duration: float) -> None:
    backend, new_budget = OPENERS[name]()
    prompts = [backend.tokenize(prompt) for prompt in PROMPTS]
    warm_up = [Arrival(0.0, 0, 4)]
    asyncio.run(run_load(_make_scheduler("batched", backend, new_budget()), warm_up, prompts))
    with (results / "requests.csv").open("a", newline="") as requests_file, (
        results / "runs.csv"
    ).open("a", newline="") as runs_file:
        requests_writer = csv.DictWriter(requests_file, fieldnames=FIELDS)
        runs_writer = csv.writer(runs_file)
        for position, rate in enumerate(rates):
            # Flipping the order keeps warm-up and drift from always favoring one mode.
            modes = ("naive", "batched") if (position + repeat) % 2 else ("batched", "naive")
            for mode in modes:
                workload = make_workload(rate, duration, SEED, len(PROMPTS))
                scheduler = _make_scheduler(mode, backend, new_budget())
                swap_before = _swap_used()
                rows = asyncio.run(run_load(scheduler, workload, prompts))
                labels = {"backend": name, "mode": mode, "rate": rate, "repeat": repeat}
                requests_writer.writerows(labels | row for row in rows)
                runs_writer.writerow([repeat, name, mode, rate, swap_before, _swap_used()])
                requests_file.flush()
                runs_file.flush()
    backend.close()


def run_experiment(results: Path, rates: list[float], duration: float, repeats: int) -> None:
    results.mkdir(parents=True, exist_ok=True)
    info = _environment() | {"rates_per_s": rates, "duration_s": duration, "repeats": repeats}
    (results / "environment.json").write_text(json.dumps(info, indent=2) + "\n")
    with (results / "requests.csv").open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=FIELDS).writeheader()
    with (results / "runs.csv").open("w", newline="") as file:
        csv.writer(file).writerow(
            ["repeat", "backend", "mode", "rate", "swap_used_before_mb", "swap_used_after_mb"]
        )
    context = multiprocessing.get_context("spawn")
    with (results / "settling.csv").open("w", newline="") as file:
        settling = csv.writer(file)
        settling.writerow(["repeat", "backend", "swap_used_at_start_mb"])
        for repeat in range(repeats):
            for name in backend_order(repeat):
                settling.writerow([repeat, name, wait_for_swap_to_settle(_swap_used, time.sleep)])
                file.flush()
                # A fresh process leaves no memory pressure from the previous backend behind.
                process = context.Process(
                    target=_run_backend, args=(results, name, repeat, rates, duration)
                )
                process.start()
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"{name} repeat {repeat} exited with {process.exitcode}")


if __name__ == "__main__":
    run_experiment(RESULTS, RATES, DURATION_SECONDS, REPEATS)
