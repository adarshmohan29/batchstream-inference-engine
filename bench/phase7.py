import asyncio
import csv
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch

from server.backend.paged import PagedBackend
from server.kv_budget import BlockBudget
from server.paged_attention_cpp import paged_attention as cpp_attention
from server.scheduler import BatchingScheduler
from server.torch_model import ModelConfig, paged_attention as python_attention

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
GGUF = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
RESULTS = ROOT / "results" / "phase7"
SEED = 0
THREADS = 4
NUM_BLOCKS = 256
BLOCK_SIZE = 16
PREFILL_CHUNK = 64
MAX_CONCURRENT = 4
MAX_QUEUE = 64
FULL_CONTEXT_TOKENS = 4096
ATTENTION_LENGTHS = [16, 64, 256, 1024, 4096]
ATTENTION_REPEATS = 15
SPEED_REPEATS = 5
SPEED_TOKENS = 24
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "List three primary colors:\n",
    "2 + 2 =",
    "Name a large river:",
    "Explain gravity in one sentence.",
    "Translate 'good morning' to Spanish:",
    "Summarize this text: " + "the quick brown fox jumps over the lazy dog " * 12,
]
IMPLEMENTATIONS = {"python": python_attention, "cpp": cpp_attention}


def bytes_per_token(config: ModelConfig) -> int:
    bf16_bytes = 2
    return 2 * config.num_layers * config.num_kv_heads * config.head_dim * bf16_bytes


def summarize_memory(
    samples: list[tuple[int, int, int]],
    block_size: int,
    full_context_tokens: int,
    token_bytes: int,
) -> dict[str, float]:
    """Each sample is (blocks in use, active sequences, blocks reserved by the budget)."""
    block_bytes = block_size * token_bytes
    context_bytes = full_context_tokens * token_bytes
    paged_peak = max(blocks for blocks, _, _ in samples) * block_bytes
    full_peak = max(active for _, active, _ in samples) * context_bytes
    reserved_peak = max(reserved for _, _, reserved in samples) * block_bytes
    return {
        "paged_peak_bytes": paged_peak,
        "full_context_peak_bytes": full_peak,
        "worst_case_reserved_peak_bytes": reserved_peak,
        "paged_mean_bytes": statistics.mean(blocks for blocks, _, _ in samples) * block_bytes,
        "full_context_mean_bytes": statistics.mean(a for _, a, _ in samples) * context_bytes,
        "peak_saving_vs_full_context_percent": 100 * (1 - paged_peak / full_peak),
        "peak_saving_vs_worst_case_reserved_percent": 100 * (1 - paged_peak / reserved_peak),
    }


class _SamplingBackend(PagedBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples: list[tuple[int, int, int]] = []
        self.budget: BlockBudget | None = None

    def step(self):
        active = len(self._sequences)
        outputs = super().step()
        reserved = NUM_BLOCKS - self.budget.available
        self.samples.append((self.blocks_in_use, active, reserved))
        return outputs


def serve(backend, specs) -> float:
    async def run():
        backend.budget = BlockBudget(NUM_BLOCKS, BLOCK_SIZE)
        scheduler = BatchingScheduler(backend, MAX_CONCURRENT, backend.budget, MAX_QUEUE)
        loop_task = asyncio.create_task(scheduler.run())
        start = time.perf_counter()
        requests = [scheduler.submit(tokens, max_tokens) for tokens, max_tokens in specs]
        await asyncio.gather(*(request.done.wait() for request in requests))
        elapsed = time.perf_counter() - start
        loop_task.cancel()
        assert not any(request.failed for request in requests)
        return elapsed

    return asyncio.run(run())


def measure_memory(backend: _SamplingBackend) -> dict[str, float]:
    rng = random.Random(SEED)
    specs = [(backend.tokenize(p), rng.choice([16, 32, 64])) for p in PROMPTS]
    backend.samples.clear()
    serve(backend, specs)
    summary = summarize_memory(
        backend.samples, BLOCK_SIZE, FULL_CONTEXT_TOKENS, bytes_per_token(backend._model.config)
    )
    summary["requests"] = len(specs)
    summary["requested_tokens"] = sum(len(tokens) + limit for tokens, limit in specs)
    return summary


def measure_attention() -> list[dict]:
    rows = []
    generator = torch.Generator().manual_seed(SEED)
    head_dim, kv_heads, groups = 64, 2, 7
    for length in ATTENTION_LENGTHS:
        blocks = -(-length // BLOCK_SIZE)
        shape = (kv_heads, blocks, BLOCK_SIZE, head_dim)
        keys = torch.randn(shape, generator=generator).to(torch.bfloat16)
        values = torch.randn(shape, generator=generator).to(torch.bfloat16)
        query = torch.randn(kv_heads * groups, head_dim, generator=generator).to(torch.bfloat16)
        table = torch.randperm(blocks, generator=generator).tolist()
        times = {name: [] for name in IMPLEMENTATIONS}
        for _ in range(ATTENTION_REPEATS):
            # Alternating keeps slow drift in the machine from favoring one implementation.
            for name, attention in IMPLEMENTATIONS.items():
                start = time.perf_counter()
                attention(query, keys, values, table, length, BLOCK_SIZE, groups, head_dim**-0.5)
                times[name].append(time.perf_counter() - start)
        python_ms, cpp_ms = (1000 * statistics.median(times[n]) for n in ("python", "cpp"))
        rows.append(
            {"length": length, "python_ms": python_ms, "cpp_ms": cpp_ms, "cpp_speedup": python_ms / cpp_ms}
        )
    return rows


def measure_end_to_end(backend: _SamplingBackend) -> list[dict]:
    rows = []
    for concurrent in (1, 4):
        specs = [(backend.tokenize(p), SPEED_TOKENS) for p in PROMPTS[:concurrent]]
        total = sum(limit for _, limit in specs)
        times = {name: [] for name in IMPLEMENTATIONS}
        serve(backend, specs)
        for repeat in range(SPEED_REPEATS):
            # Flipping the order each round keeps warm-up and drift from favoring one version.
            order = list(IMPLEMENTATIONS.items())
            for name, attention in order if repeat % 2 else reversed(order):
                backend._pool.attention = attention
                times[name].append(serve(backend, specs))
        python_rate, cpp_rate = (total / statistics.median(times[n]) for n in ("python", "cpp"))
        rows.append(
            {
                "concurrent": concurrent,
                "python_tokens_per_s": python_rate,
                "cpp_tokens_per_s": cpp_rate,
                "cpp_speedup": cpp_rate / python_rate,
            }
        )
    backend._pool.attention = python_attention
    return rows


def _sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def environment() -> dict:
    digest = hashlib.sha256((MODEL_DIR / "model.safetensors").read_bytes()).hexdigest()
    return {
        "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(_sysctl("hw.memsize")),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": version("torch"),
        "llama_cpp_python": version("llama-cpp-python"),
        "torch_threads": torch.get_num_threads(),
        "model": str(MODEL_DIR.relative_to(ROOT)),
        "model_sha256": digest,
        "dtype": "bfloat16, CPU",
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "num_blocks": NUM_BLOCKS,
        "max_concurrent": MAX_CONCURRENT,
        "full_context_tokens": FULL_CONTEXT_TOKENS,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    info = environment()
    info["swap_before"] = _sysctl("vm.swapusage")
    backend = _SamplingBackend(MODEL_DIR, GGUF, NUM_BLOCKS, BLOCK_SIZE, PREFILL_CHUNK)
    memory = measure_memory(backend)
    attention = measure_attention()
    end_to_end = measure_end_to_end(backend)
    backend.close()
    info["swap_after"] = _sysctl("vm.swapusage")
    write_csv(RESULTS / "memory.csv", [memory])
    write_csv(RESULTS / "attention_speed.csv", attention)
    write_csv(RESULTS / "end_to_end_speed.csv", end_to_end)
    (RESULTS / "environment.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
