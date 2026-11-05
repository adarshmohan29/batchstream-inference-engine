import importlib.util
from pathlib import Path

import pytest
import torch

from server.torch_model import ModelConfig, PagedKVPool

spec = importlib.util.spec_from_file_location(
    "phase7", Path(__file__).parent.parent / "bench" / "phase7.py"
)
phase7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase7)


def test_bytes_per_token_matches_real_pool_tensors():
    config = ModelConfig(3, 16, 4, 2, 10000.0, 1e-6)
    pool = PagedKVPool(config, num_blocks=5, block_size=7)
    real_bytes = 2 * pool.keys.numel() * torch.finfo(torch.bfloat16).bits // 8

    assert phase7.bytes_per_token(config) * 5 * 7 == real_bytes


def test_summarize_memory_compares_blocks_with_both_reservation_policies():
    samples = [(2, 1, 4), (6, 2, 10), (4, 2, 8)]

    summary = phase7.summarize_memory(
        samples, block_size=16, full_context_tokens=100, token_bytes=10
    )

    assert summary["paged_peak_bytes"] == 960
    assert summary["full_context_peak_bytes"] == 2000
    assert summary["worst_case_reserved_peak_bytes"] == 1600
    assert summary["paged_mean_bytes"] == 640
    assert summary["full_context_mean_bytes"] == pytest.approx(5 / 3 * 1000)
    assert summary["peak_saving_vs_full_context_percent"] == pytest.approx(52)
    assert summary["peak_saving_vs_worst_case_reserved_percent"] == pytest.approx(40)
