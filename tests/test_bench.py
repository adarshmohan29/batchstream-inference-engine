import asyncio
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from server.backend.fake import FakeBackend
from server.kv_budget import KVBudget
from server.scheduler import BatchingScheduler, NaiveScheduler

BENCH = Path(__file__).parent.parent / "bench"


def load(name):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


loadgen = load("loadgen")
report = load("report")
LARGE = 1_000_000


def arrivals(*times, max_tokens=4):
    return [loadgen.Arrival(time, 0, max_tokens) for time in times]


def run(scheduler, workload, prompts=([1, 2],)):
    return asyncio.run(loadgen.run_load(scheduler, workload, list(prompts)))


def row(backend="b", mode="batched", rate=1.0, status="ok", ttft=0.1, e2e=1.0, tokens=4, scheduled=0.0, repeat=0):
    return {
        "backend": backend, "mode": mode, "rate": rate, "repeat": repeat, "status": status,
        "ttft_s": ttft, "end_to_end_s": e2e, "tokens": tokens, "scheduled_s": scheduled,
    }


def test_workload_is_deterministic_for_a_seed():
    assert loadgen.make_workload(2.0, 30.0, 0, 8) == loadgen.make_workload(2.0, 30.0, 0, 8)


def test_seed_changes_the_workload():
    assert loadgen.make_workload(2.0, 30.0, 0, 8) != loadgen.make_workload(2.0, 30.0, 1, 8)


def test_rates_share_the_same_request_sequence():
    slow = loadgen.make_workload(1.0, 30.0, 0, 8)
    fast = loadgen.make_workload(4.0, 30.0, 0, 8)

    shared = min(len(slow), len(fast))
    assert [(a.prompt, a.max_tokens) for a in slow[:shared]] == [
        (a.prompt, a.max_tokens) for a in fast[:shared]
    ]


def test_arrivals_are_sorted_within_duration_at_about_the_rate():
    workload = loadgen.make_workload(10.0, 200.0, 0, 8)
    times = [arrival.time for arrival in workload]

    assert times == sorted(times)
    assert times[-1] < 200.0
    assert len(workload) == pytest.approx(2000, rel=0.1)


def test_load_is_open_loop_and_ignores_completions():
    backend = FakeBackend(tokens_per_sequence=50, step_seconds=0.01)
    scheduler = NaiveScheduler(backend, KVBudget(LARGE), LARGE)

    rows = run(scheduler, arrivals(0.0, 0.02, 0.04), [[1]])

    assert [r["status"] for r in rows] == ["ok", "ok", "ok"]
    assert all(r["submit_lateness_s"] < 0.02 for r in rows)


def test_latency_is_measured_from_scheduled_arrival_and_includes_queueing():
    backend = FakeBackend(tokens_per_sequence=10, step_seconds=0.01)
    scheduler = NaiveScheduler(backend, KVBudget(LARGE), LARGE)

    first, second = run(scheduler, arrivals(0.0, 0.0, max_tokens=10))

    assert second["ttft_s"] >= first["end_to_end_s"] - 0.001
    assert second["end_to_end_s"] >= second["ttft_s"]
    assert first["tokens"] == 10


def test_queue_full_is_recorded_as_rejected():
    backend = FakeBackend(tokens_per_sequence=20, step_seconds=0.01)
    scheduler = BatchingScheduler(backend, 1, KVBudget(LARGE), 1)

    rows = run(scheduler, arrivals(0.0, 0.0, 0.0))

    assert [r["status"] for r in rows].count("rejected_queue_full") >= 1
    assert all(r["ttft_s"] == "" for r in rows if r["status"] != "ok")


def test_too_large_is_recorded_separately():
    backend = FakeBackend(tokens_per_sequence=1, step_seconds=0)
    scheduler = BatchingScheduler(backend, 1, KVBudget(3), LARGE)

    [row_] = run(scheduler, arrivals(0.0, max_tokens=100))

    assert row_["status"] == "rejected_too_large"


def test_backend_failure_is_recorded_as_failed():
    class Failing(FakeBackend):
        def step(self):
            raise RuntimeError("decode failed")

    scheduler = BatchingScheduler(Failing(1, 0), 1, KVBudget(LARGE), LARGE)

    [row_] = run(scheduler, arrivals(0.0))

    assert row_["status"] == "failed"


def test_percentile_uses_nearest_rank():
    values = list(range(1, 101))

    assert report.percentile(values, 50) == 50
    assert report.percentile(values, 99) == 99
    assert report.percentile([7.0], 99) == 7.0


def test_summary_uses_makespan_and_counts_each_outcome():
    rows = [
        row(tokens=10, e2e=2.0, scheduled=0.0),
        row(tokens=10, e2e=2.0, scheduled=3.0),
        row(status="rejected_queue_full", ttft="", e2e="", tokens=0, scheduled=1.0),
        row(status="rejected_too_large", ttft="", e2e="", tokens=0, scheduled=1.0),
        row(status="failed", ttft="", e2e="", tokens=3, scheduled=1.0),
    ]

    [summary] = report.summarize(rows)

    assert summary["completed"] == 2
    assert summary["rejected_queue_full"] == 1
    assert summary["rejected_too_large"] == 1
    assert summary["failed"] == 1
    assert summary["tokens_per_s"] == pytest.approx(20 / 5.0)


def test_percentiles_ignore_rejected_and_groups_stay_separate():
    rows = [
        row(mode="naive", ttft=1.0, e2e=2.0),
        row(mode="naive", status="rejected_queue_full", ttft="", e2e="", tokens=0),
        row(mode="batched", ttft=0.5, e2e=1.0),
    ]

    summaries = {s["mode"]: s for s in report.summarize(rows)}

    assert summaries["naive"]["ttft_p50_s"] == 1.0
    assert summaries["batched"]["ttft_p50_s"] == 0.5


def test_group_with_nothing_completed_has_empty_latency():
    rows = [row(status="rejected_queue_full", ttft="", e2e="", tokens=0)]

    [summary] = report.summarize(rows)

    assert summary["ttft_p99_s"] == ""
    assert summary["tokens_per_s"] == 0


def test_summary_keeps_repeats_separate():
    rows = [row(repeat=0, ttft=1.0), row(repeat=1, ttft=3.0)]

    summaries = report.summarize(rows)

    assert [s["repeat"] for s in summaries] == [0, 1]
    assert [s["ttft_p50_s"] for s in summaries] == [1.0, 3.0]


def test_aggregate_reports_median_and_range_across_repeats():
    rows = [row(repeat=index, ttft=ttft) for index, ttft in enumerate((1.0, 9.0, 2.0))]

    [combined] = report.aggregate(report.summarize(rows))

    assert combined["repeats"] == 3
    assert combined["ttft_p50_s"] == 2.0
    assert combined["ttft_p50_s_min"] == 1.0
    assert combined["ttft_p50_s_max"] == 9.0


def test_aggregate_totals_rejections_per_repeat_before_taking_the_median():
    rows = [
        row(repeat=0, status="rejected_queue_full", ttft="", e2e="", tokens=0),
        row(repeat=0, status="failed", ttft="", e2e="", tokens=0),
        row(repeat=1),
        row(repeat=2, status="rejected_queue_full", ttft="", e2e="", tokens=0),
    ]

    [combined] = report.aggregate(report.summarize(rows))

    assert combined["rejected_total"] == 1
    assert combined["rejected_total_min"] == 0
    assert combined["rejected_total_max"] == 2


def test_aggregate_ignores_repeats_with_no_completed_requests():
    rows = [
        row(repeat=0, ttft=4.0),
        row(repeat=1, status="rejected_queue_full", ttft="", e2e="", tokens=0),
    ]

    [combined] = report.aggregate(report.summarize(rows))

    assert combined["ttft_p50_s"] == 4.0
    assert combined["ttft_p50_s_min"] == 4.0


def test_aggregate_is_empty_when_no_repeat_completed_anything():
    rows = [row(status="rejected_queue_full", ttft="", e2e="", tokens=0)]

    [combined] = report.aggregate(report.summarize(rows))

    assert combined["ttft_p99_s"] == ""
    assert combined["ttft_p99_s_max"] == ""


def test_swap_used_is_parsed_from_sysctl_output():
    assert loadgen.parse_swap_used("total = 8192.00M  used = 6559.25M  free = 1632.75M  (encrypted)") == 6559.25


def test_settling_waits_until_swap_stops_changing():
    readings = iter([8000.0, 7000.0, 6000.0, 5990.0, 5985.0, 1.0])
    sleeps = []

    settled = loadgen.wait_for_swap_to_settle(lambda: next(readings), sleeps.append)

    assert settled == 5985.0
    assert len(sleeps) == 4


def test_settling_gives_up_after_the_time_limit():
    readings = iter(range(0, 100_000, 1000))
    sleeps = []

    loadgen.wait_for_swap_to_settle(lambda: float(next(readings)), sleeps.append)

    assert len(sleeps) == loadgen.SETTLE_MAX_POLLS


def test_backend_order_flips_between_repeats():
    assert loadgen.backend_order(0) == list(reversed(loadgen.backend_order(1)))


def test_report_writes_csv_and_plots_from_csv_only(tmp_path):
    source = tmp_path / "requests.csv"
    rows = [
        row(backend=backend, mode=mode, rate=rate, e2e=rate, ttft=rate / 2, repeat=repeat)
        for backend in ("a", "b") for mode in ("naive", "batched") for rate in (1.0, 2.0)
        for repeat in (0, 1)
    ]
    with source.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report.main(source, tmp_path)

    assert (tmp_path / "summary_by_repeat.csv").exists()
    assert len(report.read_csv(tmp_path / "summary.csv")) == 8
    for name in ("tokens_per_s", "ttft", "end_to_end", "rejections"):
        assert (tmp_path / "plots" / f"{name}.png").stat().st_size > 0
