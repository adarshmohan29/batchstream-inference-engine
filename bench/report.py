import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

RESULTS = Path(__file__).parent.parent / "results" / "phase8"
MODES = ["batched", "naive"]
COLORS = {"batched": "#2a78d6", "naive": "#eb6834"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
LABEL_HEIGHT_PIXELS = 13
STATUSES = ["rejected_queue_full", "rejected_too_large", "failed"]
METRICS = [
    "tokens_per_s", "ttft_p50_s", "ttft_p99_s", "end_to_end_p50_s", "end_to_end_p99_s",
    "rejected_total", *STATUSES,
]


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(percent / 100 * len(ordered)) - 1]


def _latency(values: list[float], percent: float) -> float | str:
    return percentile(values, percent) if values else ""


def _makespan(rows: list[dict]) -> float:
    completed = [r for r in rows if r["status"] == "ok"]
    first = min(float(r["scheduled_s"]) for r in rows)
    return max(
        (float(r["scheduled_s"]) + float(r["end_to_end_s"]) for r in completed), default=first
    ) - first


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["backend"], row["mode"], float(row["rate"]), int(row["repeat"]))].append(row)
    summaries = []
    for (backend, mode, rate, repeat), group in sorted(groups.items()):
        completed = [r for r in group if r["status"] == "ok"]
        ttft = [float(r["ttft_s"]) for r in completed]
        end_to_end = [float(r["end_to_end_s"]) for r in completed]
        makespan = _makespan(group)
        tokens = sum(int(r["tokens"]) for r in completed)
        summary = {
            "backend": backend,
            "mode": mode,
            "rate": rate,
            "repeat": repeat,
            "requests": len(group),
            "completed": len(completed),
            "tokens_per_s": tokens / makespan if makespan else 0,
            "ttft_p50_s": _latency(ttft, 50),
            "ttft_p99_s": _latency(ttft, 99),
            "end_to_end_p50_s": _latency(end_to_end, 50),
            "end_to_end_p99_s": _latency(end_to_end, 99),
        }
        counts = {s: sum(r["status"] == s for r in group) for s in STATUSES}
        summaries.append(summary | counts | {"rejected_total": sum(counts.values())})
    return summaries


def aggregate(summaries: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for summary in summaries:
        groups[(summary["backend"], summary["mode"], summary["rate"])].append(summary)
    combined = []
    for (backend, mode, rate), group in sorted(groups.items()):
        row = {"backend": backend, "mode": mode, "rate": rate, "repeats": len(group)}
        for metric in METRICS:
            values = [s[metric] for s in group if s[metric] != ""]
            row[metric] = statistics.median(values) if values else ""
            row[f"{metric}_min"] = min(values) if values else ""
            row[f"{metric}_max"] = max(values) if values else ""
        combined.append(row)
    return combined


def write_summary(summaries: list[dict], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _style(axis, title: str, ylabel: str | None) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=TEXT, fontsize=11, loc="left")
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axis.grid(color=GRID, linewidth=1)
    axis.set_axisbelow(True)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.set_xlabel("Arrival rate (requests/s)", color=TEXT_SECONDARY, fontsize=9)
    if ylabel:
        axis.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)


def _draw_series(axis, summaries: list[dict], backend: str, metric: str) -> None:
    for mode in MODES:
        points = [
            (float(s["rate"]), float(s[metric]))
            for s in summaries
            if s["backend"] == backend and s["mode"] == mode and s[metric] != ""
        ]
        if not points:
            continue
        rates, values = zip(*sorted(points))
        spread = sorted(
            (float(s["rate"]), float(s[f"{metric}_min"]), float(s[f"{metric}_max"]))
            for s in summaries
            if s["backend"] == backend and s["mode"] == mode and s[metric] != ""
        )
        axis.fill_between(
            rates, [low for _, low, _ in spread], [high for _, _, high in spread],
            color=COLORS[mode], alpha=0.18, linewidth=0,
        )
        axis.plot(
            rates, values, color=COLORS[mode], linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, solid_capstyle="round", label=mode,
        )


def _label_line_ends(axis) -> None:
    """Label each line at its last point, pushing apart labels that would collide."""
    ends = []
    for line in axis.get_lines():
        x, y = line.get_xdata()[-1], line.get_ydata()[-1]
        ends.append((axis.transData.transform((x, y))[1], x, y, line.get_label()))
    ends.sort()
    minimum_gap = LABEL_HEIGHT_PIXELS
    placed = []
    for pixel_y, *_ in ends:
        placed.append(max(pixel_y, placed[-1] + minimum_gap) if placed else pixel_y)
    if placed:
        shift = (placed[-1] - ends[-1][0]) / 2
        placed = [position - shift for position in placed]
    for position, (pixel_y, x, y, label) in zip(placed, ends):
        points = (position - pixel_y) * 72 / axis.figure.dpi
        axis.annotate(
            label, (x, y), xytext=(8, points), textcoords="offset points", va="center",
            color=TEXT_SECONDARY, fontsize=9, annotation_clip=False,
        )


def _figure(summaries, panels, ylabel, path, log_y=False, title="") -> None:
    backends = sorted({s["backend"] for s in summaries})
    rows = len(panels)
    figure, axes = plt.subplots(
        rows, len(backends), figsize=(4.6 * len(backends), 3.1 * rows + 0.7),
        sharey=True, squeeze=False, facecolor=SURFACE,
    )
    for row, (metric, label) in enumerate(panels):
        for column, backend in enumerate(backends):
            axis = axes[row][column]
            panel_title = f"{backend}: {label}" if label else backend
            _style(axis, panel_title, ylabel if column == 0 else None)
            _draw_series(axis, summaries, backend, metric)
            if log_y:
                axis.set_yscale("log")
                axis.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                axis.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if not log_y:
        axes[0][0].set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper right", frameon=False, ncols=2, fontsize=9, labelcolor=TEXT
    )
    figure.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=13)
    figure.tight_layout(rect=(0, 0, 0.98, 0.97))
    # Line-end labels are added after layout, so they need room reserved beside each panel.
    figure.subplots_adjust(right=0.92, wspace=0.28)
    for axis in axes.flat:
        _label_line_ends(axis)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def write_plots(summary_csv: Path, plots: Path) -> None:
    summaries = read_csv(summary_csv)
    plots.mkdir(parents=True, exist_ok=True)
    _figure(
        summaries, [("tokens_per_s", "")], "Tokens/s", plots / "tokens_per_s.png",
        title="Throughput by arrival rate (median and range across repeats)",
    )
    _figure(
        summaries, [("ttft_p50_s", "p50"), ("ttft_p99_s", "p99")], "Seconds", plots / "ttft.png",
        log_y=True, title="Time to first token, completed requests (median and range across repeats)",
    )
    _figure(
        summaries, [("end_to_end_p50_s", "p50"), ("end_to_end_p99_s", "p99")], "Seconds",
        plots / "end_to_end.png", log_y=True, title="End-to-end latency, completed requests (median and range across repeats)",
    )
    _figure(
        summaries, [("rejected_total", "")], "Requests", plots / "rejections.png",
        title="Rejected or failed requests (median and range across repeats)",
    )


def main(source: Path, results: Path) -> None:
    summaries = summarize(read_csv(source))
    write_summary(summaries, results / "summary_by_repeat.csv")
    summary_csv = results / "summary.csv"
    write_summary(aggregate(summaries), summary_csv)
    write_plots(summary_csv, results / "plots")


if __name__ == "__main__":
    main(RESULTS / "requests.csv", RESULTS)
