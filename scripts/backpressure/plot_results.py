import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(root: Path) -> dict:
    summaries = {}
    figure, axes = plt.subplots(3, 2, figsize=(13, 12))
    for label in "ABCD":
        directory = root / label
        if not (directory / "manifest.json").exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(directory / "metrics.csv")
        source = pd.read_csv(directory / "source.csv")
        tasks = pd.read_csv(directory / "tasks.csv")
        observations = pd.read_csv(directory / "observations.csv")
        if set(tasks.task_id) != set(observations.task_id):
            raise ValueError("Observed terminal IDs do not match exported tasks")
        tasks = tasks.merge(observations, on="task_id", validate="one_to_one")
        tasks["observed_latency_s"] = (
            pd.to_datetime(tasks.observed_at, utc=True)
            - pd.to_datetime(tasks.submitted_at, utc=True)
        ).dt.total_seconds()
        axes[0, 0].plot(metrics.elapsed_s, metrics.queue_ready, label=label)
        axes[0, 1].plot(
            metrics.elapsed_s, metrics.broker_memory_bytes / 2**20, label=f"{label} broker"
        )
        axes[0, 1].plot(
            metrics.elapsed_s,
            metrics.worker_rss_bytes / 2**20,
            linestyle="--",
            label=f"{label} workers",
        )
        rates = source.published_ok.diff() / source.t_s.diff()
        rates.iloc[0] = source.published_ok.iloc[0] / source.t_s.iloc[0]
        axes[1, 0].plot(source.t_s, rates, label=label, marker=".", markersize=2)
        axes[1, 1].plot(
            metrics.elapsed_s,
            (metrics.completed.diff() / metrics.elapsed_s.diff()).rolling(5).mean(),
            label=label,
        )
        completed = tasks[tasks.status == "COMPLETED"]
        ordered = np.sort(completed.observed_latency_s.dropna())
        if len(ordered):
            axes[2, 0].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)
        axes[2, 1].plot(source.t_s, source.rss_mb, label=label, marker=".", markersize=2)
        finish = observations.elapsed_s.max()
        summaries[label] = {
            "rows": manifest["dataset"]["rows"],
            "completed": manifest["completed"],
            "failed": manifest["failed"],
            "reconciled_unique_tasks": manifest["reconciled_unique_tasks"],
            "workers": manifest["workers"],
            "duration_s": manifest["duration_s"],
            "terminal_rate_s": len(tasks) / finish,
            "peak_ready": int(metrics.queue_ready.max()),
            "peak_unacked": int(metrics.queue_unacked.max()),
            "peak_broker_memory_mb": float(metrics.broker_memory_bytes.max() / 2**20),
            "published": int(source.published_ok.iloc[-1]),
            "source_rejected": int(source.rejected.iloc[-1]),
            "source_rss_min_mb": float(source.rss_mb.min()),
            "source_rss_max_mb": float(source.rss_mb.max()),
            "mean_observed_latency_s": float(completed.observed_latency_s.mean()),
            "p95_observed_latency_s": float(completed.observed_latency_s.quantile(0.95)),
            "max_observation_gap_s": float(observations.poll_gap_s.max()),
            "negative_raw_latency_rows": int((tasks.end_to_end_ms < 0).sum()),
            "mean_inference_ms": float(completed.inference_ms.mean()),
            "redelivered_peak": float(metrics.redelivered_total.max())
            if metrics.redelivered_total.notna().any()
            else None,
        }
    properties = [
        ("Ready messages", "Elapsed, s", "Messages"),
        ("Broker memory and worker RSS", "Elapsed, s", "MiB"),
        ("Source publications (A is unconfirmed)", "Elapsed, s", "Publications/s"),
        ("Completed tasks, rolling 5 samples", "Elapsed, s", "Tasks/s"),
        ("Observed latency CDF (includes polling lag)", "Latency, s", "Fraction"),
        ("Source RSS", "Elapsed, s", "MiB"),
    ]
    for axis, (title, xlabel, ylabel) in zip(axes.flat, properties, strict=True):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[1, 0].set_yscale("symlog", linthresh=1)
    figure.tight_layout()
    figure.savefig(root / "overview.png", dpi=150)
    plt.close(figure)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot measured CSV files without synthesizing data"
    )
    parser.add_argument("directory", type=Path)
    print(json.dumps(plot_results(parser.parse_args().directory), indent=2))
