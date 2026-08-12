"""Offline matplotlib visualization of simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..sim import SimResult


class SimPlotter:
    """
    Generates offline plots from simulation results.
    """

    def __init__(self, result: SimResult, output_dir: str | Path = "outputs/plots"):
        self.result = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_train_timeline(self) -> Path:
        fig, ax = plt.subplots(figsize=(12, 3))
        for i, (start, end) in enumerate(self.result.train_timestamps):
            ax.barh(0, end - start, left=start, height=0.5, color="steelblue", edgecolor="navy", linewidth=0.5)
        ax.set_xlabel("Simulation Time (s)")
        ax.set_yticks([])
        ax.set_title("Training Step Timeline")
        path = self.output_dir / "train_timeline.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_staleness_distribution(self) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        true_vals = self.result.staleness_true_values
        reported_vals = self.result.staleness_reported_values
        if true_vals:
            bins = range(min(true_vals), max(true_vals) + 2)
            ax.hist(true_vals, bins=bins, alpha=0.7, label="True (v_min)", color="coral")
            if reported_vals:
                ax.hist(reported_vals, bins=bins, alpha=0.5, label="Reported (v_max)", color="skyblue")
        ax.set_xlabel("Staleness")
        ax.set_ylabel("Count")
        ax.set_title("Staleness Distribution")
        ax.legend()
        path = self.output_dir / "staleness_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_request_timeline(self) -> Path:
        fig, ax = plt.subplots(figsize=(12, 6))
        consumed = self.result.consumed
        if not consumed:
            plt.close(fig)
            return self.output_dir / "request_timeline.png"

        admit_times = [r.admit_time for r in consumed]
        complete_times = [r.complete_time or r.admit_time for r in consumed]
        durations = [c - a for a, c in zip(admit_times, complete_times)]
        lengths = [r.generated for r in consumed]

        sc = ax.scatter(admit_times, durations, c=lengths, s=10, alpha=0.6, cmap="viridis")
        ax.set_xlabel("Admission Time (s)")
        ax.set_ylabel("Generation Duration (s)")
        ax.set_title("Request Durations (color = output length)")
        fig.colorbar(sc, ax=ax, label="Output Tokens")
        path = self.output_dir / "request_timeline.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_throughput(self, window_size: float = 5.0) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        consumed = self.result.consumed
        if not consumed:
            plt.close(fig)
            return self.output_dir / "throughput.png"

        times = sorted(r.complete_time for r in consumed if r.complete_time is not None)
        if not times:
            plt.close(fig)
            return self.output_dir / "throughput.png"

        t_max = max(times)
        bins = np.arange(0, t_max + window_size, window_size)
        counts, edges = np.histogram(times, bins=bins)
        rates = counts / window_size

        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, rates, "b-", linewidth=1.5)
        ax.fill_between(centers, rates, alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Completions / {window_size}s")
        ax.set_title("Throughput Over Time")
        path = self.output_dir / "throughput.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_all(self) -> list[Path]:
        paths = []
        paths.append(self.plot_train_timeline())
        paths.append(self.plot_staleness_distribution())
        paths.append(self.plot_request_timeline())
        paths.append(self.plot_throughput())
        return paths
