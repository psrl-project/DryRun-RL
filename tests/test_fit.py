"""Tests for the two-stage regression cost model fitter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dryrun.cost.empirical import PSRLFitted
from dryrun.profiling.fit import fit_cost_model, save_fit_result


def _generate_synthetic_data(
    other_k: float,
    other_b: float,
    other_threshold: float,
    attn_k: float,
    attn_b: float,
    n_points: int = 50,
    noise_std: float = 0.0,
    seed: int = 42,
) -> list[dict]:
    """
    Generate synthetic profiling data from known parameters.

    Produces a grid-like sweep: for each request_num, generate several
    context lengths. This mirrors real profiling which sweeps
    (batch_size x context_length).
    """
    rng = np.random.RandomState(seed)
    data = []
    request_nums = sorted(set(rng.randint(1, 48, size=n_points // 3)))
    if not request_nums:
        request_nums = [1, 4, 8, 16, 32]
    context_lengths = [256, 512, 1024, 2048]

    for req_num in request_nums:
        for ctx_len in context_lengths:
            token_num = req_num * ctx_len

            other_latency = max(other_threshold, other_b + other_k * req_num)
            attn_latency = attn_k * token_num + attn_b
            total_latency = other_latency + attn_latency

            if noise_std > 0:
                total_latency += rng.normal(0, noise_std * total_latency)
                total_latency = max(total_latency, 1e-6)

            throughput = req_num / total_latency

            data.append({
                "request_num": int(req_num),
                "token_num": int(token_num),
                "throughput": float(throughput),
            })

    while len(data) < n_points:
        req_num = rng.randint(1, 48)
        ctx_len = rng.choice(context_lengths)
        token_num = req_num * ctx_len

        other_latency = max(other_threshold, other_b + other_k * req_num)
        attn_latency = attn_k * token_num + attn_b
        total_latency = other_latency + attn_latency

        if noise_std > 0:
            total_latency += rng.normal(0, noise_std * total_latency)
            total_latency = max(total_latency, 1e-6)

        throughput = req_num / total_latency
        data.append({
            "request_num": int(req_num),
            "token_num": int(token_num),
            "throughput": float(throughput),
        })

    return data[:n_points] if len(data) > n_points else data


class TestFitCostModel:
    """Test the two-stage regression fitter."""

    def test_perfect_recovery(self):
        """With no noise, the combined model should produce low MRE."""
        true_params = {
            "other_k": 0.0003,
            "other_b": 0.005,
            "other_threshold": 0.008,
            "attn_k": 2e-6,
            "attn_b": 0.002,
        }
        data = _generate_synthetic_data(**true_params, n_points=100, noise_std=0.0)
        result = fit_cost_model(data)

        assert result.fit_quality.other_r_squared > 0.95, (
            f"Stage 1 R² too low: {result.fit_quality.other_r_squared:.4f}."
        )
        assert result.fit_quality.attn_r_squared > 0.90, (
            f"Stage 2 R² too low: {result.fit_quality.attn_r_squared:.4f}."
        )
        assert result.fit_quality.mre_percent < 25, (
            f"Combined MRE too high: {result.fit_quality.mre_percent:.1f}%."
        )

    def test_noisy_data(self):
        """With moderate noise, the combined model should still be reasonable."""
        true_params = {
            "other_k": 0.0005,
            "other_b": 0.003,
            "other_threshold": 0.006,
            "attn_k": 5e-6,
            "attn_b": 0.001,
        }
        data = _generate_synthetic_data(**true_params, n_points=200, noise_std=0.05)
        result = fit_cost_model(data)

        assert result.fit_quality.mre_percent < 50, (
            f"MRE too high for noisy data: {result.fit_quality.mre_percent:.1f}%."
        )
        assert result.fit_quality.n_samples > 0, "No valid samples."

    def test_csv_roundtrip(self, tmp_path: Path):
        """Test loading from CSV and fitting."""
        import csv

        csv_path = tmp_path / "test_data.csv"
        data = _generate_synthetic_data(
            other_k=0.0004, other_b=0.004, other_threshold=0.007,
            attn_k=3e-6, attn_b=0.0015, n_points=80,
        )

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["request_num", "token_num", "throughput"])
            writer.writeheader()
            writer.writerows(data)

        result = fit_cost_model(str(csv_path))
        assert result.fit_quality.n_samples == 80, (
            f"Expected 80 samples, got {result.fit_quality.n_samples}."
        )

    def test_json_psrl_compatibility(self, tmp_path: Path):
        """Verify the output JSON is loadable by PSRLFitted.from_json()."""
        data = _generate_synthetic_data(
            other_k=0.0003, other_b=0.005, other_threshold=0.008,
            attn_k=2e-6, attn_b=0.002, n_points=100,
        )
        result = fit_cost_model(data)

        json_path = tmp_path / "cost_model.json"
        save_fit_result(result, json_path, key="TP1_PP1")

        model = PSRLFitted.from_json(json_path, key="TP1_PP1")
        assert model.attn_b == result.attn_latency_b
        assert model.attn_k == result.attn_latency_k
        assert model.oth == result.other_threshold
        assert model.ob == result.other_latency_b
        assert model.ok == result.other_latency_k

        latency = model.step_time(n_tokens=16, t2=0, ctxsum=8000, n_reqs=16)
        assert latency > 0, f"Cost model returned non-positive latency: {latency}."

    def test_json_merge(self, tmp_path: Path):
        """Test that multiple keys can be merged into one JSON file."""
        data = _generate_synthetic_data(
            other_k=0.0003, other_b=0.005, other_threshold=0.008,
            attn_k=2e-6, attn_b=0.002, n_points=50,
        )
        result = fit_cost_model(data)
        json_path = tmp_path / "merged.json"

        save_fit_result(result, json_path, key="TP1_PP1")
        save_fit_result(result, json_path, key="TP2_PP1", merge=True)

        with open(json_path) as fh:
            loaded = json.load(fh)

        assert "TP1_PP1" in loaded, "TP1_PP1 key missing after merge."
        assert "TP2_PP1" in loaded, "TP2_PP1 key missing after merge."

    def test_insufficient_data(self):
        """Fitting with too few data points should raise."""
        data = [
            {"request_num": 1, "token_num": 100, "throughput": 10.0},
            {"request_num": 2, "token_num": 200, "throughput": 15.0},
        ]
        with pytest.raises(ValueError, match="at least 3"):
            fit_cost_model(data)

    def test_outlier_removal(self):
        """Verify IQR outlier removal handles extreme values."""
        true_params = {
            "other_k": 0.0003,
            "other_b": 0.005,
            "other_threshold": 0.008,
            "attn_k": 2e-6,
            "attn_b": 0.002,
        }
        data = _generate_synthetic_data(**true_params, n_points=100, noise_std=0.0)

        rng = np.random.RandomState(99)
        for i in rng.choice(len(data), size=5, replace=False):
            data[i]["throughput"] *= 0.1

        result = fit_cost_model(data)
        assert result.fit_quality.n_outliers_removed >= 1, (
            "Expected at least 1 outlier to be removed."
        )
