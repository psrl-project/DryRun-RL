"""Two-stage regression fitter for inference cost models.

Fits profiling data (request_num, token_num, throughput) to the PSRLFitted
cost model formula:

    tau = attn_b + attn_k * ctxsum + max(other_threshold, other_b + other_k * n_reqs)

Stage 1: Fit the non-attention (other) component against request count.
    Model: throughput = request_num / max(threshold, b + k * request_num)
    This captures how non-attention ops (GEMMs, sampling) scale with batch
    size, including the roofline knee where the GPU is under-utilised.

Stage 2: Subtract the fitted other-latency from observed latency to isolate
    the attention residual. Fit a linear model:
    attention_latency = attn_k * token_num + attn_b
    This captures how attention cost scales with total KV cache tokens.
    An IQR-based outlier filter (2.5x threshold) removes noisy samples
    before the final refit.

The output JSON is directly loadable by `PSRLFitted.from_json()`.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from .schemas import FitQuality, FitResult

dryrun_logger = logging.getLogger("dryrun")


def _load_csv(path: str | Path) -> list[dict]:
    """Load profiling data from CSV."""
    rows = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "request_num": int(row["request_num"]),
                "token_num": int(row["token_num"]),
                "throughput": float(row["throughput"]),
            })
    return rows


def _throughput_model(request_num, k, b, threshold):
    """Throughput model: request_num / max(threshold, b + k*request_num)."""
    latency = np.maximum(threshold, b + k * request_num)
    return request_num / latency


def _latency_model(request_num, k, b, threshold):
    """Latency model: max(threshold, b + k*request_num)."""
    return np.maximum(threshold, b + k * request_num)


def _attn_latency_model(token_num, attn_k, attn_b):
    """Attention latency model: attn_k * token_num + attn_b."""
    return attn_k * token_num + attn_b


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def fit_other_latency(data: list[dict]) -> dict:
    """
    Stage 1: Fit the non-attention (other) latency component.

    Groups data by `request_num` and takes the minimum latency per group.
    The minimum isolates the non-attention floor: the attention component
    varies with context length, so the minimum-latency sample at each
    batch size approximates the latency with minimal attention contribution.

    Fits the latency model:
        latency_floor(n) = max(threshold, b + k * n)

    Returns dict with other_threshold, other_latency_b, other_latency_k,
    and r_squared.
    """
    request_nums = np.array([d["request_num"] for d in data], dtype=np.float64)
    throughputs = np.array([d["throughput"] for d in data], dtype=np.float64)

    valid = (request_nums > 0) & (throughputs > 0)
    request_nums = request_nums[valid]
    throughputs = throughputs[valid]

    if len(request_nums) < 3:
        raise ValueError(f"Need at least 3 valid data points, got {len(request_nums)}.")

    latencies = request_nums / throughputs

    groups: dict[int, list[float]] = {}
    for rn, lat in zip(request_nums, latencies):
        groups.setdefault(int(rn), []).append(lat)

    grouped_rn = []
    grouped_lat = []
    for rn in sorted(groups):
        grouped_rn.append(rn)
        grouped_lat.append(min(groups[rn]))

    grouped_rn = np.array(grouped_rn, dtype=np.float64)
    grouped_lat = np.array(grouped_lat, dtype=np.float64)

    if len(grouped_rn) < 3:
        raise ValueError(
            f"Need at least 3 distinct request_num values, got {len(grouped_rn)}."
        )

    k0 = float(np.clip((grouped_lat[-1] - grouped_lat[0]) / max(1, grouped_rn[-1] - grouped_rn[0]), 1e-6, 0.01))
    b0 = float(np.clip(grouped_lat[0] * 0.8, 1e-6, 0.1))
    t0 = float(np.clip(min(grouped_lat) * 0.9, 1e-6, 0.1))

    try:
        popt, _ = curve_fit(
            _latency_model,
            grouped_rn,
            grouped_lat,
            p0=[k0, b0, t0],
            bounds=([0, 0, 0], [0.01, 0.1, 0.1]),
            maxfev=10000,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Stage 1 fitting failed: {exc}.") from exc

    k_opt, b_opt, threshold_opt = popt
    predicted = _latency_model(grouped_rn, k_opt, b_opt, threshold_opt)
    r2 = _r_squared(grouped_lat, predicted)

    dryrun_logger.info(
        f"Stage 1 fit: k={k_opt:.7f}, b={b_opt:.7f}, threshold={threshold_opt:.7f}, "
        f"R²={r2:.4f} (on {len(grouped_rn)} grouped points)."
    )

    return {
        "other_threshold": float(threshold_opt),
        "other_latency_b": float(b_opt),
        "other_latency_k": float(k_opt),
        "r_squared": r2,
    }


def fit_attention_latency(
    data: list[dict],
    other_k: float,
    other_b: float,
    other_threshold: float,
) -> dict:
    """
    Stage 2: Fit the attention latency component from residuals.

    Subtracts the fitted non-attention latency from observed total latency,
    then fits: attention_latency = attn_k * token_num + attn_b.

    Uses IQR-based outlier removal (2.5x threshold) for robust fitting.
    """
    request_nums = np.array([d["request_num"] for d in data], dtype=np.float64)
    token_nums = np.array([d["token_num"] for d in data], dtype=np.float64)
    throughputs = np.array([d["throughput"] for d in data], dtype=np.float64)

    valid = (request_nums > 0) & (throughputs > 0) & (token_nums > 0)
    request_nums = request_nums[valid]
    token_nums = token_nums[valid]
    throughputs = throughputs[valid]

    actual_latencies = request_nums / throughputs
    other_latencies = _latency_model(request_nums, other_k, other_b, other_threshold)
    attention_latencies = actual_latencies - other_latencies

    positive = attention_latencies > 0
    token_nums = token_nums[positive]
    attention_latencies = attention_latencies[positive]

    if len(token_nums) < 3:
        dryrun_logger.warning("Too few positive attention residuals, returning zero attention fit.")
        return {
            "attn_latency_k": 0.0,
            "attn_latency_b": 0.0,
            "r_squared": 0.0,
            "n_outliers_removed": 0,
        }

    ak0 = float(np.median(attention_latencies / token_nums))
    ab0 = float(np.min(attention_latencies) * 0.5)
    ak0 = max(ak0, 1e-10)
    ab0 = max(ab0, 0.0)

    try:
        popt_initial, _ = curve_fit(
            _attn_latency_model,
            token_nums,
            attention_latencies,
            p0=[ak0, ab0],
            bounds=([0, 0], [1e-3, 1.0]),
            maxfev=10000,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Stage 2 initial fitting failed: {exc}.") from exc

    predicted_initial = _attn_latency_model(token_nums, *popt_initial)
    residuals = np.abs(attention_latencies - predicted_initial)

    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    outlier_threshold = q3 + 2.5 * iqr

    inlier_mask = residuals <= outlier_threshold
    n_outliers = int(np.sum(~inlier_mask))
    filtered_tokens = token_nums[inlier_mask]
    filtered_attn = attention_latencies[inlier_mask]

    if len(filtered_tokens) < 3:
        dryrun_logger.warning(
            f"Only {len(filtered_tokens)} inliers after outlier removal, "
            f"using initial fit."
        )
        attn_k_opt, attn_b_opt = popt_initial
        r2 = _r_squared(attention_latencies, predicted_initial)
    else:
        try:
            popt, _ = curve_fit(
                _attn_latency_model,
                filtered_tokens,
                filtered_attn,
                p0=popt_initial,
                bounds=([0, 0], [1e-3, 1.0]),
                maxfev=10000,
            )
            attn_k_opt, attn_b_opt = popt
        except RuntimeError:
            attn_k_opt, attn_b_opt = popt_initial

        predicted = _attn_latency_model(filtered_tokens, attn_k_opt, attn_b_opt)
        r2 = _r_squared(filtered_attn, predicted)

    dryrun_logger.info(
        f"Stage 2 fit: attn_k={attn_k_opt:.10f}, attn_b={attn_b_opt:.10f}, "
        f"R²={r2:.4f}, outliers_removed={n_outliers}."
    )

    return {
        "attn_latency_k": float(attn_k_opt),
        "attn_latency_b": float(attn_b_opt),
        "r_squared": r2,
        "n_outliers_removed": n_outliers,
    }


def fit_cost_model(
    data: list[dict] | str | Path,
    key: str = "TP1_PP1",
) -> FitResult:
    """
    Run the full two-stage regression pipeline.

    Args:
        data: Either a list of dicts with keys (request_num, token_num,
            throughput), or a path to a CSV file with those columns.
        key: Configuration label (e.g. "TP1_PP1") for the output JSON.

    Returns:
        `FitResult` with fitted coefficients and quality metrics.
    """
    if isinstance(data, (str, Path)):
        data = _load_csv(data)

    stage1 = fit_other_latency(data)
    stage2 = fit_attention_latency(
        data,
        other_k=stage1["other_latency_k"],
        other_b=stage1["other_latency_b"],
        other_threshold=stage1["other_threshold"],
    )

    request_nums = np.array([d["request_num"] for d in data], dtype=np.float64)
    token_nums = np.array([d["token_num"] for d in data], dtype=np.float64)
    throughputs = np.array([d["throughput"] for d in data], dtype=np.float64)

    valid = (request_nums > 0) & (throughputs > 0)
    actual_latencies = request_nums[valid] / throughputs[valid]
    predicted_latencies = (
        stage2["attn_latency_b"]
        + stage2["attn_latency_k"] * token_nums[valid]
        + _latency_model(
            request_nums[valid],
            stage1["other_latency_k"],
            stage1["other_latency_b"],
            stage1["other_threshold"],
        )
    )

    rel_errors = np.abs((actual_latencies - predicted_latencies) / actual_latencies)
    mre = float(np.mean(rel_errors) * 100)
    rmse = float(np.sqrt(np.mean((actual_latencies - predicted_latencies) ** 2)))

    quality = FitQuality(
        other_r_squared=stage1["r_squared"],
        attn_r_squared=stage2["r_squared"],
        mre_percent=mre,
        rmse=rmse,
        n_samples=int(np.sum(valid)),
        n_outliers_removed=stage2["n_outliers_removed"],
    )

    return FitResult(
        attn_latency_b=stage2["attn_latency_b"],
        attn_latency_k=stage2["attn_latency_k"],
        other_threshold=stage1["other_threshold"],
        other_latency_b=stage1["other_latency_b"],
        other_latency_k=stage1["other_latency_k"],
        fit_quality=quality,
    )


def save_fit_result(
    result: FitResult,
    output_path: str | Path,
    key: str = "TP1_PP1",
    merge: bool = True,
) -> Path:
    """
    Save fit result to JSON in PSRLFitted-compatible format.

    Args:
        result: The fit result to save.
        output_path: Path to the output JSON file.
        key: Configuration label (e.g. "TP1_PP1").
        merge: If True and the file exists, merge into existing JSON
            (add/overwrite the key). If False, overwrite the entire file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if merge and output_path.exists():
        existing = json.loads(output_path.read_text())

    existing[key] = result.to_dict()
    output_path.write_text(json.dumps(existing, indent=2) + "\n")

    dryrun_logger.info(f"Cost model saved to {output_path} (key={key!r}).")
    return output_path
