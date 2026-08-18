"""Length trace artifact I/O.

A trace artifact is a JSONL file: a `_meta` provenance and summary-stats
header line, followed by one `{"length": N}` record per sample. This is the
format written by `dryrun-workload fit-prompt`/`fit-output` and read back by
`dryrun.workload.distributions.from_trace`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_length_stats(lengths: list[int], n_bins: int = 20) -> dict[str, Any]:
    """
    Compute summary statistics and a histogram for a list of lengths.

    Args:
        lengths (list[int]): Token lengths to summarize.
        n_bins (int): Number of histogram bins.

    Returns:
        dict[str, Any]: `mean`, `median`, `p95`, `min`, `max`, and a
        `histogram` with `bin_edges`/`counts`.
    """
    array = np.asarray(lengths, dtype=np.int64)
    counts, bin_edges = np.histogram(array, bins=n_bins)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": int(array.min()),
        "max": int(array.max()),
        "histogram": {
            "bin_edges": [float(edge) for edge in bin_edges],
            "counts": [int(count) for count in counts],
        },
    }


def write_length_trace(
    lengths: list[int],
    output_path: str | Path,
    metadata: dict[str, Any],
    n_bins: int = 20,
) -> Path:
    """
    Write a length trace artifact.

    Args:
        lengths (list[int]): Token lengths to write, one per sampled request.
        output_path (str | Path): Destination JSONL path. Parent directories
            are created as needed.
        metadata (dict[str, Any]): Provenance fields merged into the `_meta`
            header (e.g. `dataset`, `tokenizer`, `seed`).
        n_bins (int): Number of histogram bins for the summary stats.

    Returns:
        Path: The path written to.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "_meta": True,
        "n_samples": len(lengths),
        "stats": compute_length_stats(lengths, n_bins=n_bins),
        **metadata,
    }
    with output.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for length in lengths:
            fh.write(json.dumps({"length": int(length)}) + "\n")
    return output
