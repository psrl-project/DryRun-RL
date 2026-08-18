"""Tests for dataset-based prompt/output length fitting and trace artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dryrun.workload.distributions import from_trace
from dryrun.workload.trace import compute_length_stats, write_length_trace

REAL_QWEN = Path("/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-8B")


def _write_chat_parquet(path: Path, prompts: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "prompt": pa.array(
                [[{"content": text, "role": "user"}] for text in prompts],
                type=pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())])),
            ),
        }
    )
    pq.write_table(table, path)


class TestComputeLengthStats:
    def test_basic_stats(self):
        stats = compute_length_stats([10, 20, 30, 40, 50], n_bins=5)
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["mean"] == 30.0
        assert len(stats["histogram"]["counts"]) == 5
        assert sum(stats["histogram"]["counts"]) == 5


class TestWriteLengthTrace:
    def test_roundtrip_with_from_trace(self, tmp_path):
        path = write_length_trace([5, 10, 15], tmp_path / "trace.jsonl", metadata={"kind": "test"})
        assert from_trace(str(path)) == [5, 10, 15]

    def test_header_contains_metadata_and_stats(self, tmp_path):
        path = write_length_trace([1, 2, 3], tmp_path / "trace.jsonl", metadata={"dataset": "foo"})
        with path.open() as fh:
            header = json.loads(fh.readline())
        assert header["_meta"] is True
        assert header["dataset"] == "foo"
        assert header["n_samples"] == 3
        assert "stats" in header


@pytest.mark.skipif(not REAL_QWEN.exists(), reason="Qwen3-8B checkpoint is not available.")
class TestFitPromptLengths:
    def test_fit_prompt_lengths_from_parquet(self, tmp_path):
        from dryrun.workload.fitting import fit_prompt_lengths

        dataset_path = tmp_path / "dataset.parquet"
        _write_chat_parquet(dataset_path, [f"Question number {i} about math." for i in range(10)])

        written = fit_prompt_lengths(
            dataset_path=dataset_path,
            tokenizer_path=str(REAL_QWEN),
            output_path=tmp_path / "prompt_lengths.jsonl",
            seed=42,
        )
        lengths = from_trace(str(written))
        assert len(lengths) == 10
        assert all(length > 0 for length in lengths)

    def test_fit_prompt_lengths_with_subsample(self, tmp_path):
        from dryrun.workload.fitting import fit_prompt_lengths

        dataset_path = tmp_path / "dataset.parquet"
        _write_chat_parquet(dataset_path, [f"Question number {i}." for i in range(20)])

        written = fit_prompt_lengths(
            dataset_path=dataset_path,
            tokenizer_path=str(REAL_QWEN),
            output_path=tmp_path / "prompt_lengths.jsonl",
            sample_size=5,
            seed=42,
        )
        assert len(from_trace(str(written))) == 5
