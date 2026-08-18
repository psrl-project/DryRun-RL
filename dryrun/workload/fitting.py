"""Fit prompt-length distributions by tokenizing dataset prompts.

Reads a chat-formatted parquet dataset (e.g. dapo-math-17k.parquet, with a
`prompt` column of `list[{content, role}]` messages) and writes the
tokenized prompt lengths as a trace artifact via `dryrun.workload.trace`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .._logging import dryrun_logger
from .trace import write_length_trace

_CONTENT_KEY = "content"


def load_prompt_messages(
    dataset_path: str | Path,
    sample_size: int | None,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Read the `prompt` column from a parquet dataset, optionally subsampled."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(dataset_path, columns=["prompt"])
    messages = table.column("prompt").to_pylist()
    if sample_size is not None and sample_size < len(messages):
        rng = random.Random(seed)
        indices = rng.sample(range(len(messages)), sample_size)
        messages = [messages[i] for i in indices]
    return messages


def render_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """
    Render chat messages to a single prompt string.

    Falls back to concatenating message contents if the tokenizer has no
    chat template configured.

    Args:
        tokenizer (Any): A HuggingFace tokenizer instance.
        messages (list[dict[str, Any]]): Chat messages with `content`/`role` keys.

    Returns:
        str: The rendered prompt text.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        return "\n".join(message[_CONTENT_KEY] for message in messages)


def fit_prompt_lengths(
    dataset_path: str | Path,
    tokenizer_path: str,
    output_path: str | Path,
    sample_size: int | None = None,
    seed: int = 42,
    batch_size: int = 256,
) -> Path:
    """
    Tokenize dataset prompts and write their lengths as a trace artifact.

    Args:
        dataset_path (str | Path): Parquet dataset with a `prompt` column of
            `list[{content, role}]` chat messages.
        tokenizer_path (str): HuggingFace tokenizer name or local path.
        output_path (str | Path): Destination for the JSONL trace artifact.
        sample_size (int | None): If set, randomly subsample this many rows
            before tokenizing. If `None`, tokenize the entire dataset.
        seed (int): RNG seed for subsampling.
        batch_size (int): Prompts tokenized per `tokenizer()` call.

    Returns:
        Path: Path to the written trace artifact.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    dryrun_logger.info(f"Loading tokenizer from {tokenizer_path!r}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    dryrun_logger.info(f"Reading dataset from {dataset_path!r}...")
    messages_list = load_prompt_messages(dataset_path, sample_size, seed)
    dryrun_logger.info(f"Tokenizing {len(messages_list)} prompts...")

    lengths: list[int] = []
    for start in range(0, len(messages_list), batch_size):
        batch = messages_list[start : start + batch_size]
        texts = [render_prompt(tokenizer, messages) for messages in batch]
        encoded = tokenizer(texts, add_special_tokens=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    written = write_length_trace(
        lengths,
        output_path,
        metadata={
            "kind": "prompt_length",
            "dataset": str(dataset_path),
            "tokenizer": str(tokenizer_path),
            "seed": seed,
        },
    )
    dryrun_logger.info(f"Wrote {len(lengths)} prompt lengths to {written}.")
    return written
