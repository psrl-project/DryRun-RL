"""CLI for fitting workload length distributions from real datasets and models."""

from __future__ import annotations

import argparse

from .._logging import dryrun_logger, setup_logging


def cmd_fit_prompt(args: argparse.Namespace) -> None:
    """Tokenize dataset prompts and write a prompt-length trace artifact."""
    from ..workload.fitting import fit_prompt_lengths  # noqa: PLC0415

    path = fit_prompt_lengths(
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    dryrun_logger.info(f"Prompt length trace written to {path}.")


def cmd_fit_output(args: argparse.Namespace) -> None:
    """Serve the model, run sampled prompts through it, and write an output-length trace."""
    from ..workload.inference import collect_output_lengths  # noqa: PLC0415

    extra_serve_args = dict(item.split("=", 1) for item in args.serve_arg)
    path = collect_output_lengths(
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        model_path=args.model,
        output_path=args.output,
        tp=args.tp,
        sample_size=args.sample_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
        concurrency=args.concurrency,
        port=args.port,
        server_ready_timeout=args.server_ready_timeout,
        extra_serve_args=extra_serve_args,
    )
    dryrun_logger.info(f"Output length trace written to {path}.")


def build_parser() -> argparse.ArgumentParser:
    """Build the `dryrun-workload` argument parser."""
    parser = argparse.ArgumentParser(
        prog="dryrun-workload",
        description="Fit prompt/output length distributions from real datasets and models.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    fit_prompt = subs.add_parser(
        "fit-prompt",
        help="Tokenize dataset prompts into a prompt-length trace artifact.",
    )
    fit_prompt.add_argument("--dataset", required=True, help="Parquet dataset path.")
    fit_prompt.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer name or local path.")
    fit_prompt.add_argument("--output", required=True, help="Output JSONL trace path.")
    fit_prompt.add_argument("--sample-size", type=int, default=None, help="Subsample this many rows.")
    fit_prompt.add_argument("--seed", type=int, default=42)
    fit_prompt.add_argument("--batch-size", type=int, default=256, help="Tokenizer batch size.")
    fit_prompt.set_defaults(func=cmd_fit_prompt)

    fit_output = subs.add_parser(
        "fit-output",
        help="Serve a model and record real output lengths into a trace artifact.",
    )
    fit_output.add_argument("--dataset", required=True, help="Parquet dataset path.")
    fit_output.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer name or local path.")
    fit_output.add_argument("--model", required=True, help="Model path passed to `vllm serve`.")
    fit_output.add_argument("--output", required=True, help="Output JSONL trace path.")
    fit_output.add_argument("--tp", type=int, default=1, help="Tensor-parallel size.")
    fit_output.add_argument("--sample-size", type=int, default=200, help="Number of prompts to sample.")
    fit_output.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens per request.")
    fit_output.add_argument("--seed", type=int, default=42)
    fit_output.add_argument("--concurrency", type=int, default=16, help="Max in-flight requests.")
    fit_output.add_argument("--port", type=int, default=8000, help="Local port for the vLLM server.")
    fit_output.add_argument(
        "--server-ready-timeout",
        type=int,
        default=900,
        help="Seconds to wait for the vLLM server to become ready.",
    )
    fit_output.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra `vllm serve` flag, e.g. --serve-arg gpu_memory_utilization=0.9. Repeatable.",
    )
    fit_output.set_defaults(func=cmd_fit_output)

    return parser


def main() -> None:
    """Run a workload fitting command."""
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
