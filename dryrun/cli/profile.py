"""Profiling CLI entrypoint (argparse-based, not Hydra)."""

from __future__ import annotations

import argparse
import logging
import sys

dryrun_logger = logging.getLogger("dryrun")


def _parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",")]


def cmd_vllm(args: argparse.Namespace) -> None:
    """Run vLLM profiling."""
    from ..profiling.schemas import VLLMProfileConfig  # noqa: PLC0415
    from ..profiling.vllm_profiler import run_vllm_profile  # noqa: PLC0415

    cfg = VLLMProfileConfig(
        model=args.model,
        tp=args.tp,
        pp=args.pp,
        batch_sizes=_parse_int_list(args.batch_sizes),
        context_lengths=_parse_int_list(args.ctx_lengths),
        output_len=args.output_len,
        port=args.port,
        warmup_requests=args.warmup,
        measure_requests=args.measure,
        timeout=args.timeout,
        extra_args=args.extra_args,
    )

    path = run_vllm_profile(cfg, args.output)
    print(f"Profiling data written to: {path}")


def cmd_megatron(args: argparse.Namespace) -> None:
    """Run Megatron training profiling."""
    from ..profiling.megatron_profiler import run_megatron_profile  # noqa: PLC0415
    from ..profiling.schemas import MegatronProfileConfig  # noqa: PLC0415

    cfg = MegatronProfileConfig(
        launch_cmd=args.launch_cmd,
        batch_sizes=_parse_int_list(args.batch_sizes),
        seq_lengths=_parse_int_list(args.seq_lengths),
        pp=args.pp,
        tp=args.tp,
        dp=args.dp,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        extra_args=args.extra_args,
    )

    path = run_megatron_profile(cfg, args.output)
    print(f"Profiling data written to: {path}")


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit a cost model from profiling data."""
    from ..profiling.fit import fit_cost_model, save_fit_result  # noqa: PLC0415

    result = fit_cost_model(args.input, key=args.key)

    print(f"Fit results for key={args.key!r}:")
    print(f"  attn_latency_b:  {result.attn_latency_b:.10f}")
    print(f"  attn_latency_k:  {result.attn_latency_k:.10f}")
    print(f"  other_threshold: {result.other_threshold:.7f}")
    print(f"  other_latency_b: {result.other_latency_b:.7f}")
    print(f"  other_latency_k: {result.other_latency_k:.7f}")
    print(f"\nQuality metrics:")
    print(f"  Other R²:  {result.fit_quality.other_r_squared:.4f}")
    print(f"  Attn R²:   {result.fit_quality.attn_r_squared:.4f}")
    print(f"  MRE:       {result.fit_quality.mre_percent:.2f}%")
    print(f"  RMSE:      {result.fit_quality.rmse:.6f}")
    print(f"  Samples:   {result.fit_quality.n_samples}")
    print(f"  Outliers:  {result.fit_quality.n_outliers_removed}")

    path = save_fit_result(result, args.output, key=args.key, merge=not args.no_merge)
    print(f"\nCost model saved to: {path}")


def main() -> None:
    """Profiling CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="dryrun-profile",
        description="DryRun-RL Profiling: benchmark inference/training and fit cost models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- vllm subcommand ---
    p_vllm = subparsers.add_parser("vllm", help="Profile vLLM decode/prefill latency.")
    p_vllm.add_argument("--model", required=True, help="Model name or path.")
    p_vllm.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    p_vllm.add_argument("--pp", type=int, default=1, help="Pipeline parallel size.")
    p_vllm.add_argument(
        "--batch-sizes", default="1,2,4,8,16,32",
        help="Comma-separated concurrent request counts to sweep.",
    )
    p_vllm.add_argument(
        "--ctx-lengths", default="128,256,512,1024,2048",
        help="Comma-separated prompt lengths to sweep.",
    )
    p_vllm.add_argument("--output-len", type=int, default=128, help="Output tokens per request.")
    p_vllm.add_argument("--port", type=int, default=8000, help="Server port.")
    p_vllm.add_argument("--warmup", type=int, default=2, help="Warmup rounds per sweep point.")
    p_vllm.add_argument("--measure", type=int, default=10, help="Measurement rounds per point.")
    p_vllm.add_argument("--timeout", type=int, default=300, help="Server startup timeout (s).")
    p_vllm.add_argument("--output", default="profiles/vllm_decode.csv", help="Output CSV path.")
    p_vllm.add_argument("extra_args", nargs="*", help="Extra args passed to vLLM server.")
    p_vllm.set_defaults(func=cmd_vllm)

    # --- megatron subcommand ---
    p_mega = subparsers.add_parser("megatron", help="Profile Megatron training step times.")
    p_mega.add_argument("--launch-cmd", required=True, help="Base training launch command.")
    p_mega.add_argument(
        "--batch-sizes", default="1,2,4,8",
        help="Comma-separated micro batch sizes to sweep.",
    )
    p_mega.add_argument(
        "--seq-lengths", default="512,1024,2048",
        help="Comma-separated sequence lengths to sweep.",
    )
    p_mega.add_argument("--pp", type=int, default=1, help="Pipeline parallel size.")
    p_mega.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    p_mega.add_argument("--dp", type=int, default=1, help="Data parallel size.")
    p_mega.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps to discard.")
    p_mega.add_argument("--measure-steps", type=int, default=20, help="Steps to measure.")
    p_mega.add_argument("--output", default="profiles/megatron_train.csv", help="Output CSV path.")
    p_mega.add_argument("extra_args", nargs="*", help="Extra args passed to training command.")
    p_mega.set_defaults(func=cmd_megatron)

    # --- fit subcommand ---
    p_fit = subparsers.add_parser(
        "fit",
        help="Fit a cost model from profiling CSV data.",
    )
    p_fit.add_argument("--input", required=True, help="Input CSV path.")
    p_fit.add_argument("--key", default="TP1_PP1", help="Config key in output JSON.")
    p_fit.add_argument("--output", default="cost_models/fitted.json", help="Output JSON path.")
    p_fit.add_argument(
        "--no-merge", action="store_true",
        help="Overwrite output file instead of merging into existing JSON.",
    )
    p_fit.set_defaults(func=cmd_fit)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
