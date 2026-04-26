from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import run_experiment_matrix


SUITES = {
    "latest": ["llm_prompt_two_character_text2img"],
    "main": [
        "llm_prompt_text2img",
        "llm_prompt_ip_adapter_text2img",
        "llm_prompt_two_character_text2img",
    ],
}

PRESETS = {
    "quick": {
        "candidate_count": 1,
        "width": 512,
        "height": 512,
        "base_seed": 2026,
        "extra_set": ["model.num_inference_steps=4"],
    },
    "main": {
        "candidate_count": 3,
        "width": 768,
        "height": 768,
        "base_seed": 2026,
        "extra_set": ["model.num_inference_steps=4"],
    },
}


def _split_profiles(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return run_experiment_matrix.parse_profiles(value)


def _default_experiment_id(*, suite: str, preset: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"local_{suite}_{preset}_{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local LLM ablation suites through the experiment matrix runner")
    parser.add_argument("--suite", choices=sorted(SUITES), default="latest")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--profiles", type=_split_profiles, help="Comma-separated profile override. Excludes rule-based by default suites only.")
    parser.add_argument("--stories", default="test_set/*.txt")
    parser.add_argument("--experiment-id", type=lambda value: run_experiment_matrix.validate_name(value, field_name="experiment-id"))
    parser.add_argument("--output-root", help="Defaults to outputs_local/<experiment-id>.")
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--extra-set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip runs that already have run_summary.json.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed child run.")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser


def matrix_argv_from_args(args: argparse.Namespace) -> list[str]:
    experiment_id = args.experiment_id or _default_experiment_id(suite=args.suite, preset=args.preset)
    profiles = args.profiles or SUITES[args.suite]
    preset = PRESETS[args.preset]
    output_root = args.output_root or str(Path("outputs_local") / experiment_id)

    argv = [
        "--experiment-id",
        experiment_id,
        "--profiles",
        ",".join(profiles),
        "--stories",
        args.stories,
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--config",
        args.config,
        "--output-root",
        output_root,
    ]

    candidate_count = args.candidate_count if args.candidate_count is not None else preset["candidate_count"]
    width = args.width if args.width is not None else preset["width"]
    height = args.height if args.height is not None else preset["height"]
    base_seed = args.base_seed if args.base_seed is not None else preset["base_seed"]
    argv.extend(["--candidate-count", str(candidate_count)])
    argv.extend(["--width", str(width)])
    argv.extend(["--height", str(height)])
    argv.extend(["--base-seed", str(base_seed)])

    extra_set = [*preset["extra_set"], *args.extra_set]
    for override in extra_set:
        argv.extend(["--extra-set", override])

    if args.dry_run:
        argv.append("--dry-run")
    if not args.no_resume:
        argv.append("--resume")
    if not args.stop_on_error:
        argv.append("--continue-on-error")
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_experiment_matrix.main(matrix_argv_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
