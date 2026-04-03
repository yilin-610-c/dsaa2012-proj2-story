from __future__ import annotations

import argparse
from pathlib import Path

from storygen.config import resolve_config
from storygen.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal story image generation baseline")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument("--profile", default="smoke_test", choices=["smoke_test", "demo_run"])
    parser.add_argument("--input", dest="input_path", help="Story input file")
    parser.add_argument("--output-root", help="Output root directory")
    parser.add_argument("--run-name", help="Explicit run name")
    parser.add_argument("--num-candidates", type=int, help="Override candidate count")
    parser.add_argument("--base-seed", type=int, help="Override base seed")
    parser.add_argument("--device", help="Override runtime device")
    parser.add_argument("--model-id", help="Override model id")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    overrides = {
        "runtime.input_path": args.input_path,
        "runtime.output_root": args.output_root,
        "runtime.run_name": args.run_name,
        "runtime.device": args.device,
        "runtime.repo_root": str(Path(".").resolve()),
        "generation.candidate_count": args.num_candidates,
        "generation.base_seed": args.base_seed,
        "model.model_id": args.model_id,
    }
    config = resolve_config(args.config, args.profile, overrides=overrides)
    summary = run_pipeline(config)
    print(f"Run complete: {summary.run_directory}")


if __name__ == "__main__":
    main()
