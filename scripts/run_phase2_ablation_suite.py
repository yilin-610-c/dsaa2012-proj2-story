from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SINGLE_STORIES = {
    "smoke": ("14",),
    "full": ("01", "14"),
}
DEFAULT_DOUBLE_STORIES = {
    "smoke": ("07",),
    "full": ("06", "07"),
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_list(value: str | None, defaults: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(defaults)
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _story_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists() or candidate.suffix:
        return candidate
    story_id = value.zfill(2) if value.isdigit() else value
    return Path("test_set") / f"{story_id}.txt"


def _story_id(path: Path) -> str:
    return path.stem


def _native_args(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.storydiffusion_root:
        argv.extend(["--storydiffusion-root", str(args.storydiffusion_root)])
    if args.native_width is not None:
        argv.extend(["--native-width", str(args.native_width)])
    if args.native_height is not None:
        argv.extend(["--native-height", str(args.native_height)])
    if args.native_num_steps is not None:
        argv.extend(["--native-num-steps", str(args.native_num_steps)])
    if args.native_seed is not None:
        argv.extend(["--native-seed", str(args.native_seed)])
    if args.native_guidance_scale is not None:
        argv.extend(["--native-guidance-scale", str(args.native_guidance_scale)])
    if args.storydiffusion_prompt_mode != "current":
        argv.extend(["--storydiffusion-prompt-mode", args.storydiffusion_prompt_mode])
    return argv


def _build_command(args: argparse.Namespace, *, story: Path, experiment_name: str, route: str, suite_dir: Path) -> list[str]:
    command = [
        "python3",
        "scripts/run_auto_story_pipeline_modular.py",
        "--input",
        str(story),
        "--run-name",
        experiment_name,
        "--output-root",
        str(suite_dir),
        "--single-env",
        args.single_env,
        "--double-env",
        args.double_env,
    ]
    if route == "single_native":
        command.extend(["--single-route", "native_storydiffusion"])
    if route in {"single_native", "double_native"}:
        command.extend(_native_args(args))
    if args.dry_run:
        command.append("--dry-run")
    if args.resume:
        command.append("--probe-skip-if-exists")
    return command


def _is_complete(output_dir: Path) -> bool:
    return (output_dir / "manifest.json").exists() or (output_dir / "run_summary.json").exists()


def _write_manifest_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _experiments(args: argparse.Namespace) -> list[tuple[Path, str, str]]:
    single_values = _split_list(args.single_stories, DEFAULT_SINGLE_STORIES[args.suite])
    double_values = _split_list(args.double_stories, DEFAULT_DOUBLE_STORIES[args.suite])
    experiments: list[tuple[Path, str, str]] = []
    for value in single_values:
        story = _story_path(value)
        sid = _story_id(story)
        experiments.append((story, f"single_storygen_default_{sid}", "single_storygen"))
        experiments.append((story, f"single_native_storydiffusion_{sid}", "single_native"))
    for value in double_values:
        story = _story_path(value)
        sid = _story_id(story)
        experiments.append((story, f"double_native_storydiffusion_{sid}", "double_native"))
    return experiments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 2 story generation ablation suite.")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--single-stories", default=None, help="Comma-separated story ids or paths.")
    parser.add_argument("--double-stories", default=None, help="Comma-separated story ids or paths.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs_ablation"))
    parser.add_argument("--single-env", default="ipadapter")
    parser.add_argument("--double-env", default="storydiffusion")
    parser.add_argument("--storydiffusion-root", type=Path, default=None)
    parser.add_argument("--native-width", type=int, default=None)
    parser.add_argument("--native-height", type=int, default=None)
    parser.add_argument("--native-num-steps", type=int, default=None)
    parser.add_argument("--native-seed", type=int, default=None)
    parser.add_argument("--native-guidance-scale", type=float, default=None)
    parser.add_argument(
        "--storydiffusion-prompt-mode",
        choices=("current", "clean"),
        default="current",
        help="Forwarded to native StoryDiffusion routes; clean uses llm_assisted PromptSpec generation_prompt.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    suite_dir = (args.output_root / args.suite).resolve()
    manifest_path = suite_dir / "suite_manifest.jsonl"
    suite_dir.mkdir(parents=True, exist_ok=True)

    final_code = 0
    for story, experiment_name, route in _experiments(args):
        output_dir = suite_dir / experiment_name
        command = _build_command(args, story=story, experiment_name=experiment_name, route=route, suite_dir=suite_dir)
        start_time = _timestamp()
        entry = {
            "experiment_name": experiment_name,
            "story_path": str(story),
            "route": route,
            "command": command,
            "output_dir": str(output_dir),
            "start_time": start_time,
        }
        if args.resume and _is_complete(output_dir):
            entry.update({"status": "skipped", "end_time": _timestamp(), "returncode": 0})
            _write_manifest_line(manifest_path, entry)
            print(f"[suite] SKIP {experiment_name}: {output_dir}", flush=True)
            continue
        print("[suite] command:", " ".join(command), flush=True)
        result = subprocess.run(command)
        entry.update({"status": "passed" if result.returncode == 0 else "failed", "end_time": _timestamp(), "returncode": result.returncode})
        _write_manifest_line(manifest_path, entry)
        if result.returncode != 0:
            final_code = result.returncode
            if not args.continue_on_error:
                break
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
