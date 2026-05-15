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
    "custom": (),
}
DEFAULT_CUSTOM_STORIES = ("test_setA/19.txt", "test_setA/03.txt", "test_set/06.txt")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_list(value: str | None, defaults: tuple[str, ...] = ()) -> list[str]:
    if value is None:
        return list(defaults)
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_int_list(value: str | None) -> list[int | None]:
    if value is None:
        return [None]
    values = _split_list(value)
    if not values:
        return [None]
    return [int(item) for item in values]


def _story_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists() or candidate.suffix:
        return candidate
    story_id = value.zfill(2) if value.isdigit() else value
    return Path("test_set") / f"{story_id}.txt"


def _story_id(path: Path) -> str:
    return path.stem


def _story_slug(path: Path) -> str:
    no_suffix = path.with_suffix("")
    parts = list(no_suffix.parts)
    for marker in ("test_setA", "test_set"):
        if marker in parts:
            parts = parts[parts.index(marker) :]
            break
    else:
        parts = [no_suffix.stem]
    raw = "_".join(parts)
    return "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")


def _native_args(args: argparse.Namespace, native_id_length: int | None = None) -> list[str]:
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
    if native_id_length is not None:
        argv.extend(["--native-id-length", str(native_id_length)])
    if args.native_guidance_scale is not None:
        argv.extend(["--native-guidance-scale", str(args.native_guidance_scale)])
    if args.native_sd_type is not None:
        argv.extend(["--native-sd-type", args.native_sd_type])
    if args.save_identity_images:
        argv.append("--save-identity-images")
    if args.storydiffusion_prompt_mode != "current":
        argv.extend(["--storydiffusion-prompt-mode", args.storydiffusion_prompt_mode])
    return argv


def _build_command(
    args: argparse.Namespace,
    *,
    story: Path,
    experiment_name: str,
    route: str,
    suite_dir: Path,
    native_id_length: int | None = None,
) -> list[str]:
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
    if route == "native":
        command.extend(["--single-route", "native_storydiffusion"])
    if route == "storygen":
        command.extend(["--double-route", "storygen"])
    if route in {"single_native", "double_native", "native"}:
        command.extend(_native_args(args, native_id_length))
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


def _parse_id_length_overrides(values: list[str]) -> dict[str, list[int | None]]:
    overrides: dict[str, list[int | None]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--native-id-lengths-override must be STORY=1,2, got: {value}")
        story_key, lengths = value.split("=", 1)
        story_key = story_key.strip()
        if not story_key:
            raise ValueError(f"--native-id-lengths-override story key cannot be empty: {value}")
        overrides[story_key] = _split_int_list(lengths)
    return overrides


def _story_match_keys(story_value: str, story: Path) -> set[str]:
    keys = {story_value, str(story), story.name, story.stem}
    try:
        keys.add(str(story.resolve()))
    except OSError:
        pass
    return keys


def _native_lengths_for_story(
    args: argparse.Namespace,
    *,
    story_value: str,
    story: Path,
    default_lengths: list[int | None],
    overrides: dict[str, list[int | None]],
) -> list[int | None]:
    keys = _story_match_keys(story_value, story)
    for key, lengths in overrides.items():
        if key in keys:
            return lengths
    return default_lengths


def _flexible_experiments(args: argparse.Namespace) -> list[dict]:
    story_values = _split_list(args.stories, DEFAULT_CUSTOM_STORIES if args.suite == "custom" else ())
    if not story_values:
        story_values = _split_list(args.single_stories, DEFAULT_SINGLE_STORIES[args.suite])
        story_values.extend(_split_list(args.double_stories, DEFAULT_DOUBLE_STORIES[args.suite]))
    methods = _split_list(args.methods, ("storygen", "native"))
    invalid_methods = sorted(set(methods) - {"storygen", "native"})
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {', '.join(invalid_methods)}")
    default_lengths = _split_int_list(args.native_id_lengths)
    if args.native_id_lengths is None and args.native_id_length is not None:
        default_lengths = [args.native_id_length]
    overrides = _parse_id_length_overrides(args.native_id_lengths_override)
    experiments: list[dict] = []
    for value in story_values:
        story = _story_path(value)
        slug = _story_slug(story)
        if "storygen" in methods:
            experiments.append(
                {
                    "story": story,
                    "story_slug": slug,
                    "experiment_name": f"storygen_default_{slug}",
                    "route": "storygen",
                    "method": "storygen",
                    "native_id_length": None,
                }
            )
        if "native" in methods:
            lengths = _native_lengths_for_story(
                args,
                story_value=value,
                story=story,
                default_lengths=default_lengths,
                overrides=overrides,
            )
            for length in lengths:
                length_label = f"id{length}" if length is not None else "default"
                experiments.append(
                    {
                        "story": story,
                        "story_slug": slug,
                        "experiment_name": f"native_{args.storydiffusion_prompt_mode}_{length_label}_{slug}",
                        "route": "native",
                        "method": "native",
                        "native_id_length": length,
                    }
                )
    return experiments


def _legacy_experiments(args: argparse.Namespace) -> list[dict]:
    single_values = _split_list(args.single_stories, DEFAULT_SINGLE_STORIES[args.suite])
    double_values = _split_list(args.double_stories, DEFAULT_DOUBLE_STORIES[args.suite])
    experiments: list[dict] = []
    for value in single_values:
        story = _story_path(value)
        sid = _story_id(story)
        experiments.append(
            {
                "story": story,
                "story_slug": sid,
                "experiment_name": f"single_storygen_default_{sid}",
                "route": "single_storygen",
                "method": "storygen",
                "native_id_length": None,
            }
        )
        experiments.append(
            {
                "story": story,
                "story_slug": sid,
                "experiment_name": f"single_native_storydiffusion_{sid}",
                "route": "single_native",
                "method": "native",
                "native_id_length": args.native_id_length,
            }
        )
    for value in double_values:
        story = _story_path(value)
        sid = _story_id(story)
        experiments.append(
            {
                "story": story,
                "story_slug": sid,
                "experiment_name": f"double_native_storydiffusion_{sid}",
                "route": "double_native",
                "method": "native",
                "native_id_length": args.native_id_length,
            }
        )
    return experiments


def _experiments(args: argparse.Namespace) -> list[dict]:
    flexible_requested = (
        args.suite == "custom"
        or args.stories is not None
        or args.methods is not None
        or args.native_id_lengths is not None
        or bool(args.native_id_lengths_override)
    )
    if flexible_requested:
        return _flexible_experiments(args)
    return _legacy_experiments(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 2 story generation ablation suite.")
    parser.add_argument("--suite", choices=("smoke", "full", "custom"), default="smoke")
    parser.add_argument("--stories", default=None, help="Comma-separated mixed story ids or paths for custom suites.")
    parser.add_argument("--methods", default=None, help="Comma-separated methods: storygen,native.")
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
    parser.add_argument("--native-id-length", type=int, default=None)
    parser.add_argument("--native-id-lengths", default=None, help="Comma-separated native id lengths, e.g. 1,2.")
    parser.add_argument(
        "--native-id-lengths-override",
        action="append",
        default=[],
        metavar="STORY=1,2",
        help="Override native id lengths for one story key/path. Repeatable.",
    )
    parser.add_argument("--native-guidance-scale", type=float, default=None)
    parser.add_argument(
        "--native-sd-type",
        default=None,
        help="Forwarded to native StoryDiffusion routes as --sd-type, e.g. Unstable, RealVision, Juggernaut, SDXL.",
    )
    parser.add_argument("--save-identity-images", action="store_true")
    parser.add_argument(
        "--storydiffusion-prompt-mode",
        choices=("current", "clean", "clean_v2", "natural"),
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
    for experiment in _experiments(args):
        story = experiment["story"]
        experiment_name = experiment["experiment_name"]
        route = experiment["route"]
        native_id_length = experiment["native_id_length"]
        output_dir = suite_dir / experiment_name
        command = _build_command(
            args,
            story=story,
            experiment_name=experiment_name,
            route=route,
            suite_dir=suite_dir,
            native_id_length=native_id_length,
        )
        start_time = _timestamp()
        entry = {
            "experiment_name": experiment_name,
            "story_path": str(story),
            "story_slug": experiment["story_slug"],
            "method": experiment["method"],
            "route": route,
            "native_id_length": native_id_length,
            "storydiffusion_prompt_mode": args.storydiffusion_prompt_mode,
            "save_identity_images": args.save_identity_images,
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
