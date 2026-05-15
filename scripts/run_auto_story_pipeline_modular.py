from __future__ import annotations

"""
Opt-in auto-router: same routing as run_auto_story_pipeline.py, but
- single path: enables modular rule prompt stack (prompt.builder=modular, backend sdxl) via --set
- double path: run_test_set with --prompt-builder modular --prompt-modular-backend storydiffusion

The legacy script is unchanged; use this wrapper for StoryDiffusion-friendly prompts.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ENTITY_PATTERN = re.compile(r"<([^<>]+)>")
SCENE_PATTERN = re.compile(r"^\[(SCENE-\d+)\]\s*(.*)$", re.DOTALL)
DEFAULT_STORYDIFFUSION_ROOT = Path(__file__).resolve().parents[1].parent
STORYDIFFUSION_APP = "gradio_app_sdxl_specific_id_low_vram.py"

DEFAULT_SINGLE_SET_OVERRIDES: tuple[str, ...] = (
    "generation.identity_conditioning.scale=0.3",
    "prompt.builder=modular",
    "prompt.modular.backend=sdxl",
)


def _parse_set_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--set override must be KEY=VALUE, got: {value}")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"--set override key cannot be empty: {value}")
    try:
        import yaml

        parsed = yaml.safe_load(raw)
    except Exception:
        parsed = raw
    return key, parsed


def classify_story(story_path: Path) -> tuple[str, int]:
    raw = story_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw.split("[SEP]") if block.strip()]
    unique_entities: list[str] = []
    for block in blocks:
        match = SCENE_PATTERN.match(block)
        if not match:
            continue
        _, scene_body = match.groups()
        entities = [ent.strip() for ent in ENTITY_PATTERN.findall(scene_body) if ent.strip()]
        for ent in dict.fromkeys(entities):
            if ent not in unique_entities:
                unique_entities.append(ent)
    unique_count = len(unique_entities)
    if unique_count >= 2:
        return "double", unique_count
    return "single", unique_count


def _merged_single_set_overrides(user_overrides: list[str]) -> list[str]:
    user_keys = {_parse_set_override(s)[0] for s in user_overrides}
    merged: list[str] = []
    for default_s in DEFAULT_SINGLE_SET_OVERRIDES:
        key, _ = _parse_set_override(default_s)
        if key not in user_keys:
            merged.append(default_s)
    merged.extend(user_overrides)
    return merged


def build_storygen_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "conda",
        "run",
        "-n",
        args.single_env,
        "python",
        "-m",
        "storygen.cli",
        "--config",
        args.config,
        "--profile",
        args.single_profile,
        "--input",
        str(args.input),
        "--run-name",
        args.run_name,
    ]
    if args.output_root:
        argv.extend(["--set", f"runtime.output_root={args.output_root}"])
    for override in _merged_single_set_overrides(args.set_overrides):
        key, value = _parse_set_override(override)
        argv.extend(["--set", f"{key}={value}"])
    return argv


def build_probe_argv(args: argparse.Namespace, *, probe_config: Path) -> list[str]:
    argv = [
        "conda",
        "run",
        "-n",
        args.double_env,
        "python",
        "storydiffusion_gradio_probe/run_probe.py",
        "--config",
        str(probe_config),
    ]
    if args.storydiffusion_root:
        argv.extend(["--storydiffusion-root", str(args.storydiffusion_root)])
    return argv


def double_run_directory(args: argparse.Namespace) -> Path:
    parent = Path(args.output_root) if args.output_root else Path("outputs")
    return (parent / args.run_name).resolve()


def build_probe_autogen_and_run_argv(
    args: argparse.Namespace, *, story_path: Path, run_dir: Path
) -> list[str]:
    argv = [
        "conda",
        "run",
        "-n",
        args.double_env,
        "python",
        "storydiffusion_gradio_probe/run_test_set.py",
        "--input-dir",
        str(story_path.parent.resolve()),
        "--config-dir",
        str(run_dir),
        "--output-root",
        str(run_dir),
        "--unwrap-output-dir",
        "--glob",
        str(story_path.name),
        "--limit",
        "1",
        "--prompt-builder",
        "modular",
        "--prompt-modular-backend",
        "storydiffusion",
        "--storydiffusion-prompt-mode",
        args.storydiffusion_prompt_mode,
        "--run",
    ]
    if args.storydiffusion_root:
        argv.extend(["--storydiffusion-root", str(args.storydiffusion_root)])
    if args.native_width is not None:
        argv.extend(["--width", str(args.native_width)])
    if args.native_height is not None:
        argv.extend(["--height", str(args.native_height)])
    if args.native_num_steps is not None:
        argv.extend(["--num-steps", str(args.native_num_steps)])
    if args.native_seed is not None:
        argv.extend(["--seed", str(args.native_seed)])
    if args.native_guidance_scale is not None:
        argv.extend(["--guidance-scale", str(args.native_guidance_scale)])
    if args.native_id_length is not None:
        argv.extend(["--id-length", str(args.native_id_length)])
    if args.native_sd_type is not None:
        argv.extend(["--sd-type", args.native_sd_type])
    if args.save_identity_images:
        argv.append("--save-identity-images")
    return argv


def storydiffusion_root(args: argparse.Namespace) -> Path:
    if args.storydiffusion_root:
        return Path(args.storydiffusion_root).expanduser().resolve()
    return DEFAULT_STORYDIFFUSION_ROOT.resolve()


def validate_native_storydiffusion_root(args: argparse.Namespace) -> bool:
    root = storydiffusion_root(args)
    app_path = root / STORYDIFFUSION_APP
    if app_path.exists():
        return True
    print(
        "Native StoryDiffusion route requires the external official StoryDiffusion repo.\n"
        f"Expected Gradio app file: {app_path}\n"
        "Clone https://github.com/HVision-NKU/StoryDiffusion and pass "
        "--storydiffusion-root /path/to/StoryDiffusion.",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-route stories using modular prompt stack (SDXL single / StoryDiffusion double)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Story input file, e.g. test_set/14.txt")
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run directory name under outputs/ (single: storygen run; double: probe output + config yaml).",
    )
    parser.add_argument("--config", default="configs/base.yaml", help="storygen base config path")
    parser.add_argument("--output-root", default=None, help="storygen output root (passed as runtime.output_root)")
    parser.add_argument("--single-env", default="ipadapter", help="Conda env name for the IP-Adapter storygen path")
    parser.add_argument(
        "--single-profile",
        default="cloud_anchor_ipadapter_story",
        help="storygen runtime profile for the single-character path",
    )
    parser.add_argument(
        "--single-route",
        choices=("storygen", "native_storydiffusion"),
        default="storygen",
        help="Single-character execution route. Defaults to the existing storygen path.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Forwarded to storygen.cli (repeatable). Modular defaults include prompt.builder=modular.",
    )
    parser.add_argument("--double-env", default="storydiffusion", help="Conda env for StoryDiffusion probe path")
    parser.add_argument(
        "--double-route",
        choices=("native_storydiffusion", "storygen"),
        default="native_storydiffusion",
        help="Double-character execution route. Defaults to the existing native StoryDiffusion probe path.",
    )
    parser.add_argument(
        "--storydiffusion-root",
        type=Path,
        default=None,
        help="Path to the external official StoryDiffusion repo for native probe routes.",
    )
    parser.add_argument("--native-width", type=int, default=None, help="Forwarded to native StoryDiffusion probe as --width.")
    parser.add_argument("--native-height", type=int, default=None, help="Forwarded to native StoryDiffusion probe as --height.")
    parser.add_argument("--native-num-steps", type=int, default=None, help="Forwarded to native StoryDiffusion probe as --num-steps.")
    parser.add_argument("--native-seed", type=int, default=None, help="Forwarded to native StoryDiffusion probe as --seed.")
    parser.add_argument(
        "--native-id-length",
        type=int,
        default=None,
        help=(
            "Forwarded to native StoryDiffusion probe as --id-length. In clean_v2 this means "
            "identity reference prompts per character; the generated YAML records the total front-loaded identity prompt count."
        ),
    )
    parser.add_argument(
        "--native-guidance-scale",
        type=float,
        default=None,
        help="Forwarded to native StoryDiffusion probe as --guidance-scale.",
    )
    parser.add_argument(
        "--native-sd-type",
        default=None,
        help="Forwarded to native StoryDiffusion probe as --sd-type, e.g. Unstable, RealVision, Juggernaut, SDXL.",
    )
    parser.add_argument(
        "--save-identity-images",
        action="store_true",
        help="Forwarded to native StoryDiffusion probe to save skipped identity reference images.",
    )
    parser.add_argument(
        "--storydiffusion-prompt-mode",
        choices=("current", "clean", "clean_v2", "natural"),
        default="current",
        help="Forwarded to native StoryDiffusion probe. Default preserves current prompt rendering.",
    )
    parser.add_argument("--probe-config", type=Path, default=None, help="Explicit probe config YAML/JSON")
    parser.add_argument(
        "--probe-config-dir",
        type=Path,
        default=Path("outputs/storydiffusion_gradio_probe/configs"),
        help="Directory for default probe configs",
    )
    parser.add_argument("--probe-skip-if-exists", action="store_true", help="Skip double run if manifest exists")
    parser.add_argument("--dry-run", action="store_true", help="Print selected command and exit")

    args = parser.parse_args(argv)
    story_path = args.input.resolve()
    if not story_path.exists():
        raise FileNotFoundError(f"Story input does not exist: {story_path}")

    route, n_entities = classify_story(story_path)

    use_native_probe = (
        (route == "double" and args.double_route == "native_storydiffusion")
        or (route == "single" and args.single_route == "native_storydiffusion")
    )

    if not use_native_probe:
        chosen = build_storygen_argv(args)
        env = {"PYTHONPATH": "src"}
    else:
        run_dir = double_run_directory(args)
        if args.probe_skip_if_exists:
            if args.probe_config is not None:
                manifest = Path("outputs/storydiffusion_gradio_probe/test_set") / story_path.stem / "manifest.json"
            else:
                manifest = run_dir / "manifest.json"
            if manifest.exists():
                print(f"[auto/modular] route={route} entities={n_entities} -> SKIP (manifest exists): {manifest}")
                return 0

        if not args.dry_run and not validate_native_storydiffusion_root(args):
            return 2

        if args.probe_config is not None:
            probe_config = args.probe_config.resolve()
            if not probe_config.exists():
                raise FileNotFoundError(f"Probe config not found: {probe_config}")
            chosen = build_probe_argv(args, probe_config=probe_config)
        else:
            chosen = build_probe_autogen_and_run_argv(args, story_path=story_path, run_dir=run_dir)
        env = {}

    print(f"[auto/modular] route={route} entities={n_entities}")
    if env:
        print("[auto/modular] command:", "PYTHONPATH=src " + " ".join(str(x) for x in chosen))
    else:
        print("[auto/modular] command:", " ".join(str(x) for x in chosen))
    if args.dry_run:
        return 0

    process_env = os.environ.copy()
    process_env.update(env)
    result = subprocess.run(chosen, env=process_env)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
