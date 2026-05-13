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
from pathlib import Path
from typing import Any


ENTITY_PATTERN = re.compile(r"<([^<>]+)>")
SCENE_PATTERN = re.compile(r"^\[(SCENE-\d+)\]\s*(.*)$", re.DOTALL)

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
    return [
        "conda",
        "run",
        "-n",
        args.double_env,
        "python",
        "storydiffusion_gradio_probe/run_probe.py",
        "--config",
        str(probe_config),
    ]


def double_run_directory(args: argparse.Namespace) -> Path:
    parent = Path(args.output_root) if args.output_root else Path("outputs")
    return (parent / args.run_name).resolve()


def build_probe_autogen_and_run_argv(
    args: argparse.Namespace, *, story_path: Path, run_dir: Path
) -> list[str]:
    return [
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
        "--run",
    ]


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
        default="cloud_storydiffusion_debug",
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

    use_native_probe = route == "double" or (route == "single" and args.single_route == "native_storydiffusion")

    if route == "single" and not use_native_probe:
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
                print(f"[auto/modular] route=double entities={n_entities} -> SKIP (manifest exists): {manifest}")
                return 0

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
