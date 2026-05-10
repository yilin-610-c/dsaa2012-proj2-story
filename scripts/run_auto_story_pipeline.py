from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ENTITY_PATTERN = re.compile(r"<([^<>]+)>")
# Must match lines like [SCENE-1] ... (same as storygen.parser)
SCENE_PATTERN = re.compile(r"^\[(SCENE-\d+)\]\s*(.*)$", re.DOTALL)


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
    """
    Classify story as 'single' or 'double' from story text.

    Counts distinct named entities in angle brackets <Name> across all [SCENE-n] blocks
    (same convention as test_set). Two or more distinct names => double pipeline
    (StoryDiffusion probe); otherwise single (storygen + ipadapter env).
    """
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
    for override in args.set_overrides:
        key, value = _parse_set_override(override)
        argv.extend(["--set", f"{key}={value}"])
    return argv


def _default_probe_config_path(config_dir: Path, story_path: Path) -> Path:
    # test_set/14.txt -> outputs/storydiffusion_gradio_probe/configs/14.yaml
    return config_dir / f"{story_path.stem}.yaml"


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
    return argv


def double_run_directory(args: argparse.Namespace) -> Path:
    """Same layout as storygen: <output_parent>/<run-name>/ (default parent is outputs/)."""
    parent = Path(args.output_root) if args.output_root else Path("outputs")
    return (parent / args.run_name).resolve()


def build_probe_autogen_and_run_argv(
    args: argparse.Namespace, *, story_path: Path, run_dir: Path
) -> list[str]:
    """
    Auto-generate a probe config from the story file via run_test_set.py and run it.

    This ensures the story is processed through the repo prompt pipeline (PromptBuilder)
    before invoking the original StoryDiffusion Gradio probe runner.

    Images and manifest are written under ``run_dir`` (same convention as single-character
    storygen: outputs/<run-name>/).
    """
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
        "--run",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-route single vs multi-character stories to the desired pipeline."
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
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Forwarded to storygen.cli (repeatable). Example: --set generation.identity_conditioning.scale=0.3",
    )

    parser.add_argument(
        "--double-env",
        default="storydiffusion",
        help="Conda env name for the StoryDiffusion probe path",
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=None,
        help="Explicit probe config YAML/JSON. Defaults to outputs/storydiffusion_gradio_probe/configs/<stem>.yaml",
    )
    parser.add_argument(
        "--probe-config-dir",
        type=Path,
        default=Path("outputs/storydiffusion_gradio_probe/configs"),
        help="Directory for default probe configs",
    )
    parser.add_argument(
        "--probe-skip-if-exists",
        action="store_true",
        help=(
            "Skip double pipeline if manifest.json already exists in the run directory "
            "(<output-root>/<run-name>/ when using auto probe; legacy path if --probe-config is set)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected command and exit")

    args = parser.parse_args(argv)
    story_path = args.input.resolve()
    if not story_path.exists():
        raise FileNotFoundError(f"Story input does not exist: {story_path}")

    route, n_entities = classify_story(story_path)

    if route == "single":
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
                print(f"[auto] route=double entities={n_entities} -> SKIP (manifest exists): {manifest}")
                return 0

        if args.probe_config is not None:
            probe_config = args.probe_config.resolve()
            if not probe_config.exists():
                raise FileNotFoundError(f"Probe config not found: {probe_config}")
            chosen = build_probe_argv(args, probe_config=probe_config)
        else:
            chosen = build_probe_autogen_and_run_argv(args, story_path=story_path, run_dir=run_dir)
        env = {}

    print(f"[auto] route={route} entities={n_entities}")
    if route == "single":
        print("[auto] command:", "PYTHONPATH=src " + " ".join(str(x) for x in chosen))
    else:
        print("[auto] command:", " ".join(str(x) for x in chosen))
    if args.dry_run:
        return 0

    process_env = os.environ.copy()
    process_env.update(env)

    result = subprocess.run(chosen, env=process_env)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

