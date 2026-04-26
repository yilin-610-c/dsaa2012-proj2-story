from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _update_scene_result(scene_result: dict[str, Any], new_prompt: str) -> dict[str, Any]:
    scene_result = dict(scene_result)
    selection = dict(scene_result.get("selection") or {})
    selection["prompt"] = new_prompt
    scene_result["selection"] = selection

    candidates = scene_result.get("candidates") or []
    updated_candidates = []
    for candidate in candidates:
        candidate = dict(candidate)
        candidate["prompt"] = new_prompt
        updated_candidates.append(candidate)
    scene_result["candidates"] = updated_candidates
    return scene_result


def _build_cli_command(run_root: Path, scene_result_path: Path) -> list[str]:
    scene_id = scene_result_path.parent.name
    input_story = run_root.parent.parent / "test_set" / "06.txt"
    if not input_story.exists():
        input_story = Path("test_set/06.txt")
    return [
        "conda",
        "activate",
        "ipadapter",
        "&&",
        "PYTHONPATH=src",
        "python",
        "-m",
        "storygen.cli",
        "--profile",
        "cloud_storydiffusion_debug",
        "--input",
        str(input_story),
        "--run-name",
        f"replay_{scene_id}",
        "--set",
        "prompt.pipeline=rule_based",
        "--set",
        "generation.identity_conditioning.enabled=false",
        "--set",
        "model.consistent_attention.enabled=false",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch a scene_result prompt and optionally replay generation")
    parser.add_argument("scene_result_path", help="Path to scene_result.json")
    parser.add_argument("new_prompt", help="Replacement prompt text")
    parser.add_argument("--backup", action="store_true", help="Write a .bak copy before modifying")
    parser.add_argument("--replay", action="store_true", help="Print a replay command after patching")
    args = parser.parse_args()

    scene_result_path = Path(args.scene_result_path).expanduser().resolve()
    scene_result = _load_json(scene_result_path)

    if args.backup:
        backup_path = scene_result_path.with_suffix(scene_result_path.suffix + ".bak")
        _save_json(backup_path, scene_result)
        print(f"Backup written to {backup_path}")

    updated = _update_scene_result(scene_result, args.new_prompt)
    _save_json(scene_result_path, updated)
    print(f"Updated prompt in {scene_result_path}")

    if args.replay:
        run_root = scene_result_path.parent.parent.parent.parent
        cmd = _build_cli_command(run_root, scene_result_path)
        print("Replay command:")
        print(" ".join(cmd))


if __name__ == "__main__":
    main()
