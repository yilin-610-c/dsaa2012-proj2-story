from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from storygen.types import RunContext, RunSummary


def create_run_context(output_root: str | Path, run_name: str) -> RunContext:
    output_root_path = Path(output_root)
    run_directory = output_root_path / run_name
    scenes_directory = run_directory / "scenes"
    scenes_directory.mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_name=run_name,
        output_root=output_root_path,
        run_directory=run_directory,
        scenes_directory=scenes_directory,
    )


def get_timestamp_string() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def scene_directory(run_context: RunContext, scene_index: int) -> Path:
    return run_context.scenes_directory / f"scene_{scene_index + 1:03d}"


def save_candidate_image(candidate_image: Any, run_context: RunContext, scene_index: int, candidate_index: int, seed: int) -> str:
    scene_dir = scene_directory(run_context, scene_index)
    candidates_dir = scene_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    output_path = candidates_dir / f"cand_{candidate_index:03d}_seed_{seed}.png"
    candidate_image.save(output_path)
    return str(output_path)


def save_selected_image(source_image_path: str | Path, run_context: RunContext, scene_index: int) -> str:
    from shutil import copyfile

    scene_dir = scene_directory(run_context, scene_index)
    scene_dir.mkdir(parents=True, exist_ok=True)
    output_path = scene_dir / "selected.png"
    copyfile(source_image_path, output_path)
    return str(output_path)


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def save_json(path: str | Path, payload: Any) -> None:
    serializable = _to_serializable(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=True)


def save_resolved_config(path: str | Path, config: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def build_manifest(summary: RunSummary) -> dict[str, Any]:
    return {
        "run_name": summary.run_name,
        "timestamp": summary.timestamp,
        "runtime_profile": summary.runtime_profile,
        "model_id": summary.model_id,
        "scene_count": len(summary.scene_results),
        "run_directory": summary.run_directory,
        "config_path": str(Path(summary.run_directory) / "config_resolved.yaml"),
        "summary_path": str(Path(summary.run_directory) / "run_summary.json"),
        "selected_outputs": [
            {
                "scene_id": result.scene_id,
                "selected_candidate_index": result.selected_candidate_index,
                "selected_seed": result.selected_seed,
                "selected_image_path": result.selected_image_path,
            }
            for result in summary.scene_results
        ],
    }
