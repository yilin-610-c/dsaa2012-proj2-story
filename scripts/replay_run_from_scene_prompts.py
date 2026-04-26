from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from storygen.generators import build_generation_backend
from storygen.types import GenerationRequest, PromptSpec, Scene, StoryGenerationRequest, StoryScenePlan


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _scene_from_result(scene_result: dict[str, Any]) -> Scene:
    scene = scene_result["scene"]
    return Scene(
        scene_id=str(scene["scene_id"]),
        index=int(scene["index"]),
        raw_text=str(scene["raw_text"]),
        clean_text=str(scene["clean_text"]),
        entities=list(scene.get("entities", [])),
    )


def _prompt_spec_from_selection(selection: dict[str, Any], new_prompt: str) -> PromptSpec:
    scene_id = str(selection.get("scene_id") or "SCENE")
    return PromptSpec(
        scene_id=scene_id,
        style_prompt="",
        character_prompt="",
        global_context_prompt="",
        scene_consistency_prompt="",
        local_prompt=new_prompt,
        action_prompt=new_prompt,
        generation_prompt=new_prompt,
        scoring_prompt=new_prompt,
        full_prompt=new_prompt,
        negative_prompt="",
    )


def _scene_result_paths(run_directory: Path) -> list[Path]:
    scenes_dir = run_directory / "scenes"
    return sorted(scenes_dir.glob("scene_*/scene_result.json"))


def _build_story_request(config: dict[str, Any], scene_result_path: Path, scene_result: dict[str, Any], prompt_spec: PromptSpec) -> StoryGenerationRequest:
    selection = dict(scene_result.get("selection") or {})
    scene = _scene_from_result(scene_result)
    backend_metadata = selection.get("metadata", {}).get("scene_backend_metadata", {})
    scene_plan = StoryScenePlan(
        scene_id=scene.scene_id,
        scene_index=scene.index,
        prompt_spec=prompt_spec,
        generation_prompt=prompt_spec.generation_prompt,
        scoring_prompt=prompt_spec.scoring_prompt,
        route_hint=dict(selection.get("metadata", {}).get("route_hint") or {}),
        identity_plan=dict(selection.get("metadata", {}).get("identity_plan") or {}),
        anchor_characters=list(selection.get("metadata", {}).get("anchor_characters") or []),
        anchor_paths=dict(selection.get("metadata", {}).get("anchor_paths") or {}),
        metadata={"scene_text": scene.clean_text, "scene_entities": list(scene.entities)},
    )
    return StoryGenerationRequest(
        story_id=scene_result_path.parents[2].name,
        seed=int(config["generation"]["base_seed"]),
        character_description=str(selection.get("prompt") or ""),
        panel_prompts=[prompt_spec.generation_prompt],
        num_identity_panels=int(config["model"].get("num_identity_panels", 3)),
        style_name=config["prompt"].get("style_name"),
        negative_prompt=config["prompt"].get("negative_prompt", ""),
        width=int(config["model"]["width"]),
        height=int(config["model"]["height"]),
        reference_image_path=backend_metadata.get("reference_image_path"),
        scene_plans=[scene_plan],
        anchor_bank_summary={},
        character_specs={},
        previous_style_reference_path=backend_metadata.get("extra_options", {}).get("style_reference_path"),
        extra_options=config["model"].get("extra_options", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay every edited scene prompt in a run directory")
    parser.add_argument("run_directory", help="Path to the run directory, e.g. outputs/compare_06_with_scene_consistency_v7")
    parser.add_argument("--backup", action="store_true", help="Write .bak copies before editing scene_result.json files")
    parser.add_argument("--output-name", default="replay", help="Output image filename stem for each scene")
    args = parser.parse_args()

    run_directory = Path(args.run_directory).expanduser().resolve()
    config_path = run_directory / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml at {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generator = build_generation_backend(config["model"], config["runtime"])
    if hasattr(generator, "load"):
        generator.load()

    scene_result_paths = _scene_result_paths(run_directory)
    if not scene_result_paths:
        raise FileNotFoundError(f"No scene_result.json files found under {run_directory / 'scenes'}")

    for scene_result_path in scene_result_paths:
        scene_result = _load_json(scene_result_path)
        selection = dict(scene_result.get("selection") or {})
        new_prompt = str(selection.get("prompt") or "").strip()
        if not new_prompt:
            raise ValueError(f"Empty prompt in {scene_result_path}")

        if args.backup:
            backup_path = scene_result_path.with_suffix(scene_result_path.suffix + ".bak")
            _save_json(backup_path, scene_result)

        updated_candidates = []
        for candidate in scene_result.get("candidates", []):
            candidate = dict(candidate)
            candidate["prompt"] = new_prompt
            updated_candidates.append(candidate)
        scene_result["candidates"] = updated_candidates
        _save_json(scene_result_path, scene_result)

        prompt_spec = _prompt_spec_from_selection(selection, new_prompt)
        request = _build_story_request(config, scene_result_path, scene_result, prompt_spec)
        result = generator.generate_story(request)
        if not result.panel_outputs or result.panel_outputs[0].image is None:
            raise RuntimeError(f"No image returned for {scene_result_path}")

        output_path = scene_result_path.parent / f"{args.output_name}.png"
        result.panel_outputs[0].image.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
