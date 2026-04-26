from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from storygen.generators import build_generation_backend
from storygen.types import GenerationRequest, PromptSpec, Scene


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


def _run_directory_from_scene_result(scene_result_path: Path) -> Path:
    return scene_result_path.parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a single scene with a modified prompt")
    parser.add_argument("scene_result_path", help="Path to scene_result.json")
    parser.add_argument("new_prompt", help="New prompt to generate with")
    parser.add_argument("--backup", action="store_true", help="Write a backup of scene_result.json before editing")
    parser.add_argument("--output-name", default="replay", help="Output image filename stem")
    args = parser.parse_args()

    scene_result_path = Path(args.scene_result_path).expanduser().resolve()
    scene_result = _load_json(scene_result_path)
    if args.backup:
        _save_json(scene_result_path.with_suffix(".json.bak"), scene_result)

    selection = dict(scene_result.get("selection") or {})
    selection["prompt"] = args.new_prompt
    scene_result["selection"] = selection
    updated_candidates = []
    for candidate in scene_result.get("candidates", []):
        candidate = dict(candidate)
        candidate["prompt"] = args.new_prompt
        updated_candidates.append(candidate)
    scene_result["candidates"] = updated_candidates
    _save_json(scene_result_path, scene_result)

    run_directory = _run_directory_from_scene_result(scene_result_path)
    config_path = run_directory / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml at {config_path}")

    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prompt_spec = _prompt_spec_from_selection(selection, args.new_prompt)
    scene = _scene_from_result(scene_result)
    generation_backend = build_generation_backend(config["model"], config["runtime"])
    if hasattr(generation_backend, "load"):
        generation_backend.load()

    request = GenerationRequest(
        scene_id=scene.scene_id,
        candidate_index=int(selection.get("panel_index", scene.index)),
        seed=int(config["generation"]["base_seed"]) + int(scene.index),
        prompt_spec=prompt_spec,
        width=int(config["model"]["width"]),
        height=int(config["model"]["height"]),
        guidance_scale=float(config["model"]["guidance_scale"]),
        num_inference_steps=int(config["model"]["num_inference_steps"]),
        reference_image_path=selection.get("metadata", {}).get("scene_backend_metadata", {}).get("reference_image_path"),
        previous_selected_image_path=selection.get("metadata", {}).get("scene_backend_metadata", {}).get("previous_selected_image_path"),
        extra_options={
            **(selection.get("metadata", {}).get("scene_backend_metadata", {}).get("extra_options") or {}),
            "generation_mode": selection.get("metadata", {}).get("scene_backend_metadata", {}).get("generation_mode", "text2img"),
        },
    )

    candidate = generation_backend.generate_scene(request)
    output_path = scene_result_path.parent / f"{args.output_name}.png"
    if candidate.image is None:
        raise RuntimeError("Generator returned no image")
    candidate.image.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
