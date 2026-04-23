from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from storygen.generators import BaseSceneGenerator
from storygen.io.results import save_json
from storygen.types import AnchorSpec, GenerationRequest, PromptSpec, RunContext


ANCHOR_TYPE_TEMPLATES = {
    "portrait": "portrait view of {identity}",
    "half_body": "half-body view of {identity}",
    "full_body": "full-body view of {identity}",
}


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_path_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "character"


def _identity_parts(character_spec: dict[str, Any]) -> list[str]:
    field_names = [
        "character_id",
        "age_band",
        "gender_presentation",
        "hair_color",
        "hairstyle",
        "skin_tone",
        "body_build",
        "signature_outfit",
        "signature_accessory",
        "profession_marker",
    ]
    parts = []
    seen = set()
    for field_name in field_names:
        text = _normalize_text(character_spec.get(field_name))
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            parts.append(text)
    return parts


def build_anchor_prompt(character_spec: dict[str, Any], anchor_type: str, prompt_suffix: str) -> str:
    template = ANCHOR_TYPE_TEMPLATES.get(anchor_type, "{identity}")
    identity = ", ".join(_identity_parts(character_spec))
    prompt = template.format(identity=identity).strip()
    suffix = _normalize_text(prompt_suffix)
    return ", ".join(part for part in [prompt, suffix] if part)


def _anchor_prompt_spec(character_id: str, anchor_type: str, prompt: str, negative_prompt: str) -> PromptSpec:
    return PromptSpec(
        scene_id=f"ANCHOR-{_safe_path_name(character_id)}-{anchor_type}",
        style_prompt="",
        character_prompt=character_id,
        global_context_prompt="",
        local_prompt=prompt,
        action_prompt=anchor_type,
        generation_prompt=prompt,
        scoring_prompt=prompt,
        full_prompt=prompt,
        negative_prompt=negative_prompt,
    )


def _anchor_type_path(anchor_dir: Path, anchor_type: str) -> Path:
    return anchor_dir / f"{anchor_type}.png"


def build_anchor_bank_plan(
    *,
    character_specs: dict[str, dict[str, Any]],
    anchor_config: dict[str, Any],
    run_context: RunContext,
    prompt_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir = run_context.run_directory / str(anchor_config.get("output_dir_name", "anchors"))
    anchor_types = [str(value) for value in anchor_config.get("anchor_types", ["portrait", "half_body"])]
    prompt_suffix = str(
        anchor_config.get(
            "prompt_suffix",
            "clean identity reference image, simple background, consistent character design",
        )
    )
    base_seed_offset = int(anchor_config.get("base_seed_offset", 900000))
    negative_prompt = str(prompt_config.get("negative_prompt", "")).strip()

    characters = {}
    for character_index, (character_id, character_spec) in enumerate(character_specs.items()):
        safe_character_id = _safe_path_name(character_id)
        character_dir = output_dir / safe_character_id
        anchors = {}
        for anchor_index, anchor_type in enumerate(anchor_types):
            seed = base_seed_offset + character_index * 100 + anchor_index
            prompt = build_anchor_prompt(character_spec, anchor_type, prompt_suffix)
            anchors[anchor_type] = {
                "anchor_type": anchor_type,
                "prompt": prompt,
                "seed": seed,
                "image_path": str(_anchor_type_path(character_dir, anchor_type)),
            }
        characters[character_id] = {
            "character_id": character_id,
            "safe_character_id": safe_character_id,
            "character_spec": character_spec,
            "anchor_spec": asdict(
                AnchorSpec(
                    character_id=character_id,
                    portrait_path=anchors.get("portrait", {}).get("image_path"),
                    half_body_path=anchors.get("half_body", {}).get("image_path"),
                    full_body_path=anchors.get("full_body", {}).get("image_path"),
                    metadata={
                        "source": "character_specs",
                        "output_dir": str(character_dir),
                    },
                )
            ),
            "anchors": anchors,
        }

    return {
        "enabled": bool(anchor_config.get("enabled", False)),
        "generate": bool(anchor_config.get("generate", False)),
        "output_dir": str(output_dir),
        "anchor_types": anchor_types,
        "characters": characters,
    }


def run_anchor_bank(
    *,
    character_specs: dict[str, dict[str, Any]],
    anchor_config: dict[str, Any],
    run_context: RunContext,
    prompt_config: dict[str, Any],
    model_config: dict[str, Any],
    generator: BaseSceneGenerator,
    event_logger: Callable[[str, Any], None],
) -> dict[str, Any]:
    plan = build_anchor_bank_plan(
        character_specs=character_specs,
        anchor_config=anchor_config,
        run_context=run_context,
        prompt_config=prompt_config,
    )
    if not plan["enabled"]:
        return plan

    event_logger(
        "anchor_bank_started",
        character_count=len(plan["characters"]),
        generate=plan["generate"],
        anchor_types=plan["anchor_types"],
    )
    for character in plan["characters"].values():
        character_dir = Path(plan["output_dir"]) / character["safe_character_id"]
        character_dir.mkdir(parents=True, exist_ok=True)
        save_json(character_dir / "anchor_spec.json", character)

        if not plan["generate"]:
            continue

        for anchor_type, anchor_payload in character["anchors"].items():
            event_logger(
                "anchor_generation_started",
                character_id=character["character_id"],
                anchor_type=anchor_type,
                seed=anchor_payload["seed"],
            )
            prompt_spec = _anchor_prompt_spec(
                character["character_id"],
                anchor_type,
                anchor_payload["prompt"],
                str(prompt_config.get("negative_prompt", "")).strip(),
            )
            request = GenerationRequest(
                scene_id=prompt_spec.scene_id,
                candidate_index=0,
                seed=int(anchor_payload["seed"]),
                prompt_spec=prompt_spec,
                width=int(model_config["width"]),
                height=int(model_config["height"]),
                guidance_scale=float(model_config["guidance_scale"]),
                num_inference_steps=int(model_config["num_inference_steps"]),
                reference_image_path=None,
                previous_selected_image_path=None,
                extra_options={
                    "generation_mode": "text2img",
                    "anchor_type": anchor_type,
                    "character_id": character["character_id"],
                    "route_reason": "anchor_bank_generation",
                },
            )
            candidate = generator.generate_scene(request)
            image_path = Path(anchor_payload["image_path"])
            image_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.image.save(image_path)
            candidate.image = None
            anchor_payload["image_path"] = str(image_path)
            event_logger(
                "anchor_generation_completed",
                character_id=character["character_id"],
                anchor_type=anchor_type,
                seed=anchor_payload["seed"],
                image_path=str(image_path),
            )

        save_json(character_dir / "anchor_spec.json", character)

    event_logger(
        "anchor_bank_completed",
        character_count=len(plan["characters"]),
        generate=plan["generate"],
        output_dir=plan["output_dir"],
    )
    return plan
