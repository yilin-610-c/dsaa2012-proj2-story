from __future__ import annotations

import re
from shutil import copyfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from storygen.generators import BaseSceneGenerator
from storygen.io.results import save_json
from storygen.scoring import CLIPConsistencyScorer
from storygen.types import CandidateScore, GenerationCandidate, Scene, Story
from storygen.types import AnchorSpec, GenerationRequest, PromptSpec, RunContext


ANCHOR_TYPE_TEMPLATES = {
    "portrait": "portrait view of {identity}",
    "half_body": "half-body view of {identity}",
    "full_body": "full-body view of {identity}",
}

ANCHOR_NEGATIVE_PROMPT_SUFFIX = "multiple people, group, duplicate face, extra face, extra person, split face, collage"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_path_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "character"


def _dedupe_parts(parts: list[str]) -> list[str]:
    output = []
    seen = set()
    for part in parts:
        text = _normalize_text(part)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _is_generic_human_character_id(character_id: str) -> bool:
    return _normalize_text(character_id).lower() in {
        "girl",
        "boy",
        "woman",
        "man",
        "person",
        "child",
        "kid",
        "lady",
    }


def _human_identity_label(character_spec: dict[str, Any]) -> str | None:
    character_id = _normalize_text(character_spec.get("character_id")).lower()
    age = _normalize_text(character_spec.get("age_band")).lower()
    gender = _normalize_text(character_spec.get("gender_presentation")).lower()
    combined = " ".join([character_id, age, gender])

    child = any(term in combined for term in ["child", "kid", "young", "girl", "boy"])
    female = any(term in combined for term in ["female", "woman", "girl", "lady"])
    male = any(term in combined for term in ["male", "man", "boy"])
    human = any(term in combined for term in ["human", "person", "child", "kid", "female", "male", "woman", "man", "girl", "boy", "lady"])

    if female:
        return "human girl child" if child else "human woman"
    if male:
        return "human boy child" if child else "human man"
    if child:
        return "human child"
    if human:
        return "human person"
    return None


def _identity_parts(character_spec: dict[str, Any]) -> list[str]:
    human_label = _human_identity_label(character_spec)
    character_id = _normalize_text(character_spec.get("character_id"))
    visual_field_names = [
        "hair_color",
        "hairstyle",
        "skin_tone",
        "body_build",
        "signature_outfit",
        "signature_accessory",
        "profession_marker",
    ]
    if human_label:
        name_part = "" if _is_generic_human_character_id(character_id) else character_id
        return _dedupe_parts([f"one {human_label}", name_part, *[character_spec.get(field_name) for field_name in visual_field_names]])

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
    parts = _dedupe_parts([character_spec.get(field_name) for field_name in field_names])
    if parts and not parts[0].lower().startswith("one "):
        parts[0] = f"one {parts[0]}"
    return parts


def build_anchor_prompt(character_spec: dict[str, Any], anchor_type: str, prompt_suffix: str) -> str:
    template = ANCHOR_TYPE_TEMPLATES.get(anchor_type, "{identity}")
    identity = ", ".join(_identity_parts(character_spec))
    prompt = template.format(identity=identity).strip()
    suffix = _normalize_text(prompt_suffix)
    return ", ".join(part for part in [prompt, "single character only", "centered", suffix] if part)


def _anchor_negative_prompt(base_negative_prompt: str) -> str:
    return ", ".join(_dedupe_parts([base_negative_prompt, ANCHOR_NEGATIVE_PROMPT_SUFFIX]))


def _anchor_prompt_spec(
    character_id: str,
    anchor_type: str,
    prompt: str,
    negative_prompt: str,
    *,
    candidate_index: int | None = None,
) -> PromptSpec:
    scene_id = f"ANCHOR-{_safe_path_name(character_id)}-{anchor_type}"
    if candidate_index is not None:
        scene_id = f"{scene_id}-cand-{candidate_index}"
    return PromptSpec(
        scene_id=scene_id,
        style_prompt="",
        character_prompt=character_id,
        global_context_prompt="",
        scene_consistency_prompt="",
        local_prompt=prompt,
        action_prompt=anchor_type,
        generation_prompt=prompt,
        scoring_prompt=prompt,
        full_prompt=prompt,
        negative_prompt=negative_prompt,
    )


def _anchor_type_path(anchor_dir: Path, anchor_type: str) -> Path:
    return anchor_dir / f"{anchor_type}.png"


def _half_body_candidate_path(anchor_dir: Path, candidate_index: int) -> Path:
    return anchor_dir / f"half_body_cand_{candidate_index}.png"


def _canonical_half_body_path(anchor_dir: Path) -> Path:
    return anchor_dir / "canonical_half_body.png"


def _canonical_anchor_metadata_path(anchor_dir: Path) -> Path:
    return anchor_dir / "canonical_anchor.json"


def _half_body_candidate_count(anchor_config: dict[str, Any]) -> int:
    raw = anchor_config.get("half_body_candidate_count", 3)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 3


def _build_half_body_candidates(
    *,
    character_dir: Path,
    prompt: str,
    base_seed: int,
    candidate_count: int,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_index": candidate_index,
            "prompt": prompt,
            "seed": base_seed + candidate_index,
            "image_path": str(_half_body_candidate_path(character_dir, candidate_index)),
        }
        for candidate_index in range(candidate_count)
    ]


def _quality_signals(image_path: Path) -> dict[str, Any]:
    signals = {
        "file_exists": image_path.exists(),
        "image_openable": False,
        "width": None,
        "height": None,
        "pixel_count": 0,
    }
    if not image_path.exists():
        return signals
    try:
        from PIL import Image

        with Image.open(image_path) as image_handle:
            image_handle.load()
            signals["image_openable"] = True
            signals["width"] = int(image_handle.width)
            signals["height"] = int(image_handle.height)
            signals["pixel_count"] = int(image_handle.width * image_handle.height)
    except Exception:
        return signals
    return signals


def _build_selector_scorer(selector_config: dict[str, Any], runtime_config: dict[str, Any]) -> CLIPConsistencyScorer | None:
    method = str(selector_config.get("method", "clip_text_alignment"))
    if method != "clip_text_alignment":
        return None
    try:
        return CLIPConsistencyScorer(
            {
                "type": "clip_consistency",
                "clip_model_id": selector_config.get("clip_model_id", "openai/clip-vit-base-patch32"),
                "clip_max_text_length": selector_config.get("clip_max_text_length", 77),
                "text_image_weight": 1.0,
                "consistency_weight": 0.0,
                "action_weight": 0.0,
            },
            runtime_config,
        )
    except Exception:
        return None


def _select_canonical_half_body(
    *,
    character_id: str,
    prompt: str,
    candidates: list[dict[str, Any]],
    selector_config: dict[str, Any],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    scorer = _build_selector_scorer(selector_config, runtime_config)
    selector_method = str(selector_config.get("method", "clip_text_alignment"))
    scored_candidates = []
    for candidate in candidates:
        image_path = Path(candidate["image_path"])
        quality = _quality_signals(image_path)
        clip_alignment = 0.0
        clip_error = None
        if scorer is not None and quality["file_exists"]:
            try:
                score = scorer.score_candidate(
                    story=Story(source_path="", raw_text="", scenes=[], all_entities=[], recurring_entities=[], entity_to_scene_ids={}),
                    scene=Scene(scene_id=f"ANCHOR-{_safe_path_name(character_id)}", index=0, raw_text="", clean_text="", entities=[character_id]),
                    prompt_spec=PromptSpec(
                        scene_id=f"ANCHOR-{_safe_path_name(character_id)}",
                        style_prompt="",
                        character_prompt=character_id,
                        global_context_prompt="",
                        scene_consistency_prompt="",
                        local_prompt=prompt,
                        action_prompt="half_body",
                        generation_prompt=prompt,
                        scoring_prompt=prompt,
                        full_prompt=prompt,
                        negative_prompt="",
                    ),
                    candidate=GenerationCandidate(
                        scene_id=f"ANCHOR-{_safe_path_name(character_id)}",
                        candidate_index=int(candidate["candidate_index"]),
                        seed=int(candidate["seed"]),
                        prompt_spec=_anchor_prompt_spec(character_id, "half_body", prompt, ""),
                        image=None,
                        image_path=str(image_path),
                        metadata={},
                    ),
                    previous_results=[],
                )
                if isinstance(score, CandidateScore):
                    clip_alignment = float(score.components.get("text_alignment", score.score))
            except Exception as exc:
                clip_error = str(exc)

        quality_score = 0.0
        if quality["file_exists"]:
            quality_score += 0.5
        if quality["image_openable"]:
            quality_score += 0.5
        final_score = clip_alignment + quality_score
        scored_candidates.append(
            {
                **candidate,
                "clip_alignment": round(float(clip_alignment), 6),
                "quality_score": round(float(quality_score), 6),
                "selector_score": round(float(final_score), 6),
                "quality": quality,
                "selector_method": selector_method,
                "clip_error": clip_error,
            }
        )

    selected = sorted(
        scored_candidates,
        key=lambda item: (-float(item["selector_score"]), item["candidate_index"], item["seed"]),
    )[0]
    return {
        "character_id": character_id,
        "anchor_type": "half_body",
        "selector_method": selector_method,
        "prompt": prompt,
        "selected_candidate_index": selected["candidate_index"],
        "selected_seed": selected["seed"],
        "selected_image_path": selected["image_path"],
        "candidates": scored_candidates,
    }


def build_anchor_bank_plan(
    *,
    character_specs: dict[str, dict[str, Any]],
    anchor_config: dict[str, Any],
    run_context: RunContext,
    prompt_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir = run_context.run_directory / str(anchor_config.get("output_dir_name", "anchors"))
    anchor_types = [str(value) for value in anchor_config.get("anchor_types", ["portrait", "half_body"])]
    half_body_candidate_count = _half_body_candidate_count(anchor_config)
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
            if anchor_type == "half_body":
                anchors[anchor_type] = {
                    "anchor_type": anchor_type,
                    "prompt": prompt,
                    "seed": seed,
                    "candidate_count": half_body_candidate_count,
                    "candidates": _build_half_body_candidates(
                        character_dir=character_dir,
                        prompt=prompt,
                        base_seed=seed,
                        candidate_count=half_body_candidate_count,
                    ),
                    "canonical_image_path": str(_canonical_half_body_path(character_dir)),
                    "canonical_metadata_path": str(_canonical_anchor_metadata_path(character_dir)),
                    "image_path": str(_canonical_half_body_path(character_dir)),
                }
            else:
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
                    half_body_path=anchors.get("half_body", {}).get("canonical_image_path")
                    or anchors.get("half_body", {}).get("image_path"),
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
        "half_body_candidate_count": half_body_candidate_count,
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
            if anchor_type == "half_body":
                event_logger(
                    "anchor_generation_started",
                    character_id=character["character_id"],
                    anchor_type=anchor_type,
                    candidate_count=anchor_payload.get("candidate_count", 1),
                )
                for candidate_payload in anchor_payload.get("candidates", []):
                    prompt_spec = _anchor_prompt_spec(
                        character["character_id"],
                        anchor_type,
                        anchor_payload["prompt"],
                        _anchor_negative_prompt(str(prompt_config.get("negative_prompt", "")).strip()),
                        candidate_index=int(candidate_payload["candidate_index"]),
                    )
                    request = GenerationRequest(
                        scene_id=prompt_spec.scene_id,
                        candidate_index=int(candidate_payload["candidate_index"]),
                        seed=int(candidate_payload["seed"]),
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
                            "anchor_candidate_index": int(candidate_payload["candidate_index"]),
                        },
                    )
                    candidate = generator.generate_scene(request)
                    image_path = Path(candidate_payload["image_path"])
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate.image.save(image_path)
                    candidate.image = None
                    candidate_payload["image_path"] = str(image_path)
                    event_logger(
                        "anchor_generation_completed",
                        character_id=character["character_id"],
                        anchor_type=anchor_type,
                        seed=int(candidate_payload["seed"]),
                        candidate_index=int(candidate_payload["candidate_index"]),
                        image_path=str(image_path),
                    )

                selection = _select_canonical_half_body(
                    character_id=character["character_id"],
                    prompt=anchor_payload["prompt"],
                    candidates=list(anchor_payload.get("candidates", [])),
                    selector_config=anchor_config.get("half_body_selector", {}),
                    runtime_config={"device": model_config.get("device", "cpu")},
                )
                canonical_image_path = Path(anchor_payload["canonical_image_path"])
                canonical_image_path.parent.mkdir(parents=True, exist_ok=True)
                copyfile(selection["selected_image_path"], canonical_image_path)
                anchor_payload["canonical_candidate_index"] = selection["selected_candidate_index"]
                anchor_payload["canonical_seed"] = selection["selected_seed"]
                anchor_payload["canonical_image_path"] = str(canonical_image_path)
                anchor_payload["image_path"] = str(canonical_image_path)
                save_json(Path(anchor_payload["canonical_metadata_path"]), selection)
                event_logger(
                    "anchor_canonical_selected",
                    character_id=character["character_id"],
                    anchor_type=anchor_type,
                    canonical_candidate_index=selection["selected_candidate_index"],
                    canonical_seed=selection["selected_seed"],
                    canonical_image_path=str(canonical_image_path),
                )
                continue

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
                _anchor_negative_prompt(str(prompt_config.get("negative_prompt", "")).strip()),
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
