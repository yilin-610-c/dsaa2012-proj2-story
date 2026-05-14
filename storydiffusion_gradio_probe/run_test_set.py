from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_INPUT_DIR = REPO_ROOT / "test_set"
DEFAULT_CONFIG_DIR = REPO_ROOT / "outputs" / "storydiffusion_gradio_probe" / "configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "storydiffusion_gradio_probe" / "test_set"
DEFAULT_STORYDIFFUSION_ROOT = REPO_ROOT.parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_PROFILE = "cloud_storydiffusion_debug"
DEFAULT_ANCHOR_RUN_DIR = REPO_ROOT / "outputs" / "compare_aesthetic_16_low_ip"

DEFAULT_NEGATIVE_PROMPT = (
    "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, "
    "bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, "
    "cloned face, ugly fingers, cartoon, cg, 3d, unreal, amputation, disconnected limbs"
)
NATIVE_CLEAN_LLM_MAX_OUTPUT_TOKENS = 2400

SCENE_PATTERN = re.compile(r"^\[(SCENE-\d+)\]\s*(.*)$", re.DOTALL)
ENTITY_PATTERN = re.compile(r"<([^<>]+)>")
LEADING_PRONOUN_PATTERN = re.compile(r"^(she|he|they|it)\b", re.IGNORECASE)
HUMAN_HINTS = {
    "artist",
    "boy",
    "chef",
    "doctor",
    "driver",
    "girl",
    "kid",
    "lily",
    "man",
    "milo",
    "person",
    "ryan",
    "sara",
    "student",
    "teacher",
    "tom",
    "woman",
}
ANIMAL_HINTS = {
    "dog",
    "cat",
    "bird",
    "horse",
    "fox",
    "wolf",
    "bear",
    "rabbit",
    "lion",
    "tiger",
    "deer",
    "fish",
}


@dataclass(frozen=True)
class SceneLine:
    scene_id: str
    raw_text: str
    entities: list[str]
    clean_text: str


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _parse_story_file(path: Path) -> list[SceneLine]:
    raw_text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw_text.split("[SEP]") if block.strip()]
    scenes: list[SceneLine] = []
    for index, block in enumerate(blocks):
        match = SCENE_PATTERN.match(block)
        if not match:
            raise ValueError(f"Invalid scene block in {path} at index {index}: {block!r}")
        scene_id, scene_body = match.groups()
        scene_body = scene_body.strip()
        entities = ENTITY_PATTERN.findall(scene_body)
        clean_text = _normalize_whitespace(ENTITY_PATTERN.sub(r"\1", scene_body))
        scenes.append(SceneLine(scene_id=scene_id, raw_text=scene_body, entities=entities, clean_text=clean_text))
    if not scenes:
        raise ValueError(f"No scenes found in {path}")
    return scenes


def _primary_entities(scenes: list[SceneLine]) -> list[str]:
    seen: list[str] = []
    for scene in scenes:
        for entity in scene.entities:
            if entity not in seen:
                seen.append(entity)
    if seen:
        return seen
    return ["Subject"]


def _subject_description(entity: str, *, use_reference_images: bool) -> str:
    normalized = entity.lower()
    trigger = " img" if use_reference_images else ""
    if any(animal in normalized for animal in ANIMAL_HINTS):
        return f"a {normalized}{trigger}, consistent markings, consistent body proportions"
    if "girl" in normalized:
        return f"a girl{trigger}, consistent hairstyle, consistent outfit"
    if "boy" in normalized:
        return f"a boy{trigger}, consistent hairstyle, consistent outfit"
    if normalized in HUMAN_HINTS or entity[:1].isupper():
        return f"a person{trigger}, consistent face, hairstyle, and outfit"
    return f"a subject{trigger}, consistent shape, colors, and details"


def _replace_leading_pronoun(text: str, fallback_entities: list[str]) -> str:
    if not fallback_entities:
        return text
    replacement = " and ".join(fallback_entities[:2])
    return LEADING_PRONOUN_PATTERN.sub(replacement, text, count=1)


def _scene_prompt(scene: SceneLine, active_entities: list[str]) -> str:
    entities = scene.entities or active_entities
    text = scene.clean_text
    if not scene.entities:
        text = _replace_leading_pronoun(text, active_entities)
    if not entities:
        return f"[NC]{text}"
    prefix = " and ".join(f"[{entity}]" for entity in entities)
    return f"{prefix} {text}"


def _identity_prompt(entity: str) -> str:
    normalized = entity.lower()
    if any(animal in normalized for animal in ANIMAL_HINTS):
        subject = f"a {normalized}"
    elif "girl" in normalized:
        subject = "a girl"
    elif "boy" in normalized:
        subject = "a boy"
    else:
        subject = "a person"
    # Important: avoid portrait-style identity prompts (they bias the bank toward headshots).
    return (
        f"[{entity}] {subject}, full body, wide shot, clear environment, "
        "dynamic pose, action readable, consistent outfit, consistent silhouette"
    )


def _ensure_storygen_imports() -> None:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))


def _load_anchor_bank_summary(run_dir: Path) -> dict[str, Any]:
    """
    Loads the Anchor Bank summary produced by the storygen pipeline.
    Expected file: outputs/<run_name>/logs/anchor_bank.json
    """
    anchor_path = run_dir / "logs" / "anchor_bank.json"
    if not anchor_path.exists():
        return {}
    try:
        return yaml.safe_load(anchor_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _anchor_half_body_path(anchor_bank_summary: dict[str, Any], character_id: str) -> str:
    characters = anchor_bank_summary.get("characters", {})
    if not isinstance(characters, dict):
        return ""
    payload = characters.get(character_id)
    if not isinstance(payload, dict):
        return ""
    anchor_spec = payload.get("anchor_spec", {})
    if isinstance(anchor_spec, dict):
        half_body = str(anchor_spec.get("half_body_path") or "").strip()
        if half_body:
            return half_body
    anchors = payload.get("anchors", {})
    if isinstance(anchors, dict):
        half = anchors.get("half_body")
        if isinstance(half, dict):
            path = str(half.get("canonical_image_path") or half.get("image_path") or "").strip()
            return path
    return ""


def _has_required_anchor_images(anchor_bank_summary: dict[str, Any], required_character_ids: list[str]) -> bool:
    if not required_character_ids:
        return True
    for character_id in required_character_ids:
        path = _anchor_half_body_path(anchor_bank_summary, character_id)
        if not path or not Path(path).exists():
            return False
    return True


def _ensure_anchor_bank_generated(
    *,
    run_dir: Path,
    prompt_profile: str,
    required_character_ids: list[str],
    device: str,
    torch_dtype: str,
) -> dict[str, Any]:
    """
    Ensure `logs/anchor_bank.json` exists and required half-body anchors are present.
    This reuses the repo's Anchor Bank implementation (same as the main pipeline).
    """
    run_dir = run_dir.resolve()
    anchor_bank_summary = _load_anchor_bank_summary(run_dir)
    if anchor_bank_summary and _has_required_anchor_images(anchor_bank_summary, required_character_ids):
        return anchor_bank_summary

    _ensure_storygen_imports()
    from storygen.anchor_bank import run_anchor_bank
    from storygen.config import resolve_config
    from storygen.generators import BaseSceneGenerator, build_generation_backend
    from storygen.io.results import append_event, create_run_context, save_json
    from storygen.parser import parse_story_file
    from storygen.prompt_pipelines import build_prompt_pipeline

    # Build a minimal run context rooted at the existing run_dir name.
    run_context = create_run_context(str(run_dir.parent), run_dir.name)

    resolved = resolve_config(
        DEFAULT_BASE_CONFIG,
        prompt_profile,
        overrides={
            "runtime.output_root": str(run_dir.parent),
            "runtime.run_name": str(run_dir.name),
            "runtime.repo_root": str(REPO_ROOT),
            "runtime.device": device,
            "runtime.torch_dtype": torch_dtype,
            # Keep generation cheap; anchors are identity-only anyway.
            "model.width": 768,
            "model.height": 768,
            "model.guidance_scale": 0.0,
            "model.num_inference_steps": 4,
            # Ensure anchor bank is enabled+generated.
            "generation.anchor_bank.enabled": True,
            "generation.anchor_bank.generate": True,
        },
    )

    # Build character_specs via the configured prompt pipeline (LLM-assisted profiles will fallback if configured).
    input_path = resolved.get("runtime", {}).get("input_path")
    if not input_path:
        raise ValueError("Resolved config missing runtime.input_path for anchor generation")
    story = parse_story_file(input_path)
    prompt_pipeline = build_prompt_pipeline(
        resolved.get("prompt", {}),
        event_logger=lambda event, **metadata: append_event(run_context, event, stage="prompt", **metadata),
    )
    prompt_bundle = prompt_pipeline.build(story)
    character_specs = prompt_bundle.metadata.get("character_specs", {}) if isinstance(prompt_bundle.metadata, dict) else {}

    # Build a scene-level generator stub for anchor generation.
    scene_stub_model_id = str(
        resolved.get("model", {}).get("anchor_bank_model_id")
        or resolved.get("model", {}).get("scene_model_id")
        or resolved.get("model", {}).get("model_id")
        or "stabilityai/sdxl-turbo"
    ).strip()
    scene_stub_generator = build_generation_backend(
        {
            **dict(resolved.get("model", {})),
            "backend": "diffusers_text2img",
            "granularity": "scene",
            "model_id": scene_stub_model_id,
            "consistent_attention": {
                **dict(dict(resolved.get("model", {})).get("consistent_attention", {})),
                "enabled": False,
            },
        },
        dict(resolved.get("runtime", {})),
    )
    if not isinstance(scene_stub_generator, BaseSceneGenerator):
        raise TypeError(f"Expected a scene-level generator, got {type(scene_stub_generator).__name__}")

    anchor_bank_summary = run_anchor_bank(
        character_specs=character_specs,
        anchor_config=dict(resolved.get("generation", {}).get("anchor_bank", {})),
        run_context=run_context,
        prompt_config=dict(resolved.get("prompt", {})),
        model_config=dict(resolved.get("model", {})) | {"device": device, "torch_dtype": torch_dtype},
        generator=scene_stub_generator,
        event_logger=lambda event, **metadata: append_event(run_context, event, stage="anchor_bank", **metadata),
    )
    if anchor_bank_summary.get("enabled", False):
        save_json(run_context.logs_directory / "anchor_bank.json", anchor_bank_summary)

    return _load_anchor_bank_summary(run_dir)


def _prompt_probe_overrides(
    *,
    generation_max_words: int,
    generation_max_chars: int,
    prompt_builder_kind: str = "legacy",
    prompt_modular_backend: str = "sdxl",
    prompt_template_pack: str | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "prompt.generation_max_words": int(generation_max_words),
        "prompt.generation_max_chars": int(generation_max_chars),
        "prompt.dual_primary_generation_max_words": int(max(generation_max_words, 70)),
        "prompt.dual_primary_generation_max_chars": int(max(generation_max_chars, 420)),
        "prompt.generation_scene_consistency_max_words": 40,
        "prompt.generation_scene_consistency_max_chars": 240,
    }
    if str(prompt_builder_kind).strip().lower() == "modular":
        overrides["prompt.builder"] = "modular"
        overrides["prompt.modular.backend"] = str(prompt_modular_backend).strip().lower()
        if prompt_template_pack:
            overrides["prompt.modular.template_pack"] = str(prompt_template_pack).strip()
    return overrides


def _build_scene_prompt_array_with_pipeline(
    input_path: Path,
    *,
    profile: str,
    generation_max_words: int,
    generation_max_chars: int,
    prompt_builder_kind: str = "legacy",
    prompt_modular_backend: str = "sdxl",
    prompt_template_pack: str | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """
    Uses the repo PromptBuilder to expand scene prompts, then formats them
    into StoryDiffusion-compatible lines with [Character] prefixes.
    Returns (prompt_array, debug_metadata).
    """
    _ensure_storygen_imports()
    from storygen.config import resolve_config
    from storygen.parser import parse_story_file
    from storygen.prompt_stack.factory import build_rule_prompt_builder

    probe_overrides = _prompt_probe_overrides(
        generation_max_words=generation_max_words,
        generation_max_chars=generation_max_chars,
        prompt_builder_kind=prompt_builder_kind,
        prompt_modular_backend=prompt_modular_backend,
        prompt_template_pack=prompt_template_pack,
    )

    resolved = resolve_config(
        DEFAULT_BASE_CONFIG,
        profile,
        overrides=probe_overrides,
    )
    prompt_config = dict(resolved.get("prompt") or {})
    builder = build_rule_prompt_builder(prompt_config)

    story = parse_story_file(input_path)
    prompt_specs = builder.build_story_prompts(story)

    def fallback_entities_for_scene(scene) -> list[str]:
        if scene.entities:
            return list(dict.fromkeys(scene.entities))
        if story.recurring_entities:
            return [story.recurring_entities[0]]
        if story.all_entities:
            return [story.all_entities[0]]
        return []

    prompt_array: list[str] = []
    for scene in story.scenes:
        entities = fallback_entities_for_scene(scene)
        spec = prompt_specs.get(scene.scene_id)
        text = spec.generation_prompt if spec else scene.clean_text
        if not entities:
            prompt_array.append(f"[NC]{text}")
            continue
        prefix = " and ".join(f"[{entity}]" for entity in entities)
        prompt_array.append(f"{prefix} {text}")

    debug = {
        "prompt_pipeline": "prompt_builder:generation_prompt",
        "prompt_builder": str(prompt_builder_kind),
        "prompt_modular_backend": str(prompt_modular_backend),
        "profile": profile,
        "base_config": str(DEFAULT_BASE_CONFIG),
        "probe_overrides": {
            "prompt.generation_max_words": int(generation_max_words),
            "prompt.generation_max_chars": int(generation_max_chars),
            "prompt.llm.fallback_to_rule_based": False,
        },
        "story_entities": list(story.all_entities),
        "recurring_entities": list(story.recurring_entities),
    }
    return prompt_array, debug, prompt_config


def _build_clean_storydiffusion_prompt_payload(
    input_path: Path,
    *,
    profile: str,
    generation_max_words: int,
    generation_max_chars: int,
    prompt_builder_kind: str = "legacy",
    prompt_modular_backend: str = "sdxl",
    prompt_template_pack: str | None = None,
    storydiffusion_prompt_mode: str = "clean",
    identity_prompts_per_character: int = 1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _ensure_storygen_imports()
    from storygen.config import resolve_config
    from storygen.parser import parse_story_file
    from storygen.prompt_pipelines import build_prompt_pipeline
    from storygen.prompt_stack.renderers.storydiffusion import (
        render_clean_native_storydiffusion_prompts,
        render_clean_v2_native_storydiffusion_prompts,
        render_natural_native_storydiffusion_prompts,
    )

    probe_overrides = _prompt_probe_overrides(
        generation_max_words=generation_max_words,
        generation_max_chars=generation_max_chars,
        prompt_builder_kind=prompt_builder_kind,
        prompt_modular_backend=prompt_modular_backend,
        prompt_template_pack=prompt_template_pack,
    )
    probe_overrides["prompt.llm.fallback_to_rule_based"] = False
    probe_overrides["prompt.llm.max_output_tokens"] = NATIVE_CLEAN_LLM_MAX_OUTPUT_TOKENS
    resolved = resolve_config(DEFAULT_BASE_CONFIG, profile, overrides=probe_overrides)
    prompt_config = dict(resolved.get("prompt") or {})
    if prompt_config.get("pipeline") != "llm_assisted":
        raise RuntimeError("clean native StoryDiffusion prompt mode requires prompt.pipeline=llm_assisted")
    story = parse_story_file(input_path)
    prompt_pipeline = build_prompt_pipeline(prompt_config)
    prompt_bundle = prompt_pipeline.build(story)
    bundle_metadata = prompt_bundle.metadata if isinstance(prompt_bundle.metadata, dict) else {}
    character_specs = bundle_metadata.get("character_specs", {})
    non_llm_specs = [
        name
        for name, spec in character_specs.items()
        if not isinstance(spec, dict) or (spec.get("metadata") or {}).get("source") != "llm_assisted"
    ]
    if not character_specs or non_llm_specs:
        raise RuntimeError(
            "clean native StoryDiffusion prompt mode requires llm_assisted character_specs; "
            f"non_llm_specs={non_llm_specs or 'missing'}"
        )
    mode = str(storydiffusion_prompt_mode or "clean").strip().lower()
    if mode == "natural":
        rendered = render_natural_native_storydiffusion_prompts(
            story,
            prompt_bundle.scene_prompts,
            character_specs,
            scene_plans=bundle_metadata.get("scene_plans", {}),
            identity_prompts_per_character=identity_prompts_per_character,
        )
    elif mode == "clean_v2":
        rendered = render_clean_v2_native_storydiffusion_prompts(
            story,
            prompt_bundle.scene_prompts,
            character_specs,
            scene_plans=bundle_metadata.get("scene_plans", {}),
            identity_prompts_per_character=identity_prompts_per_character,
        )
    else:
        rendered = render_clean_native_storydiffusion_prompts(story, prompt_bundle.scene_prompts, character_specs)
    identity_prompt_count = len(rendered.identity_reference_prompts or rendered.identity_prompts)
    saved_image_prompt_map = rendered.saved_image_prompt_map or {
        f"image_{index:03d}.png": {
            "prompt_array_index": rendered.save_image_start_index + index,
            "story_scene_prompt_index": index,
            "scene_id": (rendered.source_fields[index] or {}).get("scene_id") if index < len(rendered.source_fields) else None,
            "prompt": prompt,
        }
        for index, prompt in enumerate(rendered.scene_prompts)
    }
    payload = {
        "general_prompt": rendered.general_prompt,
        "identity_prompts": rendered.identity_prompts,
        "scene_prompts": rendered.scene_prompts,
        "identity_reference_prompts": rendered.identity_reference_prompts or rendered.identity_prompts,
        "story_scene_prompts": rendered.story_scene_prompts or rendered.scene_prompts,
        "identity_prompts_per_character": rendered.identity_prompts_per_character,
        "identity_prompt_count": identity_prompt_count,
        "story_frame_start_index": rendered.save_image_start_index,
        "saved_image_prompt_map": saved_image_prompt_map,
        "prompt_array": rendered.final_prompt_array,
        "save_image_start_index": rendered.save_image_start_index,
        "debug": {
            "mode": mode,
            "general_prompt": rendered.general_prompt,
            "identity_prompts": rendered.identity_prompts,
            "scene_prompts": rendered.scene_prompts,
            "identity_prompts_per_character": rendered.identity_prompts_per_character,
            "identity_prompt_count": identity_prompt_count,
            "story_frame_start_index": rendered.save_image_start_index,
            "identity_reference_prompts": rendered.identity_reference_prompts or rendered.identity_prompts,
            "story_scene_prompts": rendered.story_scene_prompts or rendered.scene_prompts,
            "saved_image_prompt_map": saved_image_prompt_map,
            "structured_source_fields": [
                field.get("structured_source_fields", {}) for field in rendered.source_fields
            ],
            "natural_scene_prompt": [
                field.get("natural_scene_prompt") for field in rendered.source_fields if field.get("natural_scene_prompt")
            ],
            "validation_warnings": [
                {
                    "scene_id": field.get("scene_id"),
                    "warnings": field.get("validation_warnings", []),
                }
                for field in rendered.source_fields
                if field.get("validation_warnings") is not None
            ],
            "final_prompt_array": rendered.final_prompt_array,
            "save_image_start_index": rendered.save_image_start_index,
            "character_specs": rendered.character_specs,
            "source_fields": rendered.source_fields,
            "llm_response_record": bundle_metadata.get("_llm_response_record"),
        },
    }
    debug = {
        "prompt_pipeline": "prompt_bundle:storydiffusion_clean_renderer",
        "prompt_builder": str(prompt_builder_kind),
        "prompt_modular_backend": str(prompt_modular_backend),
        "storydiffusion_prompt_mode": mode,
        "profile": profile,
        "base_config": str(DEFAULT_BASE_CONFIG),
        "probe_overrides": {
            "prompt.generation_max_words": int(generation_max_words),
            "prompt.generation_max_chars": int(generation_max_chars),
            "prompt.llm.fallback_to_rule_based": False,
            "prompt.llm.max_output_tokens": NATIVE_CLEAN_LLM_MAX_OUTPUT_TOKENS,
        },
        "story_entities": list(story.all_entities),
        "recurring_entities": list(story.recurring_entities),
    }
    return payload, debug, prompt_config


def _identity_image_prompt_map(output_dir: Path, prompt_array: list[str], identity_prompt_count: int) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for index, prompt in enumerate(prompt_array[:identity_prompt_count]):
        image_name = f"identity_{index:03d}.png"
        mapping[image_name] = {
            "path": str(output_dir / "identity_refs" / image_name),
            "prompt_array_index": index,
            "prompt": prompt,
        }
    return mapping


def build_probe_config(
    input_path: Path,
    *,
    output_root: Path,
    unwrap_output_dir: bool,
    storydiffusion_root: Path,
    use_reference_images: bool,
    reference_images: list[str],
    id_length: int,
    sd_type: str,
    style: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    sa32: float,
    sa64: float,
    height: int,
    width: int,
    prompt_profile: str,
    prompt_generation_max_words: int,
    prompt_generation_max_chars: int,
    anchor_bank_summary: dict[str, Any] | None,
    prompt_builder_kind: str = "legacy",
    prompt_modular_backend: str = "sdxl",
    prompt_template_pack: str | None = None,
    storydiffusion_prompt_mode: str = "current",
    save_identity_images: bool = False,
) -> dict[str, Any]:
    scenes = _parse_story_file(input_path)
    entities = _primary_entities(scenes)
    resolved_reference_images: list[str] = list(reference_images)
    resolved_use_reference_images = bool(use_reference_images)
    if anchor_bank_summary:
        discovered: list[str] = []
        for entity in entities:
            candidate = _anchor_half_body_path(anchor_bank_summary, entity)
            if candidate and Path(candidate).exists():
                discovered.append(candidate)
        if discovered:
            resolved_reference_images = discovered
            resolved_use_reference_images = True

    prompt_mode = str(storydiffusion_prompt_mode or "current").strip().lower()
    if prompt_mode not in {"current", "clean", "clean_v2", "natural"}:
        raise ValueError(f"Unsupported StoryDiffusion prompt mode: {storydiffusion_prompt_mode}")
    storydiffusion_prompt_debug: dict[str, Any] | None = None
    clean_internal_id_length: int | None = None
    generation_id_length_override: int | None = None
    if prompt_mode in {"clean", "clean_v2", "natural"}:
        clean_payload, pipeline_debug, prompt_config = _build_clean_storydiffusion_prompt_payload(
            input_path,
            profile=prompt_profile,
            generation_max_words=prompt_generation_max_words,
            generation_max_chars=prompt_generation_max_chars,
            prompt_builder_kind=prompt_builder_kind,
            prompt_modular_backend=prompt_modular_backend,
            prompt_template_pack=prompt_template_pack,
            storydiffusion_prompt_mode=prompt_mode,
            identity_prompts_per_character=id_length,
        )
        prompt_array = list(clean_payload["prompt_array"])
        character_prompt = str(clean_payload["general_prompt"])
        save_image_start_index = int(clean_payload["save_image_start_index"])
        storydiffusion_prompt_debug = dict(clean_payload["debug"])
        if prompt_mode in {"clean_v2", "natural"}:
            clean_internal_id_length = int(clean_payload["identity_prompts_per_character"])
            generation_id_length_override = int(clean_payload["identity_prompt_count"])
    else:
        story_prompts, pipeline_debug, prompt_config = _build_scene_prompt_array_with_pipeline(
            input_path,
            profile=prompt_profile,
            generation_max_words=prompt_generation_max_words,
            generation_max_chars=prompt_generation_max_chars,
            prompt_builder_kind=prompt_builder_kind,
            prompt_modular_backend=prompt_modular_backend,
            prompt_template_pack=prompt_template_pack,
        )
        prompt_array = [_identity_prompt(entity) for entity in entities] + story_prompts
        character_prompt = "\n".join(
            f"[{entity}] {_subject_description(entity, use_reference_images=resolved_use_reference_images)}"
            for entity in entities
        )
        save_image_start_index = len(entities)

    single_entity_prompt_count = sum(1 for prompt in prompt_array if prompt.count("[") == 1 and not prompt.startswith("[NC]"))
    effective_id_length = (
        generation_id_length_override
        if generation_id_length_override is not None
        else max(1, min(id_length, single_entity_prompt_count or len(prompt_array)))
    )
    if unwrap_output_dir:
        probe_output_dir = output_root
    else:
        probe_output_dir = output_root / input_path.stem
    identity_image_prompt_map = (
        _identity_image_prompt_map(probe_output_dir, prompt_array, save_image_start_index)
        if save_identity_images and save_image_start_index > 0
        else {}
    )
    if storydiffusion_prompt_debug is not None:
        storydiffusion_prompt_debug["save_identity_images"] = bool(save_identity_images)
        storydiffusion_prompt_debug["identity_image_prompt_map"] = identity_image_prompt_map
    return {
        "storydiffusion_root": str(storydiffusion_root),
        "output_dir": str(probe_output_dir),
        "use_reference_images": resolved_use_reference_images,
        "reference_images": resolved_reference_images,
        "save_identity_images": bool(save_identity_images),
        "source_story": str(input_path),
        "prompt_debug": pipeline_debug,
        **({"storydiffusion_prompt_debug": storydiffusion_prompt_debug} if storydiffusion_prompt_debug else {}),
        "save_image_start_index": save_image_start_index,
        "prompts": {
            "general_prompt": character_prompt,
            "prompt_array": prompt_array,
            "negative_prompt": str(prompt_config.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip(),
        },
        "generation": {
            "sd_type": sd_type,
            "style": style or "(No style)",
            "seed": seed,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "sa32": sa32,
            "sa64": sa64,
            "id_length": effective_id_length,
            **({"storydiffusion_internal_id_length": clean_internal_id_length} if clean_internal_id_length else {}),
            "height": height,
            "width": width,
            "style_strength_ratio": 20,
            "ip_adapter_strength": 0.5,
            "comic_type": "No typesetting (default)",
            "font_choice": "Inkfree.ttf",
        },
    }


def write_configs(args: argparse.Namespace) -> list[Path]:
    input_paths = sorted(args.input_dir.glob(args.glob))
    if args.limit:
        input_paths = input_paths[: args.limit]
    if not input_paths:
        raise FileNotFoundError(f"No input files matched {args.input_dir / args.glob}")

    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Optionally ensure AnchorBank assets exist (canonical_half_body.png) before writing configs.
    anchor_bank_summary = _load_anchor_bank_summary(args.anchor_run_dir) if args.anchor_run_dir else {}
    if args.ensure_anchors and args.anchor_run_dir:
        required_character_ids: list[str] = []
        for input_path in input_paths:
            scenes = _parse_story_file(input_path)
            required_character_ids.extend(_primary_entities(scenes))
        required_character_ids = list(dict.fromkeys(required_character_ids))
        anchor_bank_summary = _ensure_anchor_bank_generated(
            run_dir=args.anchor_run_dir,
            prompt_profile=str(args.prompt_profile),
            required_character_ids=required_character_ids,
            device=str(args.device),
            torch_dtype=str(args.torch_dtype),
        )
    config_paths = []
    for input_path in input_paths:
        config = build_probe_config(
            input_path,
            output_root=args.output_root,
            unwrap_output_dir=bool(args.unwrap_output_dir),
            storydiffusion_root=args.storydiffusion_root,
            use_reference_images=args.use_reference_images,
            reference_images=args.reference_image,
            id_length=args.id_length,
            sd_type=args.sd_type,
            style=args.style,
            seed=args.seed,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            sa32=args.sa32,
            sa64=args.sa64,
            height=args.height,
            width=args.width,
            prompt_profile=str(args.prompt_profile),
            prompt_generation_max_words=int(args.prompt_generation_max_words),
            prompt_generation_max_chars=int(args.prompt_generation_max_chars),
            anchor_bank_summary=anchor_bank_summary,
            prompt_builder_kind=str(args.prompt_builder),
            prompt_modular_backend=str(args.prompt_modular_backend),
            prompt_template_pack=(str(args.prompt_template_pack).strip() or None),
            storydiffusion_prompt_mode=str(args.storydiffusion_prompt_mode),
            save_identity_images=bool(args.save_identity_images),
        )
        prompt_debug = config.get("storydiffusion_prompt_debug")
        config_path = args.config_dir / f"{input_path.stem}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if prompt_debug:
            debug_text = json.dumps(prompt_debug, indent=2, ensure_ascii=False) + "\n"
            config_path.with_suffix(".storydiffusion_prompt_debug.json").write_text(debug_text, encoding="utf-8")
            if len(input_paths) == 1:
                config_path.with_name("storydiffusion_prompt_debug.json").write_text(debug_text, encoding="utf-8")
        config_paths.append(config_path)
    return config_paths


def run_configs(config_paths: list[Path]) -> None:
    runner = Path(__file__).with_name("run_probe.py")
    for config_path in config_paths:
        print(f"[batch] running {config_path}")
        subprocess.run([sys.executable, str(runner), "--config", str(config_path)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert test_set story files into original StoryDiffusion Gradio probe configs and optionally run them."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--glob", default="*.txt", help="Input filename glob inside --input-dir.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--unwrap-output-dir",
        action="store_true",
        help=(
            "Write manifest and images directly under --output-root instead of "
            "--output-root/<story_stem>/. Use with single-story runs (e.g. --limit 1)."
        ),
    )
    parser.add_argument("--storydiffusion-root", type=Path, default=DEFAULT_STORYDIFFUSION_ROOT)
    parser.add_argument("--run", action="store_true", help="Run run_probe.py for each generated config.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N sorted files.")
    parser.add_argument("--use-reference-images", action="store_true")
    parser.add_argument("--reference-image", action="append", default=[], help="Reference image path. Repeat once per character.")
    parser.add_argument("--save-identity-images", action="store_true", help="Save skipped native identity reference images.")
    parser.add_argument(
        "--anchor-run-dir",
        type=Path,
        default=DEFAULT_ANCHOR_RUN_DIR,
        help="Optional storygen run directory containing logs/anchor_bank.json (default: compare_aesthetic_16_low_ip).",
    )
    parser.add_argument("--ensure-anchors", action="store_true", help="If anchor images are missing, run AnchorBank first.")
    parser.add_argument("--device", default="cuda", help="Device used for anchor generation (when --ensure-anchors).")
    parser.add_argument("--torch-dtype", default="float16", help="Torch dtype used for anchor generation (when --ensure-anchors).")
    parser.add_argument("--sd-type", default="Unstable")
    parser.add_argument("--style", default="(No style)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--sa32", type=float, default=0.5)
    parser.add_argument("--sa64", type=float, default=0.5)
    parser.add_argument("--id-length", type=int, default=1)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--prompt-profile", default=DEFAULT_PROFILE, help="Runtime profile used for prompt_builder config.")
    parser.add_argument(
        "--prompt-builder",
        choices=("legacy", "modular"),
        default="legacy",
        help="Rule prompt assembly: legacy PromptBuilder only, or modular stack (see prompt.builder in configs/base.yaml).",
    )
    parser.add_argument(
        "--prompt-modular-backend",
        choices=("sdxl", "storydiffusion"),
        default="sdxl",
        help="When --prompt-builder=modular: sdxl matches legacy strings; storydiffusion strips SDXL spatial hacks.",
    )
    parser.add_argument(
        "--prompt-template-pack",
        default="",
        help="Optional template pack name under configs/prompt_templates/<name>.yaml (default: profile prompt.modular.template_pack).",
    )
    parser.add_argument(
        "--storydiffusion-prompt-mode",
        choices=("current", "clean", "clean_v2", "natural"),
        default="current",
        help=(
            "Native StoryDiffusion prompt rendering mode. Default preserves the existing generation_prompt prefix behavior; "
            "natural keeps clean_v2 identity prompts but renders shorter storyboard-style scene prompts."
        ),
    )
    parser.add_argument("--prompt-generation-max-words", type=int, default=60)
    parser.add_argument("--prompt-generation-max-chars", type=int, default=420)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.config_dir = args.config_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.unwrap_output_dir and not args.limit:
        raise SystemExit("--unwrap-output-dir requires --limit (single-story batch) to avoid overwriting outputs.")
    args.storydiffusion_root = args.storydiffusion_root.resolve()
    config_paths = write_configs(args)
    print("Generated configs:")
    for config_path in config_paths:
        print(f"  {config_path}")
    if args.run:
        run_configs(config_paths)


if __name__ == "__main__":
    main()
