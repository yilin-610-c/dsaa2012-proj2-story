from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from storygen.native_prompting.types import ALLOWED_SUBJECT_TYPES, ALLOWED_TARGET_BACKENDS, NativePromptPayload
from storygen.types import Story


TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")
LEADING_PRONOUN_PATTERN = re.compile(r"^(she|he|they|it)\b", re.IGNORECASE)
IDENTITY_BANNED_TERMS = ("character sheet", "turnaround", "multiple views", "duplicate subject")
BAD_PLACEHOLDERS = ("unknown", "unspecified")
SINGLE_SUBJECT_TERMS = ("single", "single-subject", "single subject", "one ", "solo", "alone")
REFERENCE_FRAMING_TERMS = ("centered", "simple background", "plain background", "clean background")
REFERENCE_ONLY_SCENE_TERMS = (
    "simple background",
    "plain background",
    "clean background",
    "centered background",
    "simple, centered background",
    "single-subject",
    "single subject",
    "one person in the image",
    "one animal in the image",
    "one subject in the image",
    "no distractions in the background",
)


@dataclass(slots=True)
class NativePromptValidationError(Exception):
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return "; ".join(self.errors)


@dataclass(slots=True)
class NativePromptValidationConfig:
    max_scene_prompt_words: int = 40
    max_scene_prompt_chars: int = 320
    max_identity_prompt_words: int = 60
    max_identity_prompt_chars: int = 420
    min_unknown_identity_words: int = 4
    min_storydiffusion_scene_words: int = 6


def normalize_targets(targets: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized = [str(target).strip().lower() for target in (targets or []) if str(target).strip()]
    if not normalized:
        normalized = ["anchor"]
    return normalized


def validate_native_prompt_payload(
    payload: NativePromptPayload,
    story: Story,
    *,
    targets: list[str] | None = None,
    config: NativePromptValidationConfig | None = None,
) -> list[str]:
    validation_config = config or NativePromptValidationConfig()
    requested_targets = normalize_targets(targets or payload.target_backends)
    errors: list[str] = []
    unknown_targets = [target for target in requested_targets if target not in ALLOWED_TARGET_BACKENDS]
    if unknown_targets:
        errors.append(f"Unknown target_backends: {unknown_targets}")
    payload_targets = normalize_targets(payload.target_backends)
    missing_targets = [target for target in requested_targets if target not in payload_targets]
    if missing_targets:
        errors.append(f"Payload target_backends missing requested targets: {missing_targets}")

    story_entities = list(story.all_entities or [])
    known_entities = set(story_entities)
    character_ids = [character.character_id for character in payload.characters]
    character_id_set = set(character_ids)
    if not known_entities:
        known_entities = set(character_ids)
    if story_entities:
        missing_characters = [entity for entity in story_entities if entity not in character_id_set]
        if missing_characters:
            errors.append(f"Missing characters for story entities: {missing_characters}")
    duplicate_characters = sorted({character_id for character_id in character_ids if character_ids.count(character_id) > 1})
    if duplicate_characters:
        errors.append(f"Duplicate character ids: {duplicate_characters}")

    for character in payload.characters:
        if not character.character_id:
            errors.append("Character has empty character_id")
        elif known_entities and character.character_id not in known_entities:
            errors.append(f"Unknown character_id: {character.character_id}")
        if character.subject_type not in ALLOWED_SUBJECT_TYPES:
            errors.append(f"Invalid subject_type for {character.character_id}: {character.subject_type}")
        _validate_text_field(
            errors,
            f"characters[{character.character_id}].stable_identity",
            character.stable_identity,
            max_words=validation_config.max_identity_prompt_words,
            max_chars=validation_config.max_identity_prompt_chars,
        )
        if character.subject_type == "unknown":
            _validate_unknown_identity(
                errors,
                f"characters[{character.character_id}].stable_identity",
                character.stable_identity,
                character.character_id,
                min_words=validation_config.min_unknown_identity_words,
            )
        if "anchor" in requested_targets:
            _validate_text_field(
                errors,
                f"characters[{character.character_id}].anchor_reference_prompt",
                character.anchor_reference_prompt,
                max_words=validation_config.max_identity_prompt_words,
                max_chars=validation_config.max_identity_prompt_chars,
            )
            _validate_anchor_reference_prompt(errors, character.character_id, character.anchor_reference_prompt)
            if character.subject_type == "unknown":
                _validate_unknown_identity(
                    errors,
                    f"characters[{character.character_id}].anchor_reference_prompt",
                    character.anchor_reference_prompt,
                    character.character_id,
                    min_words=validation_config.min_unknown_identity_words,
                )

    expected_scene_ids = [scene.scene_id for scene in story.scenes]
    actual_scene_ids = [scene.scene_id for scene in payload.scenes]
    if actual_scene_ids != expected_scene_ids:
        errors.append(f"Scene ids do not match parsed story: expected {expected_scene_ids}, got {actual_scene_ids}")
    scene_by_id = {scene.scene_id: scene for scene in payload.scenes}
    parsed_scene_by_id = {scene.scene_id: scene for scene in story.scenes}
    for scene_id in expected_scene_ids:
        scene = scene_by_id.get(scene_id)
        parsed_scene = parsed_scene_by_id[scene_id]
        if scene is None:
            continue
        if "anchor" in requested_targets:
            _validate_text_field(
                errors,
                f"scenes[{scene_id}].anchor_generation_prompt",
                scene.anchor_generation_prompt,
                max_words=validation_config.max_scene_prompt_words,
                max_chars=validation_config.max_scene_prompt_chars,
            )
            _validate_scene_prompt_not_reference_only(errors, f"scenes[{scene_id}].anchor_generation_prompt", scene.anchor_generation_prompt)
            _validate_text_field(
                errors,
                f"scenes[{scene_id}].scoring_prompt",
                scene.scoring_prompt,
                max_words=20,
                max_chars=180,
            )
            _validate_visible_ids(errors, scene_id, scene.visible_character_ids, known_entities)
            _validate_identity_conditioning_id(errors, scene_id, scene.identity_conditioning_subject_id, known_entities)
            for entity in parsed_scene.entities:
                if entity not in scene.visible_character_ids:
                    errors.append(f"scenes[{scene_id}].visible_character_ids missing explicit entity {entity}")
            if _is_pronoun_only_scene(parsed_scene.clean_text, parsed_scene.entities) and not scene.visible_character_ids:
                errors.append(f"scenes[{scene_id}].visible_character_ids must resolve pronoun-only visual scene")
        if "storydiffusion" in requested_targets:
            _validate_text_field(
                errors,
                f"scenes[{scene_id}].storydiffusion_prompt",
                scene.storydiffusion_prompt,
                max_words=validation_config.max_scene_prompt_words,
                max_chars=validation_config.max_scene_prompt_chars,
            )
            _validate_scene_prompt_not_reference_only(
                errors,
                f"scenes[{scene_id}].storydiffusion_prompt",
                scene.storydiffusion_prompt,
            )
            _validate_storydiffusion_scene_detail(
                errors,
                scene_id,
                scene.storydiffusion_prompt,
                min_words=validation_config.min_storydiffusion_scene_words,
            )
            _validate_storydiffusion_scene_prompt(errors, scene_id, scene.storydiffusion_prompt, parsed_scene.entities, known_entities)
            if _is_pronoun_only_scene(parsed_scene.clean_text, parsed_scene.entities) and not scene.storydiffusion_prompt.startswith("[NC]"):
                tags = TAG_PATTERN.findall(scene.storydiffusion_prompt)
                if not any(tag in known_entities for tag in tags):
                    errors.append(f"scenes[{scene_id}].storydiffusion_prompt must resolve pronoun-only scene to a known tag or [NC]")

    if "storydiffusion" in requested_targets:
        storydiffusion = payload.storydiffusion
        _validate_text_field(
            errors,
            "storydiffusion.general_prompt",
            storydiffusion.general_prompt,
            max_words=max(validation_config.max_identity_prompt_words, len(payload.characters) * 24),
            max_chars=max(validation_config.max_identity_prompt_chars, len(payload.characters) * 180),
        )
        _validate_storydiffusion_general_prompt_tags(
            errors,
            payload.storydiffusion.general_prompt,
            character_ids,
        )
        if not storydiffusion.identity_reference_prompts:
            errors.append("storydiffusion.identity_reference_prompts is required")
        for index, prompt in enumerate(storydiffusion.identity_reference_prompts):
            _validate_text_field(
                errors,
                f"storydiffusion.identity_reference_prompts[{index}]",
                prompt,
                max_words=validation_config.max_identity_prompt_words,
                max_chars=validation_config.max_identity_prompt_chars,
            )
            _validate_identity_reference_prompt(errors, f"storydiffusion.identity_reference_prompts[{index}]", prompt)
            _validate_storydiffusion_identity_reference_tags(
                errors,
                f"storydiffusion.identity_reference_prompts[{index}]",
                prompt,
                known_entities,
            )
        if storydiffusion.identity_prompts_per_character < 1:
            errors.append("storydiffusion.identity_prompts_per_character must be >= 1")
        expected_identity_count = len(payload.characters) * storydiffusion.identity_prompts_per_character
        if storydiffusion.identity_reference_prompts and len(storydiffusion.identity_reference_prompts) != expected_identity_count:
            errors.append(
                "storydiffusion.identity_reference_prompts count must equal "
                f"character_count * identity_prompts_per_character: expected {expected_identity_count}, "
                f"got {len(storydiffusion.identity_reference_prompts)}"
            )
    return errors


def raise_if_invalid(errors: list[str]) -> None:
    if errors:
        raise NativePromptValidationError(errors)


def _validate_text_field(errors: list[str], field_name: str, value: str, *, max_words: int, max_chars: int) -> None:
    text = str(value or "").strip()
    if not text:
        errors.append(f"{field_name} is required")
        return
    if len(text) > max_chars:
        errors.append(f"{field_name} exceeds {max_chars} chars")
    if len(_words(text)) > max_words:
        errors.append(f"{field_name} exceeds {max_words} words")
    lowered = text.lower()
    for placeholder in BAD_PLACEHOLDERS:
        if placeholder in lowered:
            errors.append(f"{field_name} contains {placeholder}")


def _validate_visible_ids(errors: list[str], scene_id: str, values: list[str], known_entities: set[str]) -> None:
    if not isinstance(values, list):
        errors.append(f"scenes[{scene_id}].visible_character_ids must be a list")
        return
    for value in values:
        if value not in known_entities:
            errors.append(f"scenes[{scene_id}].visible_character_ids contains unknown entity {value}")


def _validate_identity_conditioning_id(errors: list[str], scene_id: str, value: str | None, known_entities: set[str]) -> None:
    if value is not None and value not in known_entities:
        errors.append(f"scenes[{scene_id}].identity_conditioning_subject_id contains unknown entity {value}")


def _validate_anchor_reference_prompt(errors: list[str], character_id: str, prompt: str) -> None:
    _validate_identity_reference_prompt(errors, f"characters[{character_id}].anchor_reference_prompt", prompt)
    lowered = prompt.lower()
    if not any(term in lowered for term in SINGLE_SUBJECT_TERMS):
        errors.append(f"characters[{character_id}].anchor_reference_prompt must request a single subject")
    if not any(term in lowered for term in REFERENCE_FRAMING_TERMS):
        errors.append(f"characters[{character_id}].anchor_reference_prompt must include simple/centered reference framing")


def _validate_identity_reference_prompt(errors: list[str], field_name: str, prompt: str) -> None:
    lowered = prompt.lower()
    for term in IDENTITY_BANNED_TERMS:
        if term in lowered:
            errors.append(f"{field_name} contains banned identity-reference term: {term}")


def _validate_scene_prompt_not_reference_only(errors: list[str], field_name: str, prompt: str) -> None:
    lowered = prompt.lower()
    for term in REFERENCE_ONLY_SCENE_TERMS:
        if term in lowered:
            errors.append(f"{field_name} contains reference-only phrase: {term}")


def _validate_storydiffusion_scene_detail(errors: list[str], scene_id: str, prompt: str, *, min_words: int) -> None:
    if prompt.strip().startswith("[NC]"):
        return
    text = TAG_PATTERN.sub(" ", prompt)
    words = _words(text)
    if len(words) < min_words:
        errors.append(
            f"scenes[{scene_id}].storydiffusion_prompt is too short for a visual frame prompt; "
            f"expected at least {min_words} words after tags"
        )


def _validate_storydiffusion_general_prompt_tags(errors: list[str], prompt: str, character_ids: list[str]) -> None:
    tags = set(TAG_PATTERN.findall(prompt))
    for character_id in character_ids:
        if character_id not in tags:
            errors.append(f"storydiffusion.general_prompt missing tag [{character_id}]")


def _validate_storydiffusion_identity_reference_tags(
    errors: list[str],
    field_name: str,
    prompt: str,
    known_entities: set[str],
) -> None:
    prompt = prompt.strip()
    tags = TAG_PATTERN.findall(prompt)
    if not tags:
        errors.append(f"{field_name} must include an exact [Character] tag")
        return
    for tag in tags:
        if tag not in known_entities:
            errors.append(f"{field_name} contains unknown tag [{tag}]")


def _validate_storydiffusion_scene_prompt(
    errors: list[str],
    scene_id: str,
    prompt: str,
    explicit_entities: list[str],
    known_entities: set[str],
) -> None:
    prompt = prompt.strip()
    if not prompt:
        return
    if not (prompt.startswith("[NC]") or re.match(r"^(?:\[[^\[\]]+\]\s*)+", prompt)):
        errors.append(f"scenes[{scene_id}].storydiffusion_prompt must start with [Character] tags or [NC]")
        return
    tags = TAG_PATTERN.findall(prompt)
    for tag in tags:
        if tag != "NC" and tag not in known_entities:
            errors.append(f"scenes[{scene_id}].storydiffusion_prompt contains unknown tag [{tag}]")
    if explicit_entities:
        for entity in explicit_entities:
            if entity not in tags:
                errors.append(f"scenes[{scene_id}].storydiffusion_prompt missing explicit tag [{entity}]")


def _validate_unknown_identity(
    errors: list[str],
    field_name: str,
    value: str,
    character_id: str,
    *,
    min_words: int,
) -> None:
    words = _words(value)
    if len(words) < min_words:
        errors.append(f"{field_name} is too short for subject_type=unknown")
    stripped = TAG_PATTERN.sub("", value).strip(" ,.;")
    if stripped.lower() == character_id.lower() or not stripped:
        errors.append(f"{field_name} is only a tag or character name")


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_-]+", str(value or ""))


def _is_pronoun_only_scene(clean_text: str, explicit_entities: list[str]) -> bool:
    return not explicit_entities and bool(LEADING_PRONOUN_PATTERN.search(str(clean_text or "").strip()))


def validation_metadata(errors: list[str]) -> dict[str, Any]:
    return {"error_count": len(errors), "errors": list(errors)}
