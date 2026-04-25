from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from storygen.types import CharacterSpec, Story

MALE_PRONOUNS = {"he", "him", "his"}
FEMALE_PRONOUNS = {"she", "her", "hers"}
NEUTRAL_PRONOUNS = {"they", "them", "their", "theirs", "it"}


CHARACTER_SPEC_FIELDS = {
    "age_band",
    "gender_presentation",
    "hair_color",
    "hairstyle",
    "skin_tone",
    "body_build",
    "signature_outfit",
    "signature_accessory",
    "profession_marker",
}

LEADING_PRONOUN_PATTERN = re.compile(r"^(she|he|they|it)\b", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _story_character_ids(story: Story) -> list[str]:
    source_entities = story.recurring_entities or story.all_entities
    seen = set()
    character_ids = []
    for value in source_entities:
        character_id = _normalize_text(value)
        key = character_id.lower()
        if character_id and key not in seen:
            seen.add(key)
            character_ids.append(character_id)
    return character_ids


def _infer_story_pronoun(story: Story) -> str | None:
    for scene in story.scenes:
        match = LEADING_PRONOUN_PATTERN.match(scene.clean_text)
        if match:
            return match.group(1).lower()
    return None


def _infer_gender_presentation(story: Story) -> str | None:
    pronoun = _infer_story_pronoun(story)
    if pronoun == "he":
        return "male"
    if pronoun == "she":
        return "female"
    return None


def character_specs_to_metadata(character_specs: list[CharacterSpec]) -> dict[str, dict[str, Any]]:
    return {spec.character_id: asdict(spec) for spec in character_specs if spec.character_id}


def build_rule_based_character_specs(story: Story) -> dict[str, dict[str, Any]]:
    character_ids = _story_character_ids(story)
    if not character_ids:
        return {}
    inferred_gender = _infer_gender_presentation(story)
    return character_specs_to_metadata(
        [
            CharacterSpec(
                character_id=character_id,
                gender_presentation=inferred_gender,
                profession_marker=character_id.lower(),
                metadata={
                    "source": "rule_based",
                    "confidence": "minimal",
                    "scene_ids": story.entity_to_scene_ids.get(character_id, []),
                    "inferred_pronoun_gender": inferred_gender,
                },
            )
            for character_id in character_ids
        ]
    )


def normalize_llm_character_payload(payload: dict[str, Any]) -> dict[str, Any]:
    character_id = _normalize_text(payload.get("character_id"))
    if not character_id:
        raise ValueError("LLM character spec is missing character_id")

    normalized = {"character_id": character_id}
    for field_name in CHARACTER_SPEC_FIELDS:
        value = _normalize_text(payload.get(field_name))
        normalized[field_name] = value or None
    normalized["metadata"] = {
        "source": "llm_assisted",
        "confidence": "structured_identity",
    }
    return normalized


def build_llm_character_specs(structured_output: dict[str, Any], story: Story) -> dict[str, dict[str, Any]]:
    global_payload = structured_output.get("global", {})
    characters_payload = global_payload.get("characters", [])
    if not isinstance(characters_payload, list):
        raise ValueError("LLM global.characters must be a list")

    character_specs = []
    seen = set()
    for character_payload in characters_payload:
        if not isinstance(character_payload, dict):
            raise ValueError("Each LLM character spec must be an object")
        normalized = normalize_llm_character_payload(character_payload)
        key = normalized["character_id"].lower()
        if key in seen:
            continue
        seen.add(key)
        character_specs.append(CharacterSpec(**normalized))

    main_character = _normalize_text(global_payload.get("main_character"))
    if main_character and main_character.lower() not in seen:
        raise ValueError(f"LLM character specs do not include main_character: {main_character}")
    if not character_specs and _story_character_ids(story):
        raise ValueError("LLM character specs are empty")

    return character_specs_to_metadata(character_specs)
