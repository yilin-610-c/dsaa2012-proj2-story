from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from storygen.character_specs import (
    build_character_identity_snippet,
    build_llm_character_specs,
    build_rule_based_character_specs,
)
from storygen.llm_client import BaseLLMClient, build_llm_client
from storygen.prompt_builder import PromptBuilder
from storygen.prompt_cache import PromptCache, build_cache_record, build_prompt_cache_key
from storygen.types import PromptSpec, Story


class LLMPromptError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen = set()
    normalized = []
    for item in value:
        text = _normalize_text(item)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            normalized.append(text)
    return normalized


def _normalize_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _normalize_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trim_words(text: str, max_words: int | None) -> str:
    if not max_words:
        return text
    words = text.split()
    return " ".join(words[:max_words])


def _trim_text(text: str, *, max_words: int | None, max_chars: int | None) -> str:
    trimmed = _trim_words(_normalize_text(text), max_words)
    if max_chars and len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rsplit(" ", 1)[0].strip() or trimmed[:max_chars].strip()
    return trimmed.strip(" ,;")


def _join_parts(parts: list[str]) -> str:
    seen = set()
    output = []
    for part in parts:
        text = _normalize_text(part)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return ", ".join(output)


def _strip_leading_prompt_verb(text: str) -> str:
    cleaned = _normalize_text(text)
    cleaned = re.sub(
        r"^(illustrate|show|depict|create|generate|draw|paint|render)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _strip_question_frame(text: str) -> str:
    cleaned = _normalize_text(text).rstrip(" ?")
    patterns = [
        r"^does the image show\s+",
        r"^does the image capture\s+",
        r"^is there\s+",
        r"^is the image of\s+",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _filter_repeated_style_cues(style_cues: list[str], style_prompt: str) -> list[str]:
    style_parts = {_normalize_text(part).lower() for part in style_prompt.split(",") if _normalize_text(part)}
    style_full = _normalize_text(style_prompt).lower()
    filtered = []
    for cue in style_cues:
        cue_key = cue.lower()
        if cue_key in style_parts or cue_key == style_full:
            continue
        filtered.append(cue)
    return filtered


def _contains_human_identity(character_prompt: str) -> bool:
    lowered = character_prompt.lower()
    human_terms = [
        "woman",
        "girl",
        "man",
        "boy",
        "person",
        "human",
        "lady",
        "male",
        "female",
    ]
    return any(term in lowered for term in human_terms)


ANIMAL_IDENTITY_TERMS = {
    "animal",
    "bird",
    "cat",
    "dog",
    "horse",
    "fox",
    "wolf",
    "bear",
    "rabbit",
    "lion",
    "tiger",
    "deer",
    "fish",
    "puppy",
    "pet",
}


def _contains_animal_identity(character_prompt: str) -> bool:
    lowered = character_prompt.lower()
    animal_terms = [
        *sorted(ANIMAL_IDENTITY_TERMS),
        "wings",
        "feather",
        "fur",
    ]
    return any(term in lowered for term in animal_terms)


def _ensure_human_identity(character_prompt: str) -> str:
    if not character_prompt:
        return character_prompt
    if _contains_human_identity(character_prompt):
        return character_prompt
    if _contains_animal_identity(character_prompt):
        return character_prompt
    return _join_parts(["human person", character_prompt])


def _strip_leading_subject_reference(text: str, subject_name: str) -> str:
    cleaned = _normalize_text(text)
    subject_name = _normalize_text(subject_name)
    if not cleaned or not subject_name:
        return cleaned
    return re.sub(
        rf"^{re.escape(subject_name)}\b(?:\s+(?:is|are))?\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" ,;")


def _strip_leading_disallowed_animal_reference(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return cleaned
    animal_pattern = "|".join(re.escape(term) for term in sorted(ANIMAL_IDENTITY_TERMS, key=len, reverse=True))
    return re.sub(
        rf"^(?:a|an|the)?\s*(?:{animal_pattern})\b(?:\s+(?:is|are))?\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" ,;")


def _identity_subject_name(character_prompt: str) -> str:
    return _normalize_text(character_prompt.split(",", 1)[0])


def _prepare_scene_generation_phrase(
    generation_prompt: str,
    *,
    character_prompt: str,
    allow_animal_identity: bool,
    strip_subject: bool,
) -> str:
    cleaned = _normalize_text(generation_prompt)
    if not allow_animal_identity:
        cleaned = _strip_leading_disallowed_animal_reference(cleaned)
    if strip_subject:
        cleaned = _strip_leading_subject_reference(cleaned, _identity_subject_name(character_prompt))
    return cleaned


def _merge_identity_into_generation_prompt(
    character_prompt: str,
    generation_prompt: str,
    max_words: int,
    max_chars: int,
    *,
    allow_animal_identity: bool = True,
) -> str:
    identity_prompt = _ensure_human_identity(character_prompt)
    scene_prompt = _prepare_scene_generation_phrase(
        generation_prompt,
        character_prompt=identity_prompt,
        allow_animal_identity=allow_animal_identity,
        strip_subject=True,
    )
    merged = _join_parts([identity_prompt, scene_prompt])
    return _trim_text(merged, max_words=max_words, max_chars=max_chars)


def _contains_visual_animal_identity(character_spec: dict[str, Any], identity_cues: list[str]) -> bool:
    text = " ".join(
        [
            _normalize_text(character_spec.get("character_id")),
            _normalize_text(character_spec.get("body_build")),
            _normalize_text(character_spec.get("signature_accessory")),
            *_normalize_list(identity_cues),
        ]
    )
    return _contains_animal_identity(text)


def _lightweight_human_label(character_spec: dict[str, Any], identity_cues: list[str]) -> str:
    for cue in identity_cues:
        normalized = _normalize_text(cue)
        if normalized.lower() in {"human woman", "human girl", "human man", "human boy", "human person"}:
            return normalized

    age = _normalize_text(character_spec.get("age_band")).lower()
    gender = _normalize_text(character_spec.get("gender_presentation")).lower()
    if any(term in gender for term in ["woman", "female"]):
        return "human girl" if age in {"child", "kid", "young"} else "human woman"
    if any(term in gender for term in ["man", "male"]):
        return "human boy" if age in {"child", "kid", "young"} else "human man"
    return "human person"


def build_lightweight_character_prompt(
    character_id: str,
    character_specs_by_id: dict[str, dict[str, Any]],
    identity_cues: list[str],
    *,
    allow_animal_identity: bool = True,
    human_label_override: str | None = None,
) -> str:
    character_id = _normalize_text(character_id)
    if not character_id:
        return ""
    character_spec = character_specs_by_id.get(character_id, {"character_id": character_id})
    if allow_animal_identity and _contains_visual_animal_identity(character_spec, identity_cues):
        animal_cues = [
            _normalize_text(cue)
            for cue in identity_cues
            if _contains_animal_identity(cue)
        ]
        return _join_parts([character_id, animal_cues[0] if animal_cues else ""])
    llm_label = _lightweight_human_label(character_spec, identity_cues)
    if llm_label != "human person":
        return _join_parts([character_id, llm_label])
    if human_label_override:
        return _join_parts([character_id, human_label_override])
    return _join_parts([character_id, llm_label])


def _natural_language_list(parts: list[str]) -> str:
    items = [_normalize_text(part) for part in parts if _normalize_text(part)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _indefinite_article(phrase: str) -> str:
    normalized = _normalize_text(phrase)
    if not normalized:
        return ""
    return "an" if normalized[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _build_dual_primary_character_snippet(character_spec: dict[str, Any]) -> str:
    name = _normalize_text(character_spec.get("character_id"))
    if not name:
        return ""
    if _is_anonymous_character_spec(character_spec):
        return ""

    descriptor_parts = " ".join(
        _normalize_text(part)
        for part in [
            character_spec.get("age_band"),
            character_spec.get("gender_presentation"),
        ]
        if _normalize_text(part)
    )

    hair_color = _normalize_text(character_spec.get("hair_color"))
    hairstyle = _normalize_text(character_spec.get("hairstyle"))
    hair_phrase = ""
    if hair_color and hairstyle:
        if "hair" in hairstyle.lower():
            hair_phrase = re.sub(r"\bhair\b", f"{hair_color} hair", hairstyle, count=1, flags=re.IGNORECASE).strip()
        else:
            hair_phrase = f"{hairstyle} {hair_color} hair".strip()
    elif hairstyle:
        hair_phrase = hairstyle if "hair" in hairstyle.lower() else f"{hairstyle} hair"
    elif hair_color:
        hair_phrase = f"{hair_color} hair"

    body_details = [
        f"with {hair_phrase}" if hair_phrase else "",
        f"{_normalize_text(character_spec.get('skin_tone'))} skin" if _normalize_text(character_spec.get("skin_tone")) else "",
        f"{_normalize_text(character_spec.get('body_build'))} build" if _normalize_text(character_spec.get("body_build")) else "",
    ]

    clothing_parts = [
        _normalize_text(character_spec.get("signature_outfit")),
        _normalize_text(character_spec.get("signature_accessory")),
        _normalize_text(character_spec.get("profession_marker")),
    ]
    clothing_phrase = _natural_language_list(clothing_parts)

    sentence_parts = [f"{name} is"]
    if descriptor_parts:
        sentence_parts.append(f"{_indefinite_article(descriptor_parts)} {descriptor_parts}")
    if any(body_details):
        sentence_parts.append(_natural_language_list(body_details))
    sentence = " ".join(sentence_parts).strip()
    if clothing_phrase:
        sentence = f"{sentence}, wearing {clothing_phrase}"
    return sentence.rstrip(", ")


def _is_anonymous_character_spec(character_spec: dict[str, Any]) -> bool:
    character_id = _normalize_text(character_spec.get("character_id")).lower()
    if character_id not in {"friend", "companion", "person", "someone", "stranger"}:
        return False
    meaningful_fields = [
        _normalize_text(character_spec.get("hair_color")),
        _normalize_text(character_spec.get("hairstyle")),
        _normalize_text(character_spec.get("skin_tone")),
        _normalize_text(character_spec.get("body_build")),
        _normalize_text(character_spec.get("signature_outfit")),
        _normalize_text(character_spec.get("signature_accessory")),
        _normalize_text(character_spec.get("profession_marker")),
    ]
    gender = _normalize_text(character_spec.get("gender_presentation")).lower()
    return gender in {"", "unknown", "unspecified"} and not any(meaningful_fields)


def _anonymous_character_ids(character_specs_by_id: dict[str, dict[str, Any]]) -> set[str]:
    return {
        character_id.lower()
        for character_id, character_spec in character_specs_by_id.items()
        if _is_anonymous_character_spec(character_spec)
    }


def _is_anonymous_character_id(character_id: str, anonymous_ids: set[str]) -> bool:
    return _normalize_text(character_id).lower() in anonymous_ids


def _story_explicitly_marks_animal(story: Story, character_id: str) -> bool:
    character_id = _normalize_text(character_id)
    if not character_id:
        return False
    lowered_id = character_id.lower()
    if lowered_id in ANIMAL_IDENTITY_TERMS:
        return True
    raw_text = _normalize_text(story.raw_text).lower()
    animal_pattern = "|".join(re.escape(term) for term in sorted(ANIMAL_IDENTITY_TERMS, key=len, reverse=True))
    tagged_name = re.escape(f"<{character_id}>".lower())
    name = re.escape(lowered_id)
    patterns = [
        rf"{tagged_name}[^.!\n]*\b(?:{animal_pattern})\b",
        rf"\b(?:{animal_pattern})\b[^.!\n]*{tagged_name}",
        rf"\b{name}\b[^.!\n]*\b(?:{animal_pattern})\b",
        rf"\b(?:{animal_pattern})\b[^.!\n]*\b{name}\b",
    ]
    return any(re.search(pattern, raw_text) for pattern in patterns)


def _human_pronoun_label_for_story(story: Story, character_id: str) -> str | None:
    if _story_explicitly_marks_animal(story, character_id):
        return None
    raw_text = _normalize_text(story.raw_text)
    tagged_position = raw_text.lower().find(f"<{character_id}>".lower())
    search_text = raw_text[tagged_position:] if tagged_position >= 0 else raw_text
    if re.search(r"\b(he|him|his)\b", search_text, flags=re.IGNORECASE):
        return "human man"
    if re.search(r"\b(she|her|hers)\b", search_text, flags=re.IGNORECASE):
        return "human woman"
    lowered_id = _normalize_text(character_id).lower()
    if lowered_id in {"girl", "woman", "lady"}:
        return "human girl" if lowered_id == "girl" else "human woman"
    if lowered_id in {"boy", "man"}:
        return "human boy" if lowered_id == "boy" else "human man"
    return None


def _explicit_animal_subject_ids(story: Story, character_specs_by_id: dict[str, dict[str, Any]]) -> set[str]:
    return {
        character_id.lower()
        for character_id in character_specs_by_id
        if _story_explicitly_marks_animal(story, character_id)
    }


def _sanitize_character_specs_for_story(story: Story, characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for character_spec in characters:
        character_id = _normalize_text(character_spec.get("character_id"))
        human_label = _human_pronoun_label_for_story(story, character_id)
        if not human_label or not _contains_visual_animal_identity(character_spec, []):
            sanitized.append(character_spec)
            continue
        updated = dict(character_spec)
        updated["gender_presentation"] = "man" if human_label == "human man" else "woman" if human_label == "human woman" else "person"
        for field in ["body_build", "signature_accessory", "profession_marker"]:
            if _contains_animal_identity(_normalize_text(updated.get(field))):
                updated[field] = ""
        sanitized.append(updated)
    return sanitized


def _dual_primary_name_phrase(visible_character_ids: list[str]) -> str:
    display_names = [_normalize_text(character_id) for character_id in visible_character_ids if _normalize_text(character_id)]
    if not display_names:
        return ""
    if len(display_names) == 1:
        return display_names[0]
    if len(display_names) == 2:
        return f"{display_names[0]} and {display_names[1]}"
    return f"{', '.join(display_names[:-1])}, and {display_names[-1]}"


def _with_explicit_dual_primary_names(interaction_summary: str, visible_character_ids: list[str]) -> str:
    summary = _normalize_text(interaction_summary)
    if not summary:
        return ""
    name_phrase = _dual_primary_name_phrase(visible_character_ids)
    if not name_phrase:
        return summary
    if re.match(r"^they\b", summary, flags=re.IGNORECASE):
        return re.sub(r"^they\b", name_phrase, summary, count=1, flags=re.IGNORECASE)
    return summary


def _format_dual_primary_setting_focus(setting_focus: str | None) -> str:
    normalized = _normalize_text(setting_focus)
    lowered = normalized.lower()
    if not normalized:
        return ""
    if lowered.startswith(("in ", "at ", "on ", "inside ", "near ")):
        return normalized
    if "cafe table" in lowered:
        return "at a cafe table"
    if "exhibition wall" in lowered:
        return "at an exhibition wall"
    if lowered == "park":
        return "in a park"
    if lowered == "cafe":
        return "in a cafe"
    if lowered == "exhibition":
        return "at an exhibition"
    return normalized


def _starts_with_unresolved_pronoun(text: str) -> bool:
    return bool(re.match(r"^(he|she|they|it|his|her|their)\b", _normalize_text(text), flags=re.IGNORECASE))


def _resolved_scene_text(
    clean_text: str,
    generation_prompt: str,
    primary_action: str,
    visible_character_ids: list[str],
) -> str:
    clean = _normalize_text(clean_text)
    if not _starts_with_unresolved_pronoun(clean):
        return clean
    candidate_parts = [_normalize_text(primary_action), _normalize_text(generation_prompt)]
    visible_names = [_normalize_text(character_id) for character_id in visible_character_ids if _normalize_text(character_id)]
    for candidate in candidate_parts:
        if any(re.search(rf"\b{re.escape(name)}\b", candidate, flags=re.IGNORECASE) for name in visible_names):
            return candidate
    for candidate in candidate_parts:
        if candidate:
            return candidate
    return clean


def build_dual_primary_generation_prompt(
    visible_character_ids: list[str],
    character_specs_by_id: dict[str, dict[str, Any]],
    interaction_summary: str,
    spatial_relation: str,
    framing: str,
    setting_focus: str | None,
    *,
    fallback_character_prompt: str,
    max_words: int,
    max_chars: int,
) -> str:
    character_sentences = [
        _build_dual_primary_character_snippet(character_specs_by_id[character_id])
        for character_id in visible_character_ids
        if character_id in character_specs_by_id
    ]
    if not character_sentences and fallback_character_prompt:
        character_sentences = [_ensure_human_identity(fallback_character_prompt)]

    scene_sentences = [_with_explicit_dual_primary_names(interaction_summary, visible_character_ids)]
    framing_clause = _normalize_text(framing)
    spatial_clause = _normalize_text(spatial_relation)
    setting_clause = _format_dual_primary_setting_focus(setting_focus)
    if framing_clause and spatial_clause:
        scene_sentences.append(f"{framing_clause}, {spatial_clause}")
    elif framing_clause:
        scene_sentences.append(framing_clause)
    elif spatial_clause:
        scene_sentences.append(spatial_clause)
    if setting_clause:
        scene_sentences.append(setting_clause)

    prompt = ". ".join(
        sentence.rstrip(".")
        for sentence in [*character_sentences, *scene_sentences]
        if _normalize_text(sentence)
    ).strip()
    if prompt and not prompt.endswith("."):
        prompt = f"{prompt}."
    return _trim_text(prompt, max_words=max_words, max_chars=max_chars)


def _merge_human_negative_prompt(character_prompt: str, negative_prompt: str) -> str:
    if not _contains_human_identity(_ensure_human_identity(character_prompt)):
        return negative_prompt
    return _join_parts([negative_prompt, "cat, dog, pet animal"])


def _normalize_route_factors(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LLMPromptError("LLM route_factors must be an object")
    return {
        "same_subject": _normalize_bool(value.get("same_subject")),
        "same_setting": _normalize_bool(value.get("same_setting")),
        "body_state_change": _normalize_bool(value.get("body_state_change")),
        "primary_action_change": _normalize_bool(value.get("primary_action_change")),
        "new_key_objects": _normalize_list(value.get("new_key_objects")),
        "composition_change_needed": _normalize_bool(value.get("composition_change_needed")),
    }


def _adjust_route_change_level(route_change_level: str, route_factors: dict[str, Any]) -> tuple[str, str | None]:
    if not route_factors.get("same_setting", True) and route_factors.get("composition_change_needed"):
        return "large", "setting_change_and_composition_change_needed"
    if route_change_level == "small" and (
        route_factors.get("body_state_change")
        or route_factors.get("primary_action_change")
        or route_factors.get("composition_change_needed")
    ):
        return "medium", "small_inconsistent_with_route_factors"
    return route_change_level, None


def _has_body_action_transition(scene_text: str) -> bool:
    lowered = _normalize_text(scene_text).lower()
    transition_patterns = [
        r"\bgets?\s+inside\b",
        r"\benters?\b",
        r"\bleaves?\b",
        r"\bsits?\b",
        r"\bsitting\b",
        r"\bmoves?\s+through\s+traffic\b",
        r"\bstands?\s+under\b",
        r"\blies?\s+down\b",
        r"\blying\s+down\b",
        r"\brolls?\s+over\b",
    ]
    return any(re.search(pattern, lowered) for pattern in transition_patterns)


def _normalize_route_metadata(
    *,
    scene_index: int,
    route_change_level: str,
    continuity_route_hint: str,
    route_factors: dict[str, Any],
    route_reason: str,
    level_adjustment_reason: str | None,
    primary_visible_character_ids: list[str],
    previous_visible_character_ids: list[str],
    transition_text: str,
) -> tuple[str, str, str, str | None, str | None]:
    adjusted_level = route_change_level
    adjusted_hint = continuity_route_hint
    adjusted_reason = route_reason
    route_hint_adjustment_reason = None

    if scene_index == 0:
        adjusted_level = "large"
        adjusted_hint = "text2img"
        adjusted_reason = "Initial scene setup"
        route_hint_adjustment_reason = "initial_scene_setup"
        if route_change_level != adjusted_level and level_adjustment_reason is None:
            level_adjustment_reason = "initial_scene_setup"
        return adjusted_level, adjusted_hint, adjusted_reason, level_adjustment_reason, route_hint_adjustment_reason

    previous_set = {character_id.lower() for character_id in previous_visible_character_ids}
    current_set = {character_id.lower() for character_id in primary_visible_character_ids}
    if current_set != previous_set:
        route_hint_adjustment_reason = "visible_character_change"
    elif route_factors.get("composition_change_needed"):
        route_hint_adjustment_reason = "composition_change_needed"
    elif adjusted_level == "large":
        route_hint_adjustment_reason = "large_route_change"
    elif (
        route_factors.get("body_state_change")
        and route_factors.get("primary_action_change")
        and _has_body_action_transition(transition_text)
    ):
        route_hint_adjustment_reason = "body_action_transition"

    if route_hint_adjustment_reason:
        adjusted_hint = "text2img"
    return adjusted_level, adjusted_hint, adjusted_reason, level_adjustment_reason, route_hint_adjustment_reason


def _character_id_lookup(characters: list[dict[str, Any]]) -> dict[str, str]:
    return {character["character_id"].lower(): character["character_id"] for character in characters if character.get("character_id")}


def _normalize_optional_character_id(value: Any, valid_character_ids: dict[str, str], *, scene_id: str) -> str | None:
    character_id = _normalize_text(value)
    if not character_id:
        return None
    key = character_id.lower()
    if key not in valid_character_ids:
        raise LLMPromptError(f"Unknown identity_conditioning_subject_id for {scene_id}: {character_id}")
    return valid_character_ids[key]


def _normalize_character_id_list(value: Any, valid_character_ids: dict[str, str], *, field_name: str, scene_id: str) -> list[str]:
    normalized = []
    seen = set()
    for character_id in _normalize_list(value):
        key = character_id.lower()
        if key not in valid_character_ids:
            raise LLMPromptError(f"Unknown {field_name} for {scene_id}: {character_id}")
        canonical = valid_character_ids[key]
        canonical_key = canonical.lower()
        if canonical_key not in seen:
            seen.add(canonical_key)
            normalized.append(canonical)
    return normalized


def _derive_scene_policy(primary_visible_character_ids: list[str]) -> dict[str, Any]:
    visible_character_count = len(primary_visible_character_ids)
    if visible_character_count == 1:
        scene_focus_mode = "single_primary"
    elif visible_character_count == 2:
        scene_focus_mode = "dual_primary"
    else:
        scene_focus_mode = "other"
    return {
        "visible_character_count": visible_character_count,
        "scene_focus_mode": scene_focus_mode,
    }


def _contains_any_phrase(text: str, phrases: list[str]) -> str | None:
    normalized = _normalize_text(text).lower()
    for phrase in phrases:
        if phrase in normalized:
            return phrase
    return None


def _fallback_dual_interaction_summary(scene_text: str, primary_action: str, generation_prompt: str) -> str | None:
    normalized_scene = _trim_text(scene_text, max_words=18, max_chars=160)
    if normalized_scene and len(normalized_scene.split()) > 3:
        return normalized_scene
    normalized_action = _normalize_text(primary_action)
    if normalized_action:
        return _trim_text(f"both characters {normalized_action} together", max_words=18, max_chars=160)
    normalized_generation = _trim_text(generation_prompt, max_words=18, max_chars=160)
    if normalized_generation and " and " in normalized_generation.lower() and len(normalized_generation.split()) <= 3:
        normalized_generation = ""
    if normalized_generation and len(normalized_generation.split()) > 2:
        return normalized_generation
    return None


def _fallback_dual_spatial_relation(scene_text: str, generation_prompt: str) -> str | None:
    combined_text = _normalize_text(" ".join([scene_text, generation_prompt]))
    lowered = combined_text.lower()
    if "look at each other" in lowered or "looking at each other" in lowered:
        return "facing each other"
    if "meets" in lowered and "crowd" in lowered:
        return "meeting amid a crowd"
    if "cafe" in lowered:
        if "talk" in lowered or "continue" in lowered:
            return "sitting together at a cafe table"
        return "at a cafe table"
    if "exhibition" in lowered:
        return "standing side by side facing an exhibition wall"
    if "park" in lowered and "talk" in lowered:
        if "bench" in lowered:
            return "sitting side by side on a park bench"
        return "sitting together in a park"
    if "on the left" in lowered and "on the right" in lowered:
        return "one character on the left, the other on the right"
    if "facing each other" in lowered:
        if "across a table" in lowered:
            return "facing each other across a table"
        return "facing each other"
    if "side by side" in lowered:
        return "standing side by side" if "standing" in lowered else "sitting side by side" if "sitting" in lowered else "side by side"
    if "across a table" in lowered:
        return "across a table"
    if "next to each other" in lowered:
        return "next to each other"
    return None


def _fallback_dual_framing(scene_text: str, generation_prompt: str) -> str | None:
    combined_text = _normalize_text(" ".join([scene_text, generation_prompt])).lower()
    if "medium two-shot" in combined_text or "medium two shot" in combined_text:
        return "medium two-shot"
    if "wide shot" in combined_text:
        if "full bodies" in combined_text or "full body" in combined_text:
            return "wide shot with both full bodies visible"
        return "wide shot"
    if "close-up" in combined_text or "close up" in combined_text:
        return "close-up two-shot"
    if "full bodies" in combined_text or "full body" in combined_text:
        return "wide shot with both full bodies visible"
    if "portrait" in combined_text:
        return "portrait-style two-shot"
    if "two-shot" in combined_text or "two shot" in combined_text:
        return "two-shot"
    return None


def _fallback_setting_focus(scene_text: str, generation_prompt: str, secondary_elements: list[str]) -> str | None:
    for element in secondary_elements:
        normalized_element = _normalize_text(element)
        if normalized_element:
            return normalized_element

    combined_text = _normalize_text(" ".join([scene_text, generation_prompt])).lower()
    ordered_patterns = [
        ("art exhibition wall", "art exhibition wall"),
        ("exhibition wall", "exhibition wall"),
        ("park bench", "park bench"),
        ("cafe table", "cafe table"),
        ("kitchen window", "kitchen window"),
        ("window", "window"),
        ("kitchen", "kitchen"),
        ("exhibition", "exhibition"),
        ("cafe", "cafe table"),
        ("bench", "bench"),
        ("table", "table"),
        ("park", "park"),
        ("wall", "wall"),
    ]
    for pattern, value in ordered_patterns:
        if pattern in combined_text:
            return value
    return None


class LLMAssistedPromptBuilder:
    def __init__(
        self,
        prompt_config: dict[str, Any],
        *,
        llm_client: BaseLLMClient | None = None,
        event_logger: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.prompt_config = prompt_config
        self.llm_config = prompt_config.get("llm", {})
        self.cache_config = prompt_config.get("cache", {})
        self.artifact_config = prompt_config.get("artifact", {})
        self.rule_based_builder = PromptBuilder(prompt_config)
        self.llm_client = llm_client
        self.event_logger = event_logger
        self.last_scene_plans: dict[str, dict[str, Any]] = {}
        self.last_route_hints: dict[str, dict[str, Any]] = {}
        self.last_character_specs: dict[str, dict[str, Any]] = {}

    def build_story_prompts(self, story: Story) -> dict[str, PromptSpec]:
        try:
            structured_output = self._load_or_generate_structured_output(story)
            self.last_character_specs = build_llm_character_specs(structured_output, story)
            self.last_scene_plans = self._build_scene_plans(structured_output)
            self.last_route_hints = self._build_route_hints(self.last_scene_plans)
            return self._build_prompt_specs(story, structured_output)
        except Exception as exc:
            self._log("llm_prompt_validation_failed", error=str(exc))
            if self.llm_config.get("fallback_to_rule_based", True):
                self._log("llm_prompt_fallback_to_rule_based", error=str(exc))
                self.last_scene_plans = {}
                self.last_route_hints = {}
                self.last_character_specs = build_rule_based_character_specs(story)
                return self.rule_based_builder.build_story_prompts(story)
            if isinstance(exc, LLMPromptError):
                raise
            raise LLMPromptError(str(exc)) from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "pipeline": "llm_assisted",
            "implemented": True,
            "provider": self.llm_config.get("provider", "openai"),
            "model": self.llm_config.get("model", "gpt-4o-2024-08-06"),
            "schema_version": self.llm_config.get("schema_version", "v1"),
            "builder_version": self.llm_config.get("builder_version", "llm_assisted_v9"),
            "cache_enabled": bool(self.cache_config.get("enabled", True)),
            "artifact_path": self.artifact_config.get("path"),
            "scene_plans": self.last_scene_plans,
            "scene_route_hints": self.last_route_hints,
            "character_specs": self.last_character_specs,
        }

    def _load_or_generate_structured_output(self, story: Story) -> dict[str, Any]:
        artifact_path = self.artifact_config.get("path")
        if artifact_path:
            try:
                structured = self._load_artifact(Path(artifact_path))
                self._log("llm_prompt_artifact_loaded", path=str(artifact_path))
                return structured
            except Exception as exc:
                self._log("llm_prompt_artifact_invalid", path=str(artifact_path), error=str(exc))
                raise

        cache_key = build_prompt_cache_key(story, self.prompt_config)
        cache_enabled = bool(self.cache_config.get("enabled", True))
        cache = PromptCache(self.cache_config.get("cache_dir", ".cache/prompt_builder"))
        if cache_enabled:
            try:
                cached = cache.load(cache_key)
            except Exception as exc:
                self._log("llm_prompt_cache_invalid", cache_key=cache_key, error=str(exc))
                cached = None
            if cached is not None:
                self._log("llm_prompt_cache_hit", cache_key=cache_key)
                return self._validate_structured_output(story, cached["validated_output"])
            self._log("llm_prompt_cache_miss", cache_key=cache_key)

        response = self._call_llm(story)
        structured_output = self._validate_structured_output(story, response.parsed_json)
        request_metadata = self._request_metadata(cache_key)
        record = build_cache_record(
            cache_key=cache_key,
            request_metadata=request_metadata,
            raw_response=response.raw_text,
            parsed_response=response.parsed_json,
            validated_output=structured_output,
        )
        if cache_enabled:
            cache.save(cache_key, record)
        self._export_artifact_if_enabled(story, cache_key, record)
        return structured_output

    def _call_llm(self, story: Story):
        client = self.llm_client or build_llm_client(self.llm_config)
        self._log(
            "llm_prompt_api_call_started",
            provider=self.llm_config.get("provider", "openai"),
            model=self.llm_config.get("model", "gpt-4o-2024-08-06"),
        )
        response = client.generate_structured(
            messages=self._build_messages(story),
            json_schema=self._json_schema(),
        )
        self._log("llm_prompt_api_call_completed", metadata=response.metadata)
        return response

    def _build_messages(self, story: Story) -> list[dict[str, str]]:
        scene_lines = "\n".join(f"- {scene.scene_id}: {scene.clean_text}" for scene in story.scenes)
        system_prompt = (
            "You generate short, structured prompt planning JSON for a story image pipeline. "
            "Return only fields matching the schema. Keep prompts concise and literal. "
            "Do not invent long artistic prose. Do not directly generate the final PromptSpec. "
            "The local pipeline will assemble style_prompt, negative_prompt, character_prompt, "
            "global_context_prompt, local_prompt, and full_prompt."
        )
        user_prompt = (
            "Extract shared identity, setting, and short scene prompts from this story.\n"
            "Rules:\n"
            "- global.main_character must be exactly one character_id from global.characters. If there is no single main character, use an empty string.\n"
            "- Do not use descriptors such as human man, human woman, person, or group as global.main_character.\n"
            "- Use exact tagged entity names as character_id values when the story provides tags; only create temporary ids when no usable tag exists.\n"
            "- identity_cues must be visual and reusable across panels, not personality traits.\n"
            "- global.characters must contain stable visual identity blocks for recurring visual subjects, including animals and non-human subjects.\n"
            "- character specs may include age_band, gender_presentation, hair_color, hairstyle, skin_tone, body_build, signature_outfit, signature_accessory, and profession_marker.\n"
            "- character specs must not include scene-specific action, pose, emotion, temporary props, object state, lighting, or camera framing.\n"
            "- Leave uncertain character spec fields empty instead of inventing details.\n"
            "- If the main character is human, state that clearly in identity_cues using terms like human woman, human girl, human man, or human person.\n"
            "- If the main subject is explicitly an animal species such as bird, dog, cat, puppy, pet, or animal, state the species or animal identity clearly in identity_cues and do not describe it as human.\n"
            "- Do not infer animal identity only from actions such as sitting, lying down, rolling over, resting, laughing, or playing with toys. A tagged name with he/she pronouns should default to human unless the story explicitly says it is an animal species.\n"
            "- Never substitute an animal or pet for a human character.\n"
            "- generation_prompt must be a short image description phrase, not a command; do not start with "
            "Illustrate, Show, Depict, Create, Generate, Draw, Paint, or Render.\n"
            "- scoring_prompt must be a short descriptive phrase for CLIP-style matching, not a question.\n"
            "- action_prompt must be action-only and concise, not a sentence that starts with Show or Depict.\n"
            "- style_cues should only include extra visual style details not already present in the local style prompt.\n"
            "- continuity_subject_ids must list recurring visual subjects that should stay consistent with the previous panel.\n"
            "- continuity_route_hint must be text2img or img2img. It is only a hint; the local router makes the final decision.\n"
            "- route_change_level must be small, medium, or large compared with the previous scene.\n"
            "- Use small only when all are true: same subject, same setting, very similar camera framing/composition, same body state, and only local gaze, expression, hand, or small-object changes.\n"
            "- Use medium when the same subject continues but action, body pose/state, subject-object relationship, key objects, or readable composition changes noticeably.\n"
            "- Use large when location, time, story beat, layout, camera framing, or required pose/props change substantially, or when the previous image would prevent the current scene from being shown correctly.\n"
            "- Same character plus same setting is not enough for small. If unsure between small and medium, choose medium. If unsure between medium and large, choose large when local edits would not make the current action readable.\n"
            "- route_factors must explain the route judgment with booleans for same_subject, same_setting, body_state_change, primary_action_change, composition_change_needed, and a short new_key_objects list.\n"
            "- route_reason must be a short explanation of the scene-to-previous-scene routing judgment.\n"
            "- primary_visible_character_ids must list the main visible recurring characters in the scene using global.characters character_id values.\n"
            "- Leave primary_visible_character_ids empty for environment-only or object-only scenes with no foreground recurring character.\n"
            "- identity_conditioning_subject_id chooses the single character whose identity anchor should condition this scene. Use a character_id from global.characters only when one character is the clear identity target.\n"
            "- Set identity_conditioning_subject_id to null when primary_visible_character_ids is empty.\n"
            "- If multiple characters are equally important, or the scene is ambiguous, set identity_conditioning_subject_id to null instead of guessing.\n"
            "- Anonymous secondary people such as a friend, passerby, or crowd member should not become continuity subjects unless the story later makes them recurring.\n"
            "- Anonymous secondary people can be mentioned naturally in generation_prompt or interaction_summary, but should not appear as character specs like 'adult unknown'.\n"
            "- For pronoun-only scenes, resolve the visible characters from story context when possible. If a plural pronoun such as 'They' refers to multiple characters equally, leave identity_conditioning_subject_id null.\n"
            "- Use interaction_summary for how visible characters relate or act together in the scene.\n"
            "- For dual_primary scenes, interaction_summary must describe what the two characters are doing together, not just that both are present.\n"
            "- Use spatial_relation for where the visible characters are positioned relative to each other.\n"
            "- For dual_primary scenes, spatial_relation should be concrete when inferable, such as left/right placement, side by side, or facing each other across a table.\n"
            "- Use framing for composition guidance such as close-up, medium shot, wide shot, or clear two-person composition.\n"
            "- For dual_primary scenes, framing should be a concrete camera/composition phrase such as medium two-shot, wide shot with both full bodies visible, or clear two-person composition.\n"
            "- setting_focus is optional and should only capture the local setting emphasis for the panel.\n"
            "- For dual_primary scenes, setting_focus should capture the local visual place or key setting object when present, such as cafe table, art exhibition wall, or park bench.\n"
            "- Do not leave dual_primary scene-planning fields vague if the story text or scene prompt already supports a specific answer.\n"
            "- Do not repeat long identity descriptions inside scene-level prompts. Character identity is handled locally from global.characters.\n"
            f"Style prompt from local config: {self.prompt_config.get('style_prompt', '')}\n"
            f"Scenes:\n{scene_lines}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _json_schema(self) -> dict[str, Any]:
        scene_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "scene_id",
                "primary_action",
                "secondary_elements",
                "generation_prompt",
                "scoring_prompt",
                "action_prompt",
                "continuity_subject_ids",
                "continuity_route_hint",
                "route_change_level",
                "route_factors",
                "route_reason",
                "identity_conditioning_subject_id",
                "primary_visible_character_ids",
                "interaction_summary",
                "spatial_relation",
                "framing",
                "setting_focus",
            ],
            "properties": {
                "scene_id": {"type": "string"},
                "primary_action": {"type": "string"},
                "secondary_elements": {"type": "array", "items": {"type": "string"}},
                "generation_prompt": {"type": "string"},
                "scoring_prompt": {"type": "string"},
                "action_prompt": {"type": "string"},
                "continuity_subject_ids": {"type": "array", "items": {"type": "string"}},
                "continuity_route_hint": {"type": "string", "enum": ["text2img", "img2img"]},
                "route_change_level": {"type": "string", "enum": ["small", "medium", "large"]},
                "route_factors": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "same_subject",
                        "same_setting",
                        "body_state_change",
                        "primary_action_change",
                        "new_key_objects",
                        "composition_change_needed",
                    ],
                    "properties": {
                        "same_subject": {"type": "boolean"},
                        "same_setting": {"type": "boolean"},
                        "body_state_change": {"type": "boolean"},
                        "primary_action_change": {"type": "boolean"},
                        "new_key_objects": {"type": "array", "items": {"type": "string"}},
                        "composition_change_needed": {"type": "boolean"},
                    },
                },
                "route_reason": {"type": "string"},
                "identity_conditioning_subject_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "primary_visible_character_ids": {"type": "array", "items": {"type": "string"}},
                "interaction_summary": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "spatial_relation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "framing": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "setting_focus": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        }
        nullable_short_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        character_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
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
            ],
            "properties": {
                "character_id": {"type": "string"},
                "age_band": nullable_short_string,
                "gender_presentation": nullable_short_string,
                "hair_color": nullable_short_string,
                "hairstyle": nullable_short_string,
                "skin_tone": nullable_short_string,
                "body_build": nullable_short_string,
                "signature_outfit": nullable_short_string,
                "signature_accessory": nullable_short_string,
                "profession_marker": nullable_short_string,
            },
        }
        return {
            "name": "story_prompt_plan",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["global", "scenes"],
                "properties": {
                    "global": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["main_character", "identity_cues", "shared_setting", "style_cues", "characters"],
                        "properties": {
                            "main_character": {"type": "string"},
                            "identity_cues": {"type": "array", "items": {"type": "string"}},
                            "shared_setting": {"type": "array", "items": {"type": "string"}},
                            "style_cues": {"type": "array", "items": {"type": "string"}},
                            "characters": {"type": "array", "items": character_schema},
                        },
                    },
                    "scenes": {"type": "array", "items": scene_schema},
                },
            },
        }

    def _validate_structured_output(self, story: Story, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMPromptError("LLM prompt output must be a JSON object")
        global_payload = payload.get("global")
        scenes_payload = payload.get("scenes")
        if not isinstance(global_payload, dict) or not isinstance(scenes_payload, list):
            raise LLMPromptError("LLM prompt output is missing global/scenes")
        characters_payload = global_payload.get("characters")
        if not isinstance(characters_payload, list) or not characters_payload:
            raise LLMPromptError("LLM prompt output is missing global.characters")
        if len(scenes_payload) != len(story.scenes):
            raise LLMPromptError("LLM scene count does not match parsed story")

        expected_ids = [scene.scene_id for scene in story.scenes]
        actual_ids = [scene.get("scene_id") if isinstance(scene, dict) else None for scene in scenes_payload]
        if actual_ids != expected_ids:
            raise LLMPromptError(f"LLM scene ids do not match parsed story: expected {expected_ids}, got {actual_ids}")

        scene_text_by_id = {scene.scene_id: _normalize_text(scene.clean_text) for scene in story.scenes}
        normalized_characters = self._normalize_character_specs(global_payload.get("characters"))
        normalized_characters = _sanitize_character_specs_for_story(story, normalized_characters)
        valid_character_ids = _character_id_lookup(normalized_characters)
        character_specs_by_id = {
            character["character_id"]: character for character in normalized_characters if character.get("character_id")
        }
        anonymous_ids = _anonymous_character_ids(character_specs_by_id)
        normalized_scenes = []
        previous_visible_character_ids: list[str] = []
        for scene_index, scene_payload in enumerate(scenes_payload):
            if not isinstance(scene_payload, dict):
                raise LLMPromptError("Each LLM scene entry must be an object")
            generation_prompt = _trim_text(
                _strip_leading_prompt_verb(scene_payload.get("generation_prompt")),
                max_words=int(self.prompt_config.get("generation_max_words", 28)),
                max_chars=int(self.prompt_config.get("generation_max_chars", 220)),
            )
            scoring_prompt = _trim_text(
                _strip_question_frame(scene_payload.get("scoring_prompt")),
                max_words=int(self.prompt_config.get("scoring_max_words", 20)),
                max_chars=int(self.prompt_config.get("scoring_max_chars", 160)),
            )
            action_prompt = _trim_text(
                _strip_leading_prompt_verb(scene_payload.get("action_prompt")),
                max_words=int(self.prompt_config.get("scoring_max_words", 20)),
                max_chars=int(self.prompt_config.get("scoring_max_chars", 160)),
            )
            if not generation_prompt or not scoring_prompt or not action_prompt:
                raise LLMPromptError(f"LLM prompt fields are missing or empty for {scene_payload.get('scene_id')}")
            continuity_route_hint = _normalize_text(scene_payload.get("continuity_route_hint")).lower()
            route_change_level = _normalize_text(scene_payload.get("route_change_level")).lower()
            route_reason = _trim_text(
                scene_payload.get("route_reason"),
                max_words=24,
                max_chars=180,
            )
            if continuity_route_hint not in {"text2img", "img2img"}:
                raise LLMPromptError(f"Invalid continuity_route_hint for {scene_payload.get('scene_id')}: {continuity_route_hint}")
            if route_change_level not in {"small", "medium", "large"}:
                raise LLMPromptError(f"Invalid route_change_level for {scene_payload.get('scene_id')}: {route_change_level}")
            if not route_reason:
                raise LLMPromptError(f"LLM route_reason is missing or empty for {scene_payload.get('scene_id')}")
            route_factors = _normalize_route_factors(scene_payload.get("route_factors"))
            adjusted_route_change_level, adjustment_reason = _adjust_route_change_level(route_change_level, route_factors)
            if "identity_conditioning_subject_id" not in scene_payload:
                raise LLMPromptError(f"LLM identity_conditioning_subject_id is missing for {scene_payload.get('scene_id')}")
            identity_conditioning_subject_id = _normalize_optional_character_id(
                scene_payload.get("identity_conditioning_subject_id"),
                valid_character_ids,
                scene_id=scene_payload["scene_id"],
            )
            primary_visible_character_ids = _normalize_character_id_list(
                scene_payload.get("primary_visible_character_ids"),
                valid_character_ids,
                field_name="primary_visible_character_ids",
                scene_id=scene_payload["scene_id"],
            )
            anonymous_visible_character_ids = [
                character_id
                for character_id in primary_visible_character_ids
                if _is_anonymous_character_id(character_id, anonymous_ids)
            ]
            primary_visible_character_ids = [
                character_id
                for character_id in primary_visible_character_ids
                if not _is_anonymous_character_id(character_id, anonymous_ids)
            ]
            scene_policy = _derive_scene_policy(primary_visible_character_ids)
            if anonymous_visible_character_ids:
                scene_policy["anonymous_visible_subject_ids"] = anonymous_visible_character_ids
                scene_policy["anonymous_visible_subject_count"] = len(anonymous_visible_character_ids)
            scene_focus_mode = scene_policy["scene_focus_mode"]
            if scene_policy["visible_character_count"] == 0:
                identity_conditioning_subject_id = None
            elif (
                scene_policy["visible_character_count"] == 1
                and anonymous_visible_character_ids
                and identity_conditioning_subject_id is None
            ):
                identity_conditioning_subject_id = primary_visible_character_ids[0]
            elif identity_conditioning_subject_id and _is_anonymous_character_id(identity_conditioning_subject_id, anonymous_ids):
                identity_conditioning_subject_id = primary_visible_character_ids[0] if scene_policy["visible_character_count"] == 1 else None
            scene_text = scene_text_by_id.get(scene_payload["scene_id"], "")
            interaction_summary = _trim_text(
                scene_payload.get("interaction_summary"),
                max_words=18,
                max_chars=160,
            )
            spatial_relation = _trim_text(
                scene_payload.get("spatial_relation"),
                max_words=16,
                max_chars=140,
            )
            framing = _trim_text(
                scene_payload.get("framing"),
                max_words=16,
                max_chars=140,
            )
            setting_focus = _trim_text(
                scene_payload.get("setting_focus"),
                max_words=14,
                max_chars=120,
            )
            used_default_interaction_summary = False
            used_default_spatial_relation = False
            used_default_framing = False
            used_default_setting_focus = False
            if scene_focus_mode == "dual_primary":
                interaction_summary = interaction_summary or _fallback_dual_interaction_summary(
                    scene_text,
                    _normalize_text(scene_payload.get("primary_action")),
                    generation_prompt,
                )
                if not interaction_summary:
                    interaction_summary = "both characters are present in the same scene"
                    used_default_interaction_summary = True

                spatial_relation = spatial_relation or _fallback_dual_spatial_relation(scene_text, generation_prompt)
                if not spatial_relation:
                    ordered_visible_ids = [character_id.lower() for character_id in primary_visible_character_ids[:2]]
                    if len(ordered_visible_ids) == 2:
                        spatial_relation = f"{ordered_visible_ids[0]} on the left, {ordered_visible_ids[1]} on the right"
                    else:
                        spatial_relation = "both characters remain clearly separated"
                    used_default_spatial_relation = True

                framing = framing or _fallback_dual_framing(scene_text, generation_prompt)
                if not framing:
                    framing = "medium two-shot, both characters visible"
                    used_default_framing = True

                setting_focus = setting_focus or _fallback_setting_focus(
                    scene_text,
                    generation_prompt,
                    _normalize_list(scene_payload.get("secondary_elements")),
                )
                if not setting_focus:
                    setting_focus = None
                    used_default_setting_focus = True
                if identity_conditioning_subject_id is not None:
                    raise LLMPromptError(
                        f"LLM identity_conditioning_subject_id must be null for dual_primary scene {scene_payload['scene_id']}"
                    )
            if scene_policy["visible_character_count"] == 0:
                continuity_subject_ids = []
            else:
                continuity_subject_ids = [
                    subject_id
                    for subject_id in _normalize_list(scene_payload.get("continuity_subject_ids"))
                    if not _is_anonymous_character_id(subject_id, anonymous_ids)
                ]
            (
                adjusted_route_change_level,
                continuity_route_hint,
                route_reason,
                adjustment_reason,
                route_hint_adjustment_reason,
            ) = _normalize_route_metadata(
                scene_index=scene_index,
                route_change_level=adjusted_route_change_level,
                continuity_route_hint=continuity_route_hint,
                route_factors=route_factors,
                route_reason=route_reason,
                level_adjustment_reason=adjustment_reason,
                primary_visible_character_ids=primary_visible_character_ids,
                previous_visible_character_ids=previous_visible_character_ids,
                transition_text=" ".join(
                    [
                        scene_text,
                        _normalize_text(scene_payload.get("primary_action")),
                        generation_prompt,
                    ]
                ),
            )
            normalized_scenes.append(
                {
                    "scene_id": scene_payload["scene_id"],
                    "primary_action": _normalize_text(scene_payload.get("primary_action")),
                    "secondary_elements": _normalize_list(scene_payload.get("secondary_elements")),
                    "generation_prompt": generation_prompt,
                    "scoring_prompt": scoring_prompt,
                    "action_prompt": action_prompt,
                    "continuity_subject_ids": continuity_subject_ids,
                    "continuity_route_hint": continuity_route_hint,
                    "llm_route_change_level": route_change_level,
                    "route_change_level": adjusted_route_change_level,
                    "route_level_adjustment_reason": adjustment_reason,
                    "route_hint_adjustment_reason": route_hint_adjustment_reason,
                    "route_factors": route_factors,
                    "route_reason": route_reason,
                    "identity_conditioning_subject_id": identity_conditioning_subject_id,
                    "primary_visible_character_ids": primary_visible_character_ids,
                    "interaction_summary": interaction_summary,
                    "spatial_relation": spatial_relation,
                    "framing": framing,
                    "setting_focus": setting_focus or None,
                    "used_default_interaction_summary": used_default_interaction_summary,
                    "used_default_spatial_relation": used_default_spatial_relation,
                    "used_default_framing": used_default_framing,
                    "used_default_setting_focus": used_default_setting_focus,
                    "policy": scene_policy,
                }
            )
            previous_visible_character_ids = primary_visible_character_ids

        normalized_global = {
            "main_character": _normalize_text(global_payload.get("main_character")),
            "identity_cues": _normalize_list(global_payload.get("identity_cues")),
            "shared_setting": _normalize_list(global_payload.get("shared_setting")),
            "style_cues": _normalize_list(global_payload.get("style_cues")),
            "characters": normalized_characters,
        }
        main_character = normalized_global["main_character"].lower()
        character_ids = {character["character_id"].lower() for character in normalized_global["characters"]}
        if main_character and main_character not in character_ids:
            raise LLMPromptError(f"LLM character specs do not include main_character: {normalized_global['main_character']}")

        return {
            "global": normalized_global,
            "scenes": normalized_scenes,
        }

    def _normalize_character_specs(self, characters_payload: Any) -> list[dict[str, Any]]:
        if not isinstance(characters_payload, list) or not characters_payload:
            raise LLMPromptError("LLM global.characters must be a non-empty list")
        normalized_characters = []
        seen = set()
        for character_payload in characters_payload:
            if not isinstance(character_payload, dict):
                raise LLMPromptError("Each LLM character spec must be an object")
            character_id = _normalize_text(character_payload.get("character_id"))
            if not character_id:
                raise LLMPromptError("LLM character spec is missing character_id")
            key = character_id.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized_characters.append(
                {
                    "character_id": character_id,
                    "age_band": _normalize_text(character_payload.get("age_band")),
                    "gender_presentation": _normalize_text(character_payload.get("gender_presentation")),
                    "hair_color": _normalize_text(character_payload.get("hair_color")),
                    "hairstyle": _normalize_text(character_payload.get("hairstyle")),
                    "skin_tone": _normalize_text(character_payload.get("skin_tone")),
                    "body_build": _normalize_text(character_payload.get("body_build")),
                    "signature_outfit": _normalize_text(character_payload.get("signature_outfit")),
                    "signature_accessory": _normalize_text(character_payload.get("signature_accessory")),
                    "profession_marker": _normalize_text(character_payload.get("profession_marker")),
                }
            )
        return normalized_characters

    def _build_scene_plans(self, structured_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
        scene_plans = {}
        for scene_payload in structured_output.get("scenes", []):
            scene_id = scene_payload.get("scene_id")
            if not scene_id:
                continue
            scene_plans[scene_id] = {
                "identity_conditioning_subject_id": scene_payload.get("identity_conditioning_subject_id"),
                "primary_visible_character_ids": list(scene_payload.get("primary_visible_character_ids", [])),
                "continuity_subject_ids": list(scene_payload.get("continuity_subject_ids", [])),
                "continuity_route_hint": scene_payload.get("continuity_route_hint"),
                "llm_route_change_level": scene_payload.get("llm_route_change_level"),
                "route_change_level": scene_payload.get("route_change_level"),
                "route_level_adjustment_reason": scene_payload.get("route_level_adjustment_reason"),
                "route_hint_adjustment_reason": scene_payload.get("route_hint_adjustment_reason"),
                "route_factors": dict(scene_payload.get("route_factors", {})),
                "route_reason": scene_payload.get("route_reason"),
                "interaction_summary": scene_payload.get("interaction_summary"),
                "spatial_relation": scene_payload.get("spatial_relation"),
                "framing": scene_payload.get("framing"),
                "setting_focus": scene_payload.get("setting_focus"),
                "used_default_interaction_summary": bool(scene_payload.get("used_default_interaction_summary", False)),
                "used_default_spatial_relation": bool(scene_payload.get("used_default_spatial_relation", False)),
                "used_default_framing": bool(scene_payload.get("used_default_framing", False)),
                "used_default_setting_focus": bool(scene_payload.get("used_default_setting_focus", False)),
                "policy": dict(scene_payload.get("policy", {})),
            }
        return scene_plans

    def _build_route_hints(self, scene_plans: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        hints = {}
        for scene_id, scene_plan in scene_plans.items():
            hints[scene_id] = {
                "identity_conditioning_subject_id": scene_plan.get("identity_conditioning_subject_id"),
                "primary_visible_character_ids": list(scene_plan.get("primary_visible_character_ids", [])),
                "continuity_subject_ids": list(scene_plan.get("continuity_subject_ids", [])),
                "continuity_route_hint": scene_plan.get("continuity_route_hint"),
                "llm_route_change_level": scene_plan.get("llm_route_change_level"),
                "route_change_level": scene_plan.get("route_change_level"),
                "route_level_adjustment_reason": scene_plan.get("route_level_adjustment_reason"),
                "route_hint_adjustment_reason": scene_plan.get("route_hint_adjustment_reason"),
                "route_factors": dict(scene_plan.get("route_factors", {})),
                "route_reason": scene_plan.get("route_reason"),
            }
        return hints

    def _build_prompt_specs(self, story: Story, structured_output: dict[str, Any]) -> dict[str, PromptSpec]:
        global_payload = structured_output["global"]
        scene_payloads = {scene["scene_id"]: scene for scene in structured_output["scenes"]}
        style_prompt = _normalize_text(self.prompt_config.get("style_prompt", ""))
        negative_prompt = _normalize_text(self.prompt_config.get("negative_prompt", ""))
        main_character = global_payload.get("main_character", "")
        identity_cues = global_payload.get("identity_cues", [])
        shared_setting = global_payload.get("shared_setting", [])
        style_cues = _filter_repeated_style_cues(global_payload.get("style_cues", []), style_prompt)
        scene_continuity = _normalize_text(self.prompt_config.get("scene_continuity_prompt", ""))
        global_context_prompt = _join_parts([*shared_setting, *style_cues, scene_continuity])
        character_specs_by_id = {
            character["character_id"]: character for character in global_payload.get("characters", []) if character.get("character_id")
        }
        anonymous_ids = _anonymous_character_ids(character_specs_by_id)
        explicit_animal_ids = _explicit_animal_subject_ids(story, character_specs_by_id)
        fallback_character_prompt = _ensure_human_identity(_join_parts([main_character, *identity_cues]))

        prompt_specs = {}
        for scene in story.scenes:
            scene_payload = scene_payloads[scene.scene_id]
            visible_character_ids = list(scene_payload.get("primary_visible_character_ids", []))
            scene_policy = scene_payload.get("policy", {})
            scene_focus_mode = scene_policy.get("scene_focus_mode")
            visible_character_count = int(scene_policy.get("visible_character_count", len(visible_character_ids)) or 0)
            generation_max_words = int(self.prompt_config.get("generation_max_words", 28))
            generation_max_chars = int(self.prompt_config.get("generation_max_chars", 220))
            if visible_character_count == 0:
                character_prompt = ""
            elif scene_focus_mode == "dual_primary":
                identity_snippets = [
                    build_character_identity_snippet(character_specs_by_id[character_id])
                    for character_id in visible_character_ids
                    if character_id in character_specs_by_id
                    and not _is_anonymous_character_id(character_id, anonymous_ids)
                ]
                if not identity_snippets and main_character in character_specs_by_id:
                    identity_snippets = [build_character_identity_snippet(character_specs_by_id[main_character])]
                character_prompt = _join_parts(identity_snippets) or fallback_character_prompt
            elif scene_focus_mode == "single_primary" and scene_payload.get("identity_conditioning_subject_id"):
                identity_subject_id = scene_payload.get("identity_conditioning_subject_id", "")
                character_prompt = build_lightweight_character_prompt(
                    identity_subject_id,
                    character_specs_by_id,
                    identity_cues,
                    allow_animal_identity=_normalize_text(identity_subject_id).lower() in explicit_animal_ids,
                    human_label_override=_human_pronoun_label_for_story(story, identity_subject_id),
                )
            else:
                character_prompt = fallback_character_prompt
                if scene_focus_mode == "single_primary":
                    character_prompt = _ensure_human_identity(character_prompt)

            allow_scene_animal_identity = visible_character_count == 0 or any(
                _normalize_text(character_id).lower() in explicit_animal_ids
                for character_id in visible_character_ids
            )
            resolved_generation_phrase = _prepare_scene_generation_phrase(
                scene_payload["generation_prompt"],
                character_prompt=character_prompt,
                allow_animal_identity=allow_scene_animal_identity,
                strip_subject=False,
            )
            scene_content_parts = [
                resolved_generation_phrase,
                *scene_payload.get("secondary_elements", []),
                scene_payload.get("interaction_summary", ""),
                scene_payload.get("spatial_relation", ""),
                scene_payload.get("framing", ""),
                scene_payload.get("setting_focus", ""),
            ]
            if scene_focus_mode == "dual_primary":
                scene_content_parts.extend(
                    [
                        scene_payload.get("interaction_summary", ""),
                        scene_payload.get("spatial_relation", ""),
                        scene_payload.get("framing", ""),
                        scene_payload.get("setting_focus", ""),
                    ]
                )
            scene_content_prompt = _join_parts(scene_content_parts)
            resolved_scene_text = _resolved_scene_text(
                scene.clean_text,
                resolved_generation_phrase,
                scene_payload.get("primary_action", ""),
                visible_character_ids,
            )
            local_prompt = _join_parts(
                [
                    resolved_scene_text,
                    scene_payload.get("primary_action", ""),
                    *scene_payload.get("secondary_elements", []),
                    scene_payload.get("interaction_summary", ""),
                    scene_payload.get("spatial_relation", ""),
                    scene_payload.get("framing", ""),
                    scene_payload.get("setting_focus", ""),
                ]
            )
            full_prompt = _join_parts([style_prompt, character_prompt, global_context_prompt, local_prompt])
            if scene_focus_mode == "dual_primary":
                dual_primary_max_words = int(
                    self.prompt_config.get(
                        "dual_primary_generation_max_words",
                        max(generation_max_words, 40),
                    )
                )
                dual_primary_max_chars = int(
                    self.prompt_config.get(
                        "dual_primary_generation_max_chars",
                        max(generation_max_chars, 320),
                    )
                )
                generation_prompt = build_dual_primary_generation_prompt(
                    visible_character_ids,
                    character_specs_by_id,
                    scene_payload.get("interaction_summary", ""),
                    scene_payload.get("spatial_relation", ""),
                    scene_payload.get("framing", ""),
                    scene_payload.get("setting_focus"),
                    fallback_character_prompt=fallback_character_prompt,
                    max_words=dual_primary_max_words,
                    max_chars=dual_primary_max_chars,
                )
            elif visible_character_count == 0:
                generation_prompt = _trim_text(
                    scene_content_prompt,
                    max_words=generation_max_words,
                    max_chars=generation_max_chars,
                )
            else:
                generation_prompt = _merge_identity_into_generation_prompt(
                    character_prompt,
                    scene_content_prompt,
                    max_words=generation_max_words,
                    max_chars=generation_max_chars,
                    allow_animal_identity=allow_scene_animal_identity,
                )
            negative_prompt_for_scene = _merge_human_negative_prompt(character_prompt, negative_prompt)
            prompt_specs[scene.scene_id] = PromptSpec(
                scene_id=scene.scene_id,
                style_prompt=style_prompt,
                character_prompt=character_prompt,
                global_context_prompt=global_context_prompt,
                local_prompt=local_prompt,
                action_prompt=scene_payload["action_prompt"],
                generation_prompt=generation_prompt,
                scoring_prompt=scene_payload["scoring_prompt"],
                full_prompt=full_prompt,
                negative_prompt=negative_prompt_for_scene,
            )
        return prompt_specs

    def _load_artifact(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "validated_output" in payload:
            return payload["validated_output"]
        return payload

    def _export_artifact_if_enabled(self, story: Story, cache_key: str, record: dict[str, Any]) -> None:
        if not self.artifact_config.get("export_enabled", False):
            return
        export_dir = Path(self.artifact_config.get("export_dir", "prompt_artifacts/llm_assisted_v9"))
        story_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(story.source_path).stem).strip("_") or "story"
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.llm_config.get("model", "model")).strip("_")
        schema_version = self.llm_config.get("schema_version", "v1")
        output_path = export_dir / f"{story_slug}_{model_slug}_{schema_version}_{cache_key[:8]}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "artifact_type": "llm_assisted_prompt",
            "timestamp": _now(),
            "request_metadata": record["request_metadata"],
            "validated_output": record["validated_output"],
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, ensure_ascii=True, sort_keys=True)
        self._log("llm_prompt_artifact_exported", path=str(output_path))

    def _request_metadata(self, cache_key: str) -> dict[str, Any]:
        return {
            "cache_key": cache_key,
            "provider": self.llm_config.get("provider", "openai"),
            "model": self.llm_config.get("model", "gpt-4o-2024-08-06"),
            "schema_version": self.llm_config.get("schema_version", "v1"),
            "builder_version": self.llm_config.get("builder_version", "llm_assisted_v9"),
        }

    def _log(self, event: str, **metadata: Any) -> None:
        if self.event_logger is not None:
            self.event_logger(event, **metadata)
