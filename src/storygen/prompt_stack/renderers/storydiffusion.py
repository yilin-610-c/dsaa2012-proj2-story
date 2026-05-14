from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any

from storygen.types import PromptSpec, Story


LEADING_PRONOUN_PATTERN = re.compile(r"^(she|he|they|it)\b", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<([^<>]+)>")


@dataclass(slots=True)
class NativeStoryDiffusionPromptRender:
    general_prompt: str
    identity_prompts: list[str]
    scene_prompts: list[str]
    final_prompt_array: list[str]
    save_image_start_index: int
    character_specs: dict[str, Any] = field(default_factory=dict)
    source_fields: list[dict[str, Any]] = field(default_factory=list)
    identity_prompts_per_character: int = 1
    identity_reference_prompts: list[str] = field(default_factory=list)
    story_scene_prompts: list[str] = field(default_factory=list)
    saved_image_prompt_map: dict[str, Any] = field(default_factory=dict)


def _split_comma_clauses(text: str) -> list[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def _filter_comma_clauses(text: str, bans: list[str]) -> str:
    if not text or not bans:
        return text
    bans_l = [b.lower() for b in bans if b]
    kept: list[str] = []
    for clause in _split_comma_clauses(text):
        low = clause.lower()
        if any(b in low for b in bans_l):
            continue
        kept.append(clause)
    return ", ".join(kept)


def _strip_leading_prefix_chunks(text: str, prefixes: list[str]) -> str:
    """Remove leading comma-separated chunks that match any prefix (case-insensitive)."""
    if not text or not prefixes:
        return text
    prefixes_l = [p.strip().lower() for p in prefixes if p and p.strip()]
    remainder = text.strip()
    changed = True
    while changed and remainder:
        changed = False
        clauses = _split_comma_clauses(remainder)
        if not clauses:
            break
        first = clauses[0]
        if first.lower() in prefixes_l:
            clauses = clauses[1:]
            remainder = ", ".join(clauses)
            changed = True
            continue
        for pref in prefixes_l:
            if first.lower().startswith(pref + " ") or first.lower() == pref:
                clauses = clauses[1:]
                remainder = ", ".join(clauses)
                changed = True
                break
    return remainder.strip().strip(",").strip()


def sanitize_prompt_specs_for_storydiffusion(
    specs: dict[str, PromptSpec],
    rules: dict[str, Any],
) -> dict[str, PromptSpec]:
    """
    Drop SDXL-oriented spatial / repetition clauses that StoryDiffusion handles via attention.
    """
    bans = list(rules.get("segment_bans") or [])
    fields = list(rules.get("comma_filter_fields") or [])
    gen_prefixes = list(rules.get("generation_prompt_strip_prefix_chunks") or [])

    out: dict[str, PromptSpec] = {}
    for scene_id, spec in specs.items():
        new_spec = deepcopy(spec)
        for field_name in fields:
            cur = getattr(new_spec, field_name, "") or ""
            if isinstance(cur, str):
                setattr(new_spec, field_name, _filter_comma_clauses(cur, bans))
        gp = new_spec.generation_prompt or ""
        if gen_prefixes:
            new_spec.generation_prompt = _strip_leading_prefix_chunks(gp, gen_prefixes)
        out[scene_id] = new_spec
    return out


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        key = re.sub(r"\s+", " ", value).strip().lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _spec_value(spec: Any, field_name: str) -> str:
    if isinstance(spec, dict):
        return _normalize_whitespace(spec.get(field_name, ""))
    return _normalize_whitespace(getattr(spec, field_name, ""))


def _character_spec_for(character_specs: dict[str, Any], character_id: str) -> Any:
    if character_id in character_specs:
        return character_specs[character_id]
    normalized_id = character_id.strip().lower()
    for key, value in character_specs.items():
        if str(key).strip().lower() == normalized_id:
            return value
    return {}


def _looks_generic(value: str) -> bool:
    return value.strip().lower() in {
        "",
        "unknown",
        "none",
        "person",
        "human",
        "subject",
        "character",
        "man",
        "woman",
        "boy",
        "girl",
    }


ANIMAL_SPECIES_BY_ID = {
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "puppy": "dog",
    "kitten": "cat",
}
ROBOT_IDS = {"robot", "android"}
HUMAN_ROLE_IDS = {
    "artist",
    "baby",
    "boy",
    "chef",
    "child",
    "driver",
    "girl",
    "kid",
    "lady",
    "man",
    "runner",
    "student",
    "traveler",
    "woman",
}


def _subject_type(character_id: str, spec: Any) -> str:
    explicit = _spec_value(spec, "subject_type").lower()
    if explicit in {"human", "animal", "robot", "object", "vehicle"}:
        return explicit
    lowered_id = character_id.lower()
    combined = " ".join(
        [
            lowered_id,
            _spec_value(spec, "species").lower(),
            _spec_value(spec, "material").lower(),
            _spec_value(spec, "shape_features").lower(),
            _spec_value(spec, "gender_presentation").lower(),
        ]
    )
    if lowered_id in ANIMAL_SPECIES_BY_ID or any(term in combined for term in ("cat", "dog", "bird", "puppy", "animal", "fur", "feather")):
        return "animal"
    if lowered_id in ROBOT_IDS or any(term in combined for term in ("robot", "android", "mechanical", "metal")):
        return "robot"
    if lowered_id in HUMAN_ROLE_IDS or any(term in combined for term in ("human", "woman", "girl", "man", "boy", "female", "male")):
        return "human"
    return "human"


def _base_human_descriptor(character_id: str, spec: Any) -> str:
    gender = _spec_value(spec, "gender_presentation").lower()
    age = _spec_value(spec, "age_band").lower()
    marker = _spec_value(spec, "profession_marker").lower()
    name = character_id.lower()
    combined = " ".join([gender, age, marker, name])
    if "girl" in combined or ("female" in combined and "child" in combined):
        return "human girl"
    if "boy" in combined or ("male" in combined and "child" in combined):
        return "human boy"
    if "female" in combined or "woman" in combined:
        return "human woman"
    if "male" in combined or "man" in combined:
        return "human man"
    return "human person"


def _human_descriptor(character_id: str, spec: Any) -> str:
    parts: list[str] = [_base_human_descriptor(character_id, spec)]
    hair_color = _spec_value(spec, "hair_color")
    hairstyle = _spec_value(spec, "hairstyle")
    if hair_color and hairstyle:
        parts.append(f"{hair_color} {hairstyle}")
    elif hair_color:
        parts.append(f"{hair_color} hair")
    elif hairstyle:
        parts.append(hairstyle)
    for field_name in ("skin_tone", "body_build", "signature_outfit", "signature_accessory", "profession_marker"):
        value = _spec_value(spec, field_name)
        if value and not _looks_generic(value) and value.lower() not in " ".join(parts).lower():
            parts.append(value)
    return ", ".join(_unique(parts))


def _animal_descriptor(character_id: str, spec: Any) -> str:
    species = _spec_value(spec, "species").lower() or ANIMAL_SPECIES_BY_ID.get(character_id.lower(), character_id.lower()) or "animal"
    noun = " ".join(part for part in [_spec_value(spec, "body_size"), _spec_value(spec, "fur_color"), species] if part).strip()
    return ", ".join(_unique([part for part in [noun or species, _spec_value(spec, "fur_pattern"), _spec_value(spec, "markings")] if part])) or species


def _visual_animal_pattern(value: str) -> str:
    text = _normalize_whitespace(value)
    if not text:
        return ""
    lowered = text.lower()
    if lowered == "spotted" or "spotted" in lowered:
        return "visible spotted coat pattern"
    if lowered == "striped" or "stripe" in lowered:
        return "visible striped fur pattern"
    return text


def _character_descriptor_v2(character_id: str, spec: Any) -> str:
    subject_type = _subject_type(character_id, spec)
    if subject_type == "animal":
        species = _spec_value(spec, "species").lower() or ANIMAL_SPECIES_BY_ID.get(character_id.lower(), character_id.lower()) or "animal"
        noun = " ".join(
            part for part in [_spec_value(spec, "body_size"), _spec_value(spec, "fur_color"), species] if part
        ).strip()
        return ", ".join(
            _unique(
                [
                    noun or species,
                    _visual_animal_pattern(_spec_value(spec, "fur_pattern")),
                    _spec_value(spec, "markings"),
                ]
            )
        )
    if subject_type in {"robot", "object", "vehicle"}:
        return _robot_descriptor(character_id, spec)
    parts: list[str] = [_base_human_descriptor(character_id, spec)]
    hair_color = _spec_value(spec, "hair_color")
    hairstyle = _spec_value(spec, "hairstyle")
    if hair_color and hairstyle:
        parts.append(f"{hairstyle} {hair_color} hair" if "hair" not in hairstyle.lower() else f"{hair_color} {hairstyle}")
    elif hair_color:
        parts.append(f"{hair_color} hair")
    elif hairstyle:
        parts.append(hairstyle if "hair" in hairstyle.lower() else f"{hairstyle} hair")
    for field_name in ("signature_outfit", "signature_accessory", "body_build", "profession_marker"):
        value = _spec_value(spec, field_name)
        if value and not _looks_generic(value) and value.lower() not in " ".join(parts).lower():
            parts.append(value)
    return ", ".join(_unique(parts))


def _robot_descriptor(character_id: str, spec: Any) -> str:
    subject_type = _subject_type(character_id, spec)
    kind = "robot" if subject_type == "robot" else subject_type
    noun = " ".join(part for part in [_spec_value(spec, "color_scheme"), _spec_value(spec, "material"), kind] if part).strip()
    parts = [
        noun,
        _spec_value(spec, "shape_features"),
        _spec_value(spec, "signature_parts"),
    ]
    return ", ".join(_unique([part for part in parts if part])) or f"metal {kind}"


def _character_descriptor(character_id: str, spec: Any) -> str:
    subject_type = _subject_type(character_id, spec)
    if subject_type == "animal":
        return _animal_descriptor(character_id, spec)
    if subject_type in {"robot", "object", "vehicle"}:
        return _robot_descriptor(character_id, spec)
    return _human_descriptor(character_id, spec)


def _clean_scene_text(text: str) -> str:
    text = TAG_PATTERN.sub(r"\1", text)
    return _normalize_whitespace(text).rstrip(".")


def _replace_leading_pronoun(text: str, resolved_entities: list[str]) -> str:
    if not resolved_entities:
        return text
    pronoun = LEADING_PRONOUN_PATTERN.match(text or "")
    if not pronoun:
        return text
    token = pronoun.group(1).lower()
    if token == "they" and len(resolved_entities) >= 2:
        replacement = " and ".join(resolved_entities[:2])
    else:
        replacement = resolved_entities[0]
    return LEADING_PRONOUN_PATTERN.sub(replacement, text, count=1)


def _resolve_scene_entities(scene_entities: list[str], text: str, previous_entities: list[str], story_entities: list[str]) -> list[str]:
    entities = _unique(scene_entities)
    pronoun = LEADING_PRONOUN_PATTERN.match(text or "")
    if pronoun:
        token = pronoun.group(1).lower()
        if token == "they":
            base = _unique(previous_entities + story_entities)[:2]
            entities = _unique(base + entities)
        elif previous_entities:
            entities = _unique(previous_entities[:1] + entities)
    if not entities:
        entities = _unique(previous_entities or story_entities[:1])
    return entities


def _compact_consistency_clause(value: str) -> str:
    text = _normalize_whitespace(value)
    if not text:
        return ""
    banned_fragments = [
        "same person across all scenes",
        "consistent face",
        "consistent hairstyle",
        "consistent outfit",
        "consistent background identity",
        "maintain the same background",
        "keep the same background",
        "same scene entities",
        "maintain the same location identity",
    ]
    clauses = []
    for clause in _split_comma_clauses(text):
        low = clause.lower()
        if any(fragment in low for fragment in banned_fragments):
            continue
        clause = re.sub(r"^new scene setting:\s*", "setting: ", clause, flags=re.IGNORECASE)
        clause = re.sub(r"^same setting:\s*", "setting: ", clause, flags=re.IGNORECASE)
        clauses.append(clause.strip())
    return ", ".join(_unique([clause for clause in clauses if clause]))


def _clean_generation_prompt(value: str) -> str:
    text = _normalize_whitespace(value)
    if not text:
        return ""
    text = re.sub(r"\b[A-Z][A-Za-z0-9_-]*\s+is\.\s*", "", text)
    banned_fragments = [
        "same person across all scenes",
        "consistent face",
        "consistent hairstyle",
        "consistent outfit",
        "consistent background identity",
        "maintain the same background identity",
    ]
    clauses: list[str] = []
    for clause in _split_comma_clauses(text):
        low = clause.lower()
        if any(fragment in low for fragment in banned_fragments):
            continue
        clauses.append(clause.strip())
    return ", ".join(_unique([clause for clause in clauses if clause]))


def _strip_empty_identity_sentences(text: str) -> str:
    return re.sub(r"\b[A-Z][A-Za-z0-9_-]*\s+is\.\s*", "", _normalize_whitespace(text))


def _remove_full_identity_sentences(text: str, entity_names: list[str]) -> str:
    cleaned = _strip_empty_identity_sentences(text)
    for entity in entity_names:
        name = re.escape(entity)
        cleaned = re.sub(
            rf"\b{name}\s+is\s+(?:an?|the)?\s*[^.]*\b(?:human|person|man|woman|boy|girl|cat|dog|bird|animal|robot)\b[^.]*\.\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return _normalize_whitespace(cleaned).strip(" ,.;")


def _strip_leading_entity_names(text: str, entity_names: list[str]) -> str:
    cleaned = _normalize_whitespace(text).strip(" ,.;")
    if not cleaned or not entity_names:
        return cleaned
    if len(entity_names) >= 2:
        pair = rf"{re.escape(entity_names[0])}\s+and\s+{re.escape(entity_names[1])}"
        cleaned = re.sub(rf"^{pair}\s+(?:are|is)\s+", "", cleaned, flags=re.IGNORECASE)
    for entity in entity_names:
        cleaned = re.sub(rf"^{re.escape(entity)}\s+(?:is|are)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,.;")


def _format_setting_clause(value: str) -> str:
    normalized = _normalize_whitespace(value).strip(" ,.;")
    if not normalized:
        return ""
    lowered = normalized.lower()
    if lowered.startswith(("in ", "at ", "on ", "inside ", "near ", "under ", "over ")):
        return normalized
    if any(term in lowered for term in ("table", "wall", "door", "window", "bench")):
        return f"at a {normalized}" if not lowered.startswith(("a ", "an ", "the ")) else f"at {normalized}"
    article = "an" if normalized[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"in {article} {normalized}" if not lowered.startswith(("a ", "an ", "the ")) else f"in {normalized}"


def _clean_framing_clause(value: str, subject_types: list[str]) -> str:
    framing = _normalize_whitespace(value).strip(" ,.;")
    if not framing:
        return ""
    if subject_types and all(subject_type == "animal" for subject_type in subject_types):
        framing = re.sub(r"\bmedium two-shot\b", "", framing, flags=re.IGNORECASE)
        framing = re.sub(r"\btwo-person\b", "two-animal", framing, flags=re.IGNORECASE)
        framing = re.sub(r"\bcharacters\b", "animals", framing, flags=re.IGNORECASE)
    return _normalize_whitespace(framing).strip(" ,.;")


def _story_scene_prompt_v2(
    *,
    tags: str,
    scene_text: str,
    spec: PromptSpec | None,
    scene_plan: dict[str, Any] | None,
    resolved_entities: list[str],
    subject_types: dict[str, str],
) -> str:
    scene_plan = scene_plan or {}
    resolved_subject_types = [subject_types.get(entity, "human") for entity in resolved_entities]
    all_animals = bool(resolved_subject_types) and all(subject_type == "animal" for subject_type in resolved_subject_types)
    has_nonhuman = any(subject_type in {"animal", "robot", "object", "vehicle"} for subject_type in resolved_subject_types)

    action_source = (
        _normalize_whitespace(scene_plan.get("interaction_summary"))
        or _normalize_whitespace(getattr(spec, "action_prompt", ""))
        or _remove_full_identity_sentences(getattr(spec, "generation_prompt", "") if spec else "", resolved_entities)
        or scene_text
    )
    action = _strip_leading_entity_names(_remove_full_identity_sentences(action_source, resolved_entities), resolved_entities)
    if not action:
        action = _strip_leading_entity_names(scene_text, resolved_entities)

    spatial_relation = _normalize_whitespace(scene_plan.get("spatial_relation"))
    framing = _clean_framing_clause(_normalize_whitespace(scene_plan.get("framing")), resolved_subject_types)
    setting_focus = _format_setting_clause(_normalize_whitespace(scene_plan.get("setting_focus")))

    clauses = [action, spatial_relation, framing, setting_focus]
    if len(resolved_entities) >= 2:
        if all_animals:
            clauses.extend(["both animals visible", "two-animal composition"])
        elif has_nonhuman:
            clauses.extend(["all subjects visible", "clear multi-subject composition"])
        else:
            clauses.extend(["both characters visible", "medium two-shot"])
    else:
        if framing and "shot" in framing.lower():
            clauses.append("")
        else:
            clauses.append("medium shot")
        clauses.append("action readable")
    cleaned_clauses = _unique([clause for clause in clauses if clause])
    prompt = f"{tags} {', '.join(cleaned_clauses)}"
    prompt = re.sub(r"\s+,", ",", prompt)
    return _normalize_whitespace(prompt)


def _storydiffusion_tags(resolved_entities: list[str]) -> str:
    return " ".join(f"[{entity}]" for entity in resolved_entities) if resolved_entities else "[NC]"


def _scene_plan_value(scene_plan: dict[str, Any], field_name: str) -> str:
    return _normalize_whitespace(scene_plan.get(field_name))


def _natural_action_source(
    *,
    scene_text: str,
    spec: PromptSpec | None,
    scene_plan: dict[str, Any],
    resolved_entities: list[str],
) -> str:
    scene_text = _remove_full_identity_sentences(scene_text, resolved_entities)
    generation_prompt = _remove_full_identity_sentences(getattr(spec, "generation_prompt", "") if spec else "", resolved_entities)
    generation_prompt = re.sub(r"^LLM optimized\s+", "", generation_prompt, flags=re.IGNORECASE).strip(" ,.;")
    action_prompt = _normalize_whitespace(getattr(spec, "action_prompt", "") if spec else "")
    interaction = _scene_plan_value(scene_plan, "interaction_summary")
    candidates = [scene_text, interaction, action_prompt, generation_prompt]
    for candidate in candidates:
        cleaned = _remove_full_identity_sentences(candidate, resolved_entities).strip(" ,.;")
        if cleaned and cleaned.lower() not in {"unknown", "unspecified"}:
            return cleaned
    return scene_text


def _gerund_phrase(text: str) -> str:
    replacements = {
        "hides": "hiding",
        "hide": "hiding",
        "watches": "watching",
        "watch": "watching",
        "looks": "looking",
        "look": "looking",
        "moves": "moving",
        "move": "moving",
        "runs": "running",
        "run": "running",
        "walks": "walking",
        "walk": "walking",
        "stands": "standing",
        "stand": "standing",
        "sits": "sitting",
        "sit": "sitting",
        "waits": "waiting",
        "wait": "waiting",
        "repairs": "repairing",
        "repair": "repairing",
        "reads": "reading",
        "read": "reading",
        "takes": "taking",
        "take": "taking",
    }
    phrase = text.strip(" ,.;")
    if not phrase:
        return phrase
    if re.search(r"\bhides?\s+in\s+a\s+corner\s+and\s+watches?\b", phrase, flags=re.IGNORECASE):
        return re.sub(
            r"\bhides?\s+in\s+a\s+corner\s+and\s+watches?\b",
            "hiding and watching from a corner",
            phrase,
            count=1,
            flags=re.IGNORECASE,
        )
    first, *rest = phrase.split(" ", 1)
    mapped = replacements.get(first.lower())
    if mapped:
        return " ".join([mapped] + rest)
    return phrase


def _naturalize_action(text: str, resolved_entities: list[str]) -> str:
    action = _normalize_whitespace(text).strip(" ,.;")
    if not action:
        return action
    if len(resolved_entities) == 1:
        entity = resolved_entities[0]
        stripped = _strip_leading_entity_names(action, [entity])
        stripped = re.sub(rf"^{re.escape(entity)}\s+", "", stripped, count=1, flags=re.IGNORECASE).strip(" ,.;")
        if stripped != action:
            return _gerund_phrase(stripped)
        return _gerund_phrase(action)
    return re.sub(r"\b(are|is)\s+talking\b", "talking", action, flags=re.IGNORECASE).strip(" ,.;")


def _setting_already_present(action: str, setting: str) -> bool:
    if not action or not setting:
        return False
    action_low = action.lower()
    setting_low = setting.lower()
    if setting_low in action_low:
        return True
    setting_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", setting_low)
        if len(token) >= 4 and token not in {"with", "from", "inside", "near", "under", "over"}
    }
    return any(token in action_low for token in setting_tokens)


def _natural_setting_clause(setting_focus: str, action: str) -> str:
    setting = _format_setting_clause(setting_focus)
    if _setting_already_present(action, setting):
        return ""
    return setting


def _natural_spatial_clause(spatial_relation: str, scene_text: str) -> str:
    spatial = _normalize_whitespace(spatial_relation).strip(" ,.;")
    if not spatial:
        return ""
    spatial_low = spatial.lower()
    scene_low = scene_text.lower()
    if ("left" in spatial_low or "right" in spatial_low) and not ("left" in scene_low or "right" in scene_low):
        return ""
    return spatial


def _natural_framing_clause(value: str, subject_types: list[str]) -> str:
    framing = _clean_framing_clause(value, subject_types)
    if not framing:
        return ""
    allowed = [
        "close-up",
        "close up",
        "medium shot",
        "wide shot",
        "full body",
        "full-body",
        "medium two-shot",
        "two-shot",
    ]
    low = framing.lower()
    if any(term in low for term in allowed):
        return framing
    return ""


def _contains_full_identity_sentence(prompt: str, entity_names: list[str]) -> bool:
    for entity in entity_names:
        if re.search(rf"\b{re.escape(entity)}\s+is\s+(?:an?|the)?\s*[^,.;]+", prompt, flags=re.IGNORECASE):
            return True
    return False


def _validate_natural_scene_prompt(
    *,
    prompt: str,
    tags: str,
    resolved_entities: list[str],
    resolved_subject_types: list[str],
    spatial_relation: str,
    scene_text: str,
) -> list[str]:
    warnings: list[str] = []
    low = prompt.lower()
    if not prompt.startswith(tags):
        warnings.append("missing_exact_visible_character_tags")
    if any(subject_type == "animal" for subject_type in resolved_subject_types):
        for banned in ("person", "outfit", "clothes"):
            if banned in low:
                warnings.append(f"animal_prompt_contains_{banned}")
    if any(subject_type == "robot" for subject_type in resolved_subject_types):
        for banned in ("human person", "outfit", "clothes"):
            if banned in low:
                warnings.append(f"robot_prompt_contains_{banned.replace(' ', '_')}")
    for banned in ("unknown", "unspecified", "is."):
        if banned in low:
            warnings.append(f"prompt_contains_{banned.replace('.', '').replace(' ', '_')}")
    if _contains_full_identity_sentence(prompt, resolved_entities):
        warnings.append("prompt_contains_full_identity_sentence")
    spatial_low = spatial_relation.lower()
    scene_low = scene_text.lower()
    if ("left" in low or "right" in low) and not (
        ("left" in spatial_low or "right" in spatial_low) and ("left" in scene_low or "right" in scene_low)
    ):
        warnings.append("left_right_relation_not_supported_by_story_text")
    return _unique(warnings)


def _story_scene_prompt_natural(
    *,
    tags: str,
    scene_text: str,
    spec: PromptSpec | None,
    scene_plan: dict[str, Any] | None,
    resolved_entities: list[str],
    subject_types: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    scene_plan = scene_plan or {}
    resolved_subject_types = [subject_types.get(entity, "human") for entity in resolved_entities]
    action_source = _natural_action_source(
        scene_text=scene_text,
        spec=spec,
        scene_plan=scene_plan,
        resolved_entities=resolved_entities,
    )
    action = _naturalize_action(action_source, resolved_entities)
    spatial_relation = _natural_spatial_clause(_scene_plan_value(scene_plan, "spatial_relation"), scene_text)
    setting = _natural_setting_clause(_scene_plan_value(scene_plan, "setting_focus"), action)
    framing = _natural_framing_clause(_scene_plan_value(scene_plan, "framing"), resolved_subject_types)

    clauses = _unique([action, spatial_relation, setting, framing])
    prompt = f"{tags} {', '.join(clauses)}" if clauses else f"{tags} {scene_text}".strip()
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r"\bin a open\b", "in an open", prompt, flags=re.IGNORECASE)
    prompt = _normalize_whitespace(prompt).strip(" ,.;")
    warnings = _validate_natural_scene_prompt(
        prompt=prompt,
        tags=tags,
        resolved_entities=resolved_entities,
        resolved_subject_types=resolved_subject_types,
        spatial_relation=_scene_plan_value(scene_plan, "spatial_relation"),
        scene_text=scene_text,
    )
    return prompt, {
        "natural_scene_prompt": prompt,
        "validation_warnings": warnings,
        "structured_source_fields": {
            "action_source": action_source,
            "action": action,
            "setting_focus": _scene_plan_value(scene_plan, "setting_focus"),
            "spatial_relation": _scene_plan_value(scene_plan, "spatial_relation"),
            "used_spatial_relation": spatial_relation,
            "framing": _scene_plan_value(scene_plan, "framing"),
            "used_framing": framing,
            "resolved_subject_types": resolved_subject_types,
        },
    }


def _saved_image_prompt_map(
    scene_prompts: list[str],
    source_fields: list[dict[str, Any]],
    story_frame_start_index: int,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for index, prompt in enumerate(scene_prompts):
        source = source_fields[index] if index < len(source_fields) else {}
        mapping[f"image_{index:03d}.png"] = {
            "prompt_array_index": story_frame_start_index + index,
            "story_scene_prompt_index": index,
            "scene_id": source.get("scene_id"),
            "prompt": prompt,
        }
    return mapping


def _identity_prompt(entity: str, descriptor: str, subject_type: str) -> str:
    if subject_type == "animal":
        return (
            f"[{entity}] full body animal character reference of {entity}, {descriptor}, "
            "single animal only, centered, neutral pose, simple background"
        )
    if subject_type in {"robot", "object", "vehicle"}:
        label = "robot" if subject_type == "robot" else subject_type
        return (
            f"[{entity}] full body {label} character reference of {entity}, {descriptor}, "
            f"single {label} only, centered, neutral pose, simple background"
        )
    return (
        f"[{entity}] full body character reference of {entity}, {descriptor}, "
        "clear face, complete outfit visible, single character only, centered, neutral pose, simple background"
    )


def _identity_reference_prompt_v2(entity: str, descriptor: str, subject_type: str, variant_index: int) -> str:
    animal_views = ["full body animal reference", "side view animal reference", "three-quarter animal reference"]
    human_views = ["full body character reference", "side view character reference", "three-quarter character reference"]
    robot_views = ["full body robot reference", "side view robot reference", "three-quarter robot reference"]
    if subject_type == "animal":
        view = animal_views[variant_index % len(animal_views)]
        return f"[{entity}] {view}, {descriptor}, single animal only, centered, neutral pose, simple background"
    if subject_type in {"robot", "object", "vehicle"}:
        label = "robot" if subject_type == "robot" else subject_type
        views = robot_views if subject_type == "robot" else [f"full body {label} reference", f"side view {label} reference", f"three-quarter {label} reference"]
        view = views[variant_index % len(views)]
        return f"[{entity}] {view}, {descriptor}, single {label} only, centered, neutral pose, simple background"
    view = human_views[variant_index % len(human_views)]
    return f"[{entity}] {view}, {descriptor}, single character only, centered, neutral pose, simple background"


def _identity_reference_prompts_v2(
    story_entities: list[str],
    descriptors: dict[str, str],
    subject_types: dict[str, str],
    identity_prompts_per_character: int,
) -> list[str]:
    prompts: list[str] = []
    per_character = max(1, int(identity_prompts_per_character))
    for entity in story_entities:
        for variant_index in range(per_character):
            prompts.append(_identity_reference_prompt_v2(entity, descriptors[entity], subject_types[entity], variant_index))
    return prompts


def render_clean_native_storydiffusion_prompts(
    story: Story,
    prompt_specs: dict[str, PromptSpec],
    character_specs: dict[str, Any] | None = None,
) -> NativeStoryDiffusionPromptRender:
    character_specs = character_specs or {}
    story_entities = list(story.all_entities or [])
    if not story_entities:
        story_entities = ["Subject"]

    descriptors = {
        entity: _character_descriptor(entity, _character_spec_for(character_specs, entity))
        for entity in story_entities
    }
    subject_types = {
        entity: _subject_type(entity, _character_spec_for(character_specs, entity))
        for entity in story_entities
    }
    general_prompt = "\n".join(f"[{entity}] {descriptors[entity]}" for entity in story_entities)
    identity_prompts = [
        _identity_prompt(entity, descriptors[entity], subject_types[entity])
        for entity in story_entities
    ]

    scene_prompts: list[str] = []
    source_fields: list[dict[str, Any]] = []
    previous_entities: list[str] = []
    for scene in story.scenes:
        spec = prompt_specs.get(scene.scene_id)
        resolved_entities = _resolve_scene_entities(scene.entities, scene.clean_text, previous_entities, story_entities)
        clean_text = _replace_leading_pronoun(_clean_scene_text(scene.clean_text), resolved_entities)
        tags = " ".join(f"[{entity}]" for entity in resolved_entities) if resolved_entities else "[NC]"
        generation_prompt = _clean_generation_prompt(spec.generation_prompt) if spec else ""
        clauses = [generation_prompt or clean_text]
        if spec:
            consistency = _compact_consistency_clause(spec.scene_consistency_prompt)
            prompt_base = ", ".join(clauses).lower()
            if consistency and consistency.lower() not in prompt_base:
                clauses.append(consistency)
        prompt_base = ", ".join(clauses).lower()
        if len(resolved_entities) >= 2:
            resolved_subject_types = [subject_types.get(entity, "human") for entity in resolved_entities]
            if resolved_subject_types and all(subject_type == "animal" for subject_type in resolved_subject_types):
                clauses.extend(["both animals visible", "two-animal composition"])
            elif any(subject_type in {"robot", "object", "vehicle"} for subject_type in resolved_subject_types):
                clauses.extend(["all subjects visible", "clear multi-subject composition"])
            else:
                clauses.extend(["both characters visible", "medium two-shot"])
        else:
            if not any(term in prompt_base for term in ("wide shot", "medium shot", "full-body", "full body")):
                clauses.append("medium full-body shot")
            clauses.append("action readable")
        prompt = f"{tags} {', '.join(_unique([clause for clause in clauses if clause]))}"
        scene_prompts.append(prompt)
        source_fields.append(
            {
                "scene_id": scene.scene_id,
                "raw_text": scene.raw_text,
                "clean_text": scene.clean_text,
                "scene_entities": list(scene.entities),
                "resolved_entities": resolved_entities,
                "action_prompt": spec.action_prompt if spec else "",
                "scene_consistency_prompt": spec.scene_consistency_prompt if spec else "",
                "generation_prompt": spec.generation_prompt if spec else "",
            }
        )
        if resolved_entities:
            previous_entities = resolved_entities

    final_prompt_array = identity_prompts + scene_prompts
    return NativeStoryDiffusionPromptRender(
        general_prompt=general_prompt,
        identity_prompts=identity_prompts,
        scene_prompts=scene_prompts,
        final_prompt_array=final_prompt_array,
        save_image_start_index=len(identity_prompts),
        character_specs=character_specs,
        source_fields=source_fields,
    )


def render_clean_v2_native_storydiffusion_prompts(
    story: Story,
    prompt_specs: dict[str, PromptSpec],
    character_specs: dict[str, Any] | None = None,
    *,
    scene_plans: dict[str, dict[str, Any]] | None = None,
    identity_prompts_per_character: int = 1,
) -> NativeStoryDiffusionPromptRender:
    character_specs = character_specs or {}
    scene_plans = scene_plans or {}
    story_entities = list(story.all_entities or [])
    if not story_entities:
        story_entities = ["Subject"]

    descriptors = {
        entity: _character_descriptor_v2(entity, _character_spec_for(character_specs, entity))
        for entity in story_entities
    }
    subject_types = {
        entity: _subject_type(entity, _character_spec_for(character_specs, entity))
        for entity in story_entities
    }
    general_prompt = "\n".join(f"[{entity}] {descriptors[entity]}" for entity in story_entities)
    per_character = max(1, int(identity_prompts_per_character))
    identity_prompts = _identity_reference_prompts_v2(story_entities, descriptors, subject_types, per_character)

    scene_prompts: list[str] = []
    source_fields: list[dict[str, Any]] = []
    previous_entities: list[str] = []
    for scene in story.scenes:
        spec = prompt_specs.get(scene.scene_id)
        resolved_entities = _resolve_scene_entities(scene.entities, scene.clean_text, previous_entities, story_entities)
        clean_text = _replace_leading_pronoun(_clean_scene_text(scene.clean_text), resolved_entities)
        tags = " ".join(f"[{entity}]" for entity in resolved_entities) if resolved_entities else "[NC]"
        prompt = _story_scene_prompt_v2(
            tags=tags,
            scene_text=clean_text,
            spec=spec,
            scene_plan=scene_plans.get(scene.scene_id),
            resolved_entities=resolved_entities,
            subject_types=subject_types,
        )
        scene_prompts.append(prompt)
        source_fields.append(
            {
                "scene_id": scene.scene_id,
                "raw_text": scene.raw_text,
                "clean_text": scene.clean_text,
                "scene_entities": list(scene.entities),
                "resolved_entities": resolved_entities,
                "action_prompt": spec.action_prompt if spec else "",
                "scene_consistency_prompt": spec.scene_consistency_prompt if spec else "",
                "generation_prompt": spec.generation_prompt if spec else "",
                "scene_plan": dict(scene_plans.get(scene.scene_id, {})),
            }
        )
        if resolved_entities:
            previous_entities = resolved_entities

    final_prompt_array = identity_prompts + scene_prompts
    story_frame_start_index = len(identity_prompts)
    return NativeStoryDiffusionPromptRender(
        general_prompt=general_prompt,
        identity_prompts=identity_prompts,
        scene_prompts=scene_prompts,
        final_prompt_array=final_prompt_array,
        save_image_start_index=story_frame_start_index,
        character_specs=character_specs,
        source_fields=source_fields,
        identity_prompts_per_character=per_character,
        identity_reference_prompts=identity_prompts,
        story_scene_prompts=scene_prompts,
        saved_image_prompt_map=_saved_image_prompt_map(scene_prompts, source_fields, story_frame_start_index),
    )


def render_natural_native_storydiffusion_prompts(
    story: Story,
    prompt_specs: dict[str, PromptSpec],
    character_specs: dict[str, Any] | None = None,
    *,
    scene_plans: dict[str, dict[str, Any]] | None = None,
    identity_prompts_per_character: int = 1,
) -> NativeStoryDiffusionPromptRender:
    character_specs = character_specs or {}
    scene_plans = scene_plans or {}
    story_entities = list(story.all_entities or [])
    if not story_entities:
        story_entities = ["Subject"]

    descriptors = {
        entity: _character_descriptor_v2(entity, _character_spec_for(character_specs, entity))
        for entity in story_entities
    }
    subject_types = {
        entity: _subject_type(entity, _character_spec_for(character_specs, entity))
        for entity in story_entities
    }
    general_prompt = "\n".join(f"[{entity}] {descriptors[entity]}" for entity in story_entities)
    per_character = max(1, int(identity_prompts_per_character))
    identity_prompts = _identity_reference_prompts_v2(story_entities, descriptors, subject_types, per_character)

    scene_prompts: list[str] = []
    source_fields: list[dict[str, Any]] = []
    previous_entities: list[str] = []
    for scene in story.scenes:
        spec = prompt_specs.get(scene.scene_id)
        resolved_entities = _resolve_scene_entities(scene.entities, scene.clean_text, previous_entities, story_entities)
        clean_text = _replace_leading_pronoun(_clean_scene_text(scene.clean_text), resolved_entities)
        tags = _storydiffusion_tags(resolved_entities)
        prompt, natural_debug = _story_scene_prompt_natural(
            tags=tags,
            scene_text=clean_text,
            spec=spec,
            scene_plan=scene_plans.get(scene.scene_id),
            resolved_entities=resolved_entities,
            subject_types=subject_types,
        )
        scene_prompts.append(prompt)
        source_fields.append(
            {
                "scene_id": scene.scene_id,
                "raw_text": scene.raw_text,
                "clean_text": scene.clean_text,
                "scene_entities": list(scene.entities),
                "resolved_entities": resolved_entities,
                "action_prompt": spec.action_prompt if spec else "",
                "scene_consistency_prompt": spec.scene_consistency_prompt if spec else "",
                "generation_prompt": spec.generation_prompt if spec else "",
                "scene_plan": dict(scene_plans.get(scene.scene_id, {})),
                **natural_debug,
            }
        )
        if resolved_entities:
            previous_entities = resolved_entities

    final_prompt_array = identity_prompts + scene_prompts
    story_frame_start_index = len(identity_prompts)
    return NativeStoryDiffusionPromptRender(
        general_prompt=general_prompt,
        identity_prompts=identity_prompts,
        scene_prompts=scene_prompts,
        final_prompt_array=final_prompt_array,
        save_image_start_index=story_frame_start_index,
        character_specs=character_specs,
        source_fields=source_fields,
        identity_prompts_per_character=per_character,
        identity_reference_prompts=identity_prompts,
        story_scene_prompts=scene_prompts,
        saved_image_prompt_map=_saved_image_prompt_map(scene_prompts, source_fields, story_frame_start_index),
    )
