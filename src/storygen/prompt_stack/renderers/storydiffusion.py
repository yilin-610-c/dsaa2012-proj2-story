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
    for value in values:
        value = str(value or "").strip()
        if value and value not in out:
            out.append(value)
    return out


def _spec_value(spec: Any, field_name: str) -> str:
    if isinstance(spec, dict):
        return _normalize_whitespace(spec.get(field_name, ""))
    return _normalize_whitespace(getattr(spec, field_name, ""))


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
        entity: _character_descriptor(entity, character_specs.get(entity, {}))
        for entity in story_entities
    }
    subject_types = {
        entity: _subject_type(entity, character_specs.get(entity, {}))
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
