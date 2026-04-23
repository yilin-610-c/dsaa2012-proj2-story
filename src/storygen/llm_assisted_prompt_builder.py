from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from storygen.character_specs import build_llm_character_specs, build_rule_based_character_specs
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


def _trim_words(text: str, max_words: int | None) -> str:
    if not max_words:
        return text
    words = text.split()
    return " ".join(words[:max_words])


def _trim_text(text: str, *, max_words: int | None, max_chars: int | None) -> str:
    trimmed = _trim_words(_normalize_text(text), max_words)
    if max_chars and len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rsplit(" ", 1)[0].strip() or trimmed[:max_chars].strip()
    return trimmed


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


def _contains_animal_identity(character_prompt: str) -> bool:
    lowered = character_prompt.lower()
    animal_terms = [
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


def _merge_identity_into_generation_prompt(character_prompt: str, generation_prompt: str, max_words: int, max_chars: int) -> str:
    identity_prompt = _ensure_human_identity(character_prompt)
    merged = _join_parts([identity_prompt, generation_prompt])
    return _trim_text(merged, max_words=max_words, max_chars=max_chars)


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
        self.last_route_hints: dict[str, dict[str, Any]] = {}
        self.last_character_specs: dict[str, dict[str, Any]] = {}

    def build_story_prompts(self, story: Story) -> dict[str, PromptSpec]:
        try:
            structured_output = self._load_or_generate_structured_output(story)
            self.last_route_hints = self._build_route_hints(structured_output)
            self.last_character_specs = build_llm_character_specs(structured_output, story)
            return self._build_prompt_specs(story, structured_output)
        except Exception as exc:
            self._log("llm_prompt_validation_failed", error=str(exc))
            if self.llm_config.get("fallback_to_rule_based", True):
                self._log("llm_prompt_fallback_to_rule_based", error=str(exc))
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
            "builder_version": self.llm_config.get("builder_version", "llm_assisted_v7"),
            "cache_enabled": bool(self.cache_config.get("enabled", True)),
            "artifact_path": self.artifact_config.get("path"),
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
            "- identity_cues must be visual and reusable across panels, not personality traits.\n"
            "- global.characters must contain stable visual identity blocks for recurring characters.\n"
            "- character specs may include age_band, gender_presentation, hair_color, hairstyle, skin_tone, body_build, signature_outfit, signature_accessory, and profession_marker.\n"
            "- character specs must not include scene-specific action, pose, emotion, temporary props, object state, lighting, or camera framing.\n"
            "- Leave uncertain character spec fields empty instead of inventing details.\n"
            "- If the main character is human, state that clearly in identity_cues using terms like human woman, human girl, human man, or human person.\n"
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
            "- identity_conditioning_subject_id chooses the single character whose identity anchor should condition this scene. Use a character_id from global.characters only when one character is the clear identity target.\n"
            "- If multiple characters are equally important, or the scene is ambiguous, set identity_conditioning_subject_id to null instead of guessing.\n"
            "- For pronoun-only scenes, resolve the visible characters from story context when possible. If a plural pronoun such as 'They' refers to multiple characters equally, leave identity_conditioning_subject_id null.\n"
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

        normalized_characters = self._normalize_character_specs(global_payload.get("characters"))
        valid_character_ids = _character_id_lookup(normalized_characters)
        normalized_scenes = []
        for scene_payload in scenes_payload:
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
            normalized_scenes.append(
                {
                    "scene_id": scene_payload["scene_id"],
                    "primary_action": _normalize_text(scene_payload.get("primary_action")),
                    "secondary_elements": _normalize_list(scene_payload.get("secondary_elements")),
                    "generation_prompt": generation_prompt,
                    "scoring_prompt": scoring_prompt,
                    "action_prompt": action_prompt,
                    "continuity_subject_ids": _normalize_list(scene_payload.get("continuity_subject_ids")),
                    "continuity_route_hint": continuity_route_hint,
                    "llm_route_change_level": route_change_level,
                    "route_change_level": adjusted_route_change_level,
                    "route_level_adjustment_reason": adjustment_reason,
                    "route_factors": route_factors,
                    "route_reason": route_reason,
                    "identity_conditioning_subject_id": identity_conditioning_subject_id,
                    "primary_visible_character_ids": primary_visible_character_ids,
                }
            )

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

    def _build_route_hints(self, structured_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
        hints = {}
        for scene_payload in structured_output.get("scenes", []):
            scene_id = scene_payload.get("scene_id")
            if not scene_id:
                continue
            hints[scene_id] = {
                "identity_conditioning_subject_id": scene_payload.get("identity_conditioning_subject_id"),
                "primary_visible_character_ids": list(scene_payload.get("primary_visible_character_ids", [])),
                "continuity_subject_ids": list(scene_payload.get("continuity_subject_ids", [])),
                "continuity_route_hint": scene_payload.get("continuity_route_hint"),
                "llm_route_change_level": scene_payload.get("llm_route_change_level"),
                "route_change_level": scene_payload.get("route_change_level"),
                "route_level_adjustment_reason": scene_payload.get("route_level_adjustment_reason"),
                "route_factors": dict(scene_payload.get("route_factors", {})),
                "route_reason": scene_payload.get("route_reason"),
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
        character_prompt = _join_parts([main_character, *identity_cues])
        character_prompt = _ensure_human_identity(character_prompt)
        global_context_prompt = _join_parts([*shared_setting, *style_cues, scene_continuity])

        prompt_specs = {}
        for scene in story.scenes:
            scene_payload = scene_payloads[scene.scene_id]
            local_prompt = _join_parts(
                [
                    scene.clean_text,
                    scene_payload.get("primary_action", ""),
                    *scene_payload.get("secondary_elements", []),
                ]
            )
            full_prompt = _join_parts([style_prompt, character_prompt, global_context_prompt, local_prompt])
            generation_prompt = _merge_identity_into_generation_prompt(
                character_prompt,
                scene_payload["generation_prompt"],
                max_words=int(self.prompt_config.get("generation_max_words", 28)),
                max_chars=int(self.prompt_config.get("generation_max_chars", 220)),
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
        export_dir = Path(self.artifact_config.get("export_dir", "prompt_artifacts/llm_assisted_v7"))
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
            "builder_version": self.llm_config.get("builder_version", "llm_assisted_v7"),
        }

    def _log(self, event: str, **metadata: Any) -> None:
        if self.event_logger is not None:
            self.event_logger(event, **metadata)
