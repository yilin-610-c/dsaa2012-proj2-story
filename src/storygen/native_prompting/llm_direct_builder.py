from __future__ import annotations

import json
from typing import Any, Callable

from storygen.llm_client import BaseLLMClient, LLMResponse, build_llm_client
from storygen.native_prompting.repair import diff_payload_fields
from storygen.native_prompting.types import ALLOWED_TARGET_BACKENDS, NativePromptPayload
from storygen.native_prompting.validator import (
    NativePromptValidationConfig,
    NativePromptValidationError,
    normalize_targets,
    validate_native_prompt_payload,
)
from storygen.types import Story


class LLMDirectPromptBuilder:
    def __init__(
        self,
        prompt_config: dict[str, Any],
        *,
        llm_client: BaseLLMClient | None = None,
        event_logger: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.prompt_config = prompt_config
        self.direct_config = prompt_config.get("llm_direct", {}) or {}
        self.llm_config = prompt_config.get("llm", {}) or {}
        self.targets = normalize_targets(self.direct_config.get("targets", ["anchor"]))
        self.llm_client = llm_client
        self.event_logger = event_logger
        self.last_response_record: dict[str, Any] | None = None
        self.last_validation_errors: list[str] = []
        self.last_repair_errors: list[str] = []
        self.last_repair_diff: list[dict[str, Any]] = []
        self.last_payload: NativePromptPayload | None = None

    def build(self, story: Story) -> NativePromptPayload:
        self._validate_targets()
        initial_response = self._generate(story)
        initial_payload = NativePromptPayload.from_dict(initial_response.parsed_json)
        initial_errors = validate_native_prompt_payload(
            initial_payload,
            story,
            targets=self.targets,
            config=self._validation_config(),
        )
        self.last_validation_errors = initial_errors
        self.last_response_record = {
            "initial_response": {
                "raw_text": initial_response.raw_text,
                "parsed_json": initial_response.parsed_json,
                "metadata": initial_response.metadata,
            },
            "validation_errors": list(initial_errors),
            "repair_attempted": False,
            "repair_diff": [],
        }
        if not initial_errors:
            self.last_payload = initial_payload
            return initial_payload

        repair_attempts = int(self.direct_config.get("repair_attempts", 1))
        if repair_attempts < 1:
            raise NativePromptValidationError(initial_errors)

        repair_response = self._repair(story, initial_response.parsed_json, initial_errors)
        repaired_payload = NativePromptPayload.from_dict(repair_response.parsed_json)
        repair_errors = validate_native_prompt_payload(
            repaired_payload,
            story,
            targets=self.targets,
            config=self._validation_config(),
        )
        self.last_repair_errors = repair_errors
        self.last_repair_diff = diff_payload_fields(initial_response.parsed_json, repair_response.parsed_json)
        self.last_response_record.update(
            {
                "repair_attempted": True,
                "repair_response": {
                    "raw_text": repair_response.raw_text,
                    "parsed_json": repair_response.parsed_json,
                    "metadata": repair_response.metadata,
                },
                "repair_validation_errors": list(repair_errors),
                "repair_diff": list(self.last_repair_diff),
            }
        )
        if repair_errors:
            raise NativePromptValidationError(repair_errors)
        self.last_payload = repaired_payload
        return repaired_payload

    def metadata(self) -> dict[str, Any]:
        return {
            "pipeline": "llm_direct",
            "implemented": True,
            "target_backends": list(self.targets),
            "provider": self.llm_config.get("provider", "openai"),
            "model": self.llm_config.get("model", "gpt-4o-2024-08-06"),
            "schema_version": self.direct_config.get("schema_version", "llm_direct_v1"),
            "builder_version": self.direct_config.get("builder_version", "llm_direct_v1"),
            "validation_errors": list(self.last_validation_errors),
            "repair_errors": list(self.last_repair_errors),
            "repair_diff": list(self.last_repair_diff),
            "_llm_response_record": self.last_response_record,
        }

    def _generate(self, story: Story) -> LLMResponse:
        client = self.llm_client or build_llm_client(self.llm_config)
        response = client.generate_structured(messages=self._build_messages(story), json_schema=self._json_schema())
        self._log("llm_direct_prompt_api_call_completed", metadata=response.metadata)
        return response

    def _repair(self, story: Story, previous_payload: dict[str, Any], validation_errors: list[str]) -> LLMResponse:
        client = self.llm_client or build_llm_client(self.llm_config)
        response = client.generate_structured(
            messages=self._build_repair_messages(story, previous_payload, validation_errors),
            json_schema=self._json_schema(),
        )
        self._log("llm_direct_prompt_repair_completed", metadata=response.metadata)
        return response

    def _validate_targets(self) -> None:
        invalid = [target for target in self.targets if target not in ALLOWED_TARGET_BACKENDS]
        if invalid:
            raise NativePromptValidationError([f"Unsupported llm_direct target: {invalid}"])

    def _validation_config(self) -> NativePromptValidationConfig:
        return NativePromptValidationConfig(
            max_scene_prompt_words=int(self.direct_config.get("max_scene_prompt_words", 40)),
            max_scene_prompt_chars=int(self.direct_config.get("max_scene_prompt_chars", 320)),
            max_identity_prompt_words=int(self.direct_config.get("max_identity_prompt_words", 60)),
            max_identity_prompt_chars=int(self.direct_config.get("max_identity_prompt_chars", 420)),
            min_unknown_identity_words=int(self.direct_config.get("min_unknown_identity_words", 4)),
            min_storydiffusion_scene_words=int(self.direct_config.get("min_storydiffusion_scene_words", 6)),
        )

    def _build_messages(self, story: Story) -> list[dict[str, str]]:
        scene_lines = "\n".join(
            f"- {scene.scene_id}: raw={scene.raw_text!r}; clean={scene.clean_text!r}; explicit_entities={scene.entities}"
            for scene in story.scenes
        )
        target_text = ", ".join(self.targets)
        anchor_rules = ""
        if "anchor" in self.targets:
            anchor_rules = (
                "- Because target_backends includes anchor, every scene must include final anchor_generation_prompt, "
                "short semantic scoring_prompt, visible_character_ids, and identity_conditioning_subject_id. "
                "Every character must include final anchor_reference_prompt.\n"
                "- scoring_prompt should be short and semantic, focusing on the main subject and action; avoid style words, long identity details, and camera jargon.\n"
                "- anchor_generation_prompt is for standard diffusion / Anchor / IP-Adapter and must not require StoryDiffusion tags.\n"
                "- anchor_generation_prompt is a story scene prompt, not a reference image prompt; do not include reference-only constraints such as simple/plain/clean/centered background, single-subject, or one-person-in-image wording.\n"
            )
        storydiffusion_rules = ""
        if "storydiffusion" in self.targets:
            storydiffusion_rules = (
                "- Because target_backends includes storydiffusion, every scene must include final storydiffusion_prompt starting with exact [Character] tags or [NC].\n"
                "- storydiffusion.general_prompt must be final multi-line character identity text using exact tags.\n"
                "- storydiffusion.identity_reference_prompts must contain final executable identity rows that include exact [Character] tags; identity_prompts_per_character is required.\n"
                "- storydiffusion_prompt is inserted directly into StoryDiffusion prompt_array; it must be a concise but visually specific frame prompt, not a bare action or an intermediate phrase.\n"
                "- For storydiffusion_prompt, include the visible action, local setting or important object, pose/spatial relation when relevant, continuity cue when implied, and camera framing when helpful.\n"
                "- Do not put identity-reference-only wording in storydiffusion_prompt, such as simple/plain/clean/centered background, single-subject, or one-person-in-image constraints.\n"
            )
        system_prompt = (
            "You write final executable prompts for an automated story image pipeline. "
            "Return only JSON matching the schema. Do not output analysis outside JSON. "
            "All fields ending in _prompt must be final executable prompts written by you, not ingredients for local rendering."
        )
        user_prompt = (
            f"Target backends: {target_text}\n"
            "Story entities from parser: "
            f"{story.all_entities}\n"
            "Scenes:\n"
            f"{scene_lines}\n\n"
            "Rules:\n"
            "- Do not rely on local code to add hair, outfit, species, setting, action, framing, emotion, background, or reference constraints.\n"
            "- subject_type is your judgment and must be one of human, animal, robot, object, vehicle, unknown.\n"
            "- If subject_type is unknown, stable_identity and reference prompts must still be visually concrete and drawable.\n"
            "- stable_identity is short identity text for inspection and general prompt reasoning.\n"
            "- anchor_reference_prompt is a final executable identity/reference prompt; include an exact [Character] tag plus single-subject and simple/centered-background reference constraints yourself.\n"
            "- Do not include banned layout terms such as character sheet, turnaround, multiple views, or duplicate subject in identity prompts.\n"
            "- Resolve pronouns from story context using exact character ids.\n"
            "- Do not invent unrelated story events.\n"
            "- Keep reference-image constraints inside identity/reference prompts only; story scene prompts should describe the actual story environment.\n"
            f"{anchor_rules}"
            f"{storydiffusion_rules}"
            "- notes are debug-only and are not used to assemble prompts.\n"
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _build_repair_messages(
        self,
        story: Story,
        previous_payload: dict[str, Any],
        validation_errors: list[str],
    ) -> list[dict[str, str]]:
        base_messages = self._build_messages(story)
        repair_prompt = (
            "Your previous JSON failed validation. Return a complete corrected JSON payload. "
            "Do not explain. Do not rely on local code to repair prompt semantics.\n"
            f"Validation errors:\n{json.dumps(validation_errors, indent=2)}\n"
            f"Previous payload:\n{json.dumps(previous_payload, indent=2, ensure_ascii=False)}"
        )
        return [*base_messages, {"role": "user", "content": repair_prompt}]

    def _json_schema(self) -> dict[str, Any]:
        string_or_null = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        return {
            "name": "llm_direct_prompt_payload",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target_backends", "characters", "scenes", "storydiffusion", "notes"],
                "properties": {
                    "target_backends": {"type": "array", "items": {"type": "string"}},
                    "characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["character_id", "subject_type", "stable_identity", "anchor_reference_prompt"],
                            "properties": {
                                "character_id": {"type": "string"},
                                "subject_type": {"type": "string"},
                                "stable_identity": {"type": "string"},
                                "anchor_reference_prompt": {"type": "string"},
                            },
                        },
                    },
                    "scenes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "scene_id",
                                "visible_character_ids",
                                "identity_conditioning_subject_id",
                                "anchor_generation_prompt",
                                "storydiffusion_prompt",
                                "scoring_prompt",
                                "action_prompt",
                            ],
                            "properties": {
                                "scene_id": {"type": "string"},
                                "visible_character_ids": {"type": "array", "items": {"type": "string"}},
                                "identity_conditioning_subject_id": string_or_null,
                                "anchor_generation_prompt": {"type": "string"},
                                "storydiffusion_prompt": {"type": "string"},
                                "scoring_prompt": {"type": "string"},
                                "action_prompt": {"type": "string"},
                            },
                        },
                    },
                    "storydiffusion": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "general_prompt",
                            "identity_reference_prompts",
                            "identity_prompts_per_character",
                            "negative_prompt_extra",
                        ],
                        "properties": {
                            "general_prompt": {"type": "string"},
                            "identity_reference_prompts": {"type": "array", "items": {"type": "string"}},
                            "identity_prompts_per_character": {"type": "integer"},
                            "negative_prompt_extra": {"type": "string"},
                        },
                    },
                    "notes": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["identity_reasoning", "continuity_reasoning", "self_check"],
                        "properties": {
                            "identity_reasoning": {"type": "string"},
                            "continuity_reasoning": {"type": "string"},
                            "self_check": {"type": "string"},
                        },
                    },
                },
            },
        }

    def _log(self, event: str, **metadata: Any) -> None:
        if self.event_logger:
            self.event_logger(event, **metadata)
