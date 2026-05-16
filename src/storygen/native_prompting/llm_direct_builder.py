from __future__ import annotations

import json
from typing import Any, Callable

from storygen.llm_client import BaseLLMClient, LLMResponse, build_llm_client
from storygen.native_prompting.repair import diff_payload_fields
from storygen.native_prompting.types import ALLOWED_TARGET_BACKENDS, NativePromptPayload
from storygen.native_prompting.validator import (
    NativePromptValidationConfig,
    NativePromptValidationError,
    ValidationIssue,
    blocking_issues,
    hard_issues,
    issue_messages,
    issues_to_dicts,
    normalize_targets,
    repair_issues,
    validate_native_prompt_payload,
    warning_issues,
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
        self.last_validation_issues: list[dict[str, Any]] = []
        self.last_warnings: list[dict[str, Any]] = []
        self.last_unresolved_errors: list[dict[str, Any]] = []
        self.last_validation_status = "passed"
        self.last_generation_allowed = True
        self.last_repair_attempts_used = 0
        self.last_payload: NativePromptPayload | None = None

    def build(self, story: Story) -> NativePromptPayload:
        self._validate_targets()
        initial_response = self._generate(story)
        initial_payload = NativePromptPayload.from_dict(initial_response.parsed_json)
        initial_issues = validate_native_prompt_payload(
            initial_payload,
            story,
            targets=self.targets,
            config=self._validation_config(),
        )
        current_payload = initial_payload
        current_json = initial_response.parsed_json
        current_issues = initial_issues
        validation_rounds = [
            {
                "round": 0,
                "source": "initial",
                "issues": issues_to_dicts(initial_issues),
            }
        ]
        repair_records: list[dict[str, Any]] = []
        repair_diffs: list[dict[str, Any]] = []
        self.last_response_record = {
            "initial_response": {
                "raw_text": initial_response.raw_text,
                "parsed_json": initial_response.parsed_json,
                "metadata": initial_response.metadata,
            },
            "validation_issues": issues_to_dicts(initial_issues),
            "validation_errors": issue_messages(blocking_issues(initial_issues)),
            "warnings": issues_to_dicts(warning_issues(initial_issues)),
            "repair_attempted": False,
            "repair_attempts_used": 0,
            "validation_rounds": validation_rounds,
            "repair_diff": [],
        }

        repair_attempts = int(self.direct_config.get("repair_attempts", 1))
        for attempt_index in range(max(0, repair_attempts)):
            if not blocking_issues(current_issues):
                break
            repair_response = self._repair(story, current_json, current_issues)
            repaired_payload = NativePromptPayload.from_dict(repair_response.parsed_json)
            repaired_issues = validate_native_prompt_payload(
                repaired_payload,
                story,
                targets=self.targets,
                config=self._validation_config(),
            )
            diff = diff_payload_fields(current_json, repair_response.parsed_json)
            repair_diffs.extend(diff)
            repair_record = {
                "attempt": attempt_index + 1,
                "request_issues": issues_to_dicts(blocking_issues(current_issues)),
                "response": {
                    "raw_text": repair_response.raw_text,
                    "parsed_json": repair_response.parsed_json,
                    "metadata": repair_response.metadata,
                },
                "validation_issues": issues_to_dicts(repaired_issues),
                "validation_errors": issue_messages(blocking_issues(repaired_issues)),
                "warnings": issues_to_dicts(warning_issues(repaired_issues)),
                "repair_diff": diff,
            }
            repair_records.append(repair_record)
            validation_rounds.append(
                {
                    "round": attempt_index + 1,
                    "source": "repair",
                    "issues": issues_to_dicts(repaired_issues),
                }
            )
            current_payload = repaired_payload
            current_json = repair_response.parsed_json
            current_issues = repaired_issues

        decision = self._validation_decision(current_issues, repair_attempts_used=len(repair_records))
        final_blocking = blocking_issues(current_issues)
        final_warnings = warning_issues(current_issues)
        self.last_validation_errors = issue_messages(blocking_issues(initial_issues))
        self.last_repair_errors = issue_messages(final_blocking) if repair_records else []
        self.last_repair_diff = repair_diffs
        self.last_validation_issues = issues_to_dicts(current_issues)
        self.last_warnings = issues_to_dicts(final_warnings)
        self.last_unresolved_errors = issues_to_dicts(final_blocking)
        self.last_validation_status = str(decision["validation_status"])
        self.last_generation_allowed = bool(decision["generation_allowed"])
        self.last_repair_attempts_used = len(repair_records)
        self.last_response_record.update(
            {
                "repair_attempted": bool(repair_records),
                "repair_attempts_used": len(repair_records),
                "repair_attempts": repair_records,
                "repair_response": repair_records[-1]["response"] if repair_records else None,
                "repair_validation_issues": issues_to_dicts(current_issues) if repair_records else [],
                "repair_validation_errors": issue_messages(final_blocking) if repair_records else [],
                "repair_diff": repair_diffs,
                "validation_rounds": validation_rounds,
                "final_validation_issues": issues_to_dicts(current_issues),
                "validation_status": decision["validation_status"],
                "unresolved_errors": issues_to_dicts(final_blocking),
                "warnings": issues_to_dicts(final_warnings),
                "generation_allowed": decision["generation_allowed"],
                "validation_policy": decision,
            }
        )
        if not decision["generation_allowed"]:
            raise NativePromptValidationError(final_blocking)
        self.last_payload = current_payload
        return current_payload

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
            "validation_issues": list(self.last_validation_issues),
            "warnings": list(self.last_warnings),
            "validation_status": self.last_validation_status,
            "repair_attempts_used": self.last_repair_attempts_used,
            "unresolved_errors": list(self.last_unresolved_errors),
            "generation_allowed": self.last_generation_allowed,
            "_llm_response_record": self.last_response_record,
        }

    def _generate(self, story: Story) -> LLMResponse:
        client = self.llm_client or build_llm_client(self.llm_config)
        response = client.generate_structured(messages=self._build_messages(story), json_schema=self._json_schema())
        self._log("llm_direct_prompt_api_call_completed", metadata=response.metadata)
        return response

    def _repair(self, story: Story, previous_payload: dict[str, Any], validation_issues: list[ValidationIssue]) -> LLMResponse:
        client = self.llm_client or build_llm_client(self.llm_config)
        response = client.generate_structured(
            messages=self._build_repair_messages(story, previous_payload, validation_issues),
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

    def _validation_decision(self, issues: list[ValidationIssue], *, repair_attempts_used: int) -> dict[str, Any]:
        policy = str(self.direct_config.get("validation_policy", "strict")).strip().lower()
        allow_boundary = bool(self.direct_config.get("allow_generation_with_boundary_errors", False))
        on_hard_error = str(self.direct_config.get("on_hard_error", "fail")).strip().lower()
        hard = hard_issues(issues)
        repair = repair_issues(issues)
        warnings = warning_issues(issues)
        hard_error_action = "fail"
        if hard:
            status = "failed_hard_error"
            allowed = False
            if policy == "best_effort" and on_hard_error == "skip_story":
                hard_error_action = "skip_story"
        elif repair:
            allowed = policy == "best_effort" and allow_boundary
            status = "best_effort_with_unresolved_boundary_errors" if allowed else "failed_hard_error"
        elif warnings:
            status = "passed_with_warnings"
            allowed = True
        else:
            status = "passed"
            allowed = True
        return {
            "validation_status": status,
            "repair_attempts_used": repair_attempts_used,
            "unresolved_errors": issues_to_dicts([*hard, *repair]),
            "warnings": issues_to_dicts(warnings),
            "generation_allowed": allowed,
            "validation_policy": policy,
            "on_hard_error": on_hard_error,
            "hard_error_action": hard_error_action,
            "allow_generation_with_boundary_errors": allow_boundary,
        }

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
                "- scoring_prompt should be short, objective, and semantic, focusing on the main subject and action; avoid mood words, style words, long identity details, camera jargon, and subjective adjectives.\n"
                "- anchor_generation_prompt is for standard diffusion / Anchor / IP-Adapter and must not require StoryDiffusion tags.\n"
                "- anchor_generation_prompt is a story scene prompt, not a reference image prompt; do not include reference-only constraints such as simple/plain/clean/centered background, single-subject, or one-person-in-image wording.\n"
                "- anchor_reference_prompt should show only the character, not a story scene: use neutral standing pose or simple half-body/full-body pose, plain/simple background, no story location, no temporary prop, no scene action, and no other people.\n"
                "- For dual-person scenes, identity_conditioning_subject_id should usually be null; do not over-engineer IP-Adapter conditioning for multi-character scenes.\n"
            )
        storydiffusion_rules = ""
        if "storydiffusion" in self.targets:
            storydiffusion_rules = (
                "- Because target_backends includes storydiffusion, every scene must include final storydiffusion_prompt starting with exact [Character] tags or [NC].\n"
                "- storydiffusion.general_prompt must be compact multi-line stable visual identity text using exact tags only; no story actions, scene settings, personality, preferences, habits, or narrative background.\n"
                "- storydiffusion.identity_reference_prompts must contain final executable identity rows that include exact [Character] tags; identity_prompts_per_character is required.\n"
                "- StoryDiffusion identity rows should stay compact and should not over-constrain pose.\n"
                "- storydiffusion_prompt is inserted directly into StoryDiffusion prompt_array; it must be a concise but visually specific frame prompt, not a bare action or an intermediate phrase.\n"
                "- For storydiffusion_prompt, include the visible action, local setting or important object, pose/spatial relation when relevant, continuity cue when implied, and camera framing when helpful.\n"
                "- Do not put identity-reference-only wording in storydiffusion_prompt, such as simple/plain/clean/centered background, single-subject, or one-person-in-image constraints.\n"
                "- Use [NC] only for a true cutaway with no visible recurring subject. In an ongoing single-character story, include the character if they can naturally observe or experience the environment.\n"
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
            "- stable_identity is short stable visual identity text for inspection and general prompt reasoning.\n"
            "- stable_identity, general_prompt, anchor_reference_prompt, and identity_reference_prompts must describe only stable visual identity. They must not include scene-specific locations, temporary props, actions, story events, personality, preferences, habits, or narrative background.\n"
            "- anchor_reference_prompt is a final executable identity/reference prompt; include an exact [Character] tag plus single-subject and simple/centered-background reference constraints yourself.\n"
            "- Do not include banned layout terms such as character sheet, turnaround, multiple views, or duplicate subject in identity prompts.\n"
            "- Resolve pronouns from story context using exact character ids.\n"
            "- If a scene combines a pronoun with an explicit tagged character and describes an interaction, resolve the pronoun and include both participants in visible ids and scene prompts.\n"
            "- Do not invent unrelated story events.\n"
            "- Scene prompts should prioritize concrete visible content: subject, action, object, setting, spatial relation, and camera framing. Avoid inner thoughts, personality, or vague mood adjectives unless directly visible in the story.\n"
            "- negative prompt fields must not forbid content required by any story scene, such as multiple characters, crowds, traffic, complex backgrounds, vehicles, or props.\n"
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
        validation_issues: list[ValidationIssue],
    ) -> list[dict[str, str]]:
        base_messages = self._build_messages(story)
        repair_prompt = (
            "Your previous JSON failed validation. Return a complete corrected JSON payload. "
            "Do not explain. Do not rely on local code to repair prompt semantics. "
            "Preserve valid fields as much as possible and revise only fields implicated by validation issues.\n"
            f"Original story:\n{story.raw_text}\n"
            f"Target backends:\n{json.dumps(self.targets)}\n"
            f"Validation issues:\n{json.dumps(issues_to_dicts(validation_issues), indent=2)}\n"
            f"Previous payload:\n{json.dumps(previous_payload, indent=2, ensure_ascii=False)}"
        )
        return [*base_messages, {"role": "user", "content": repair_prompt}]

    def _json_schema(self) -> dict[str, Any]:
        string_or_null = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        character_properties: dict[str, Any] = {
            "character_id": {"type": "string"},
            "subject_type": {"type": "string"},
            "stable_identity": {"type": "string"},
        }
        character_required = ["character_id", "subject_type", "stable_identity"]
        if "anchor" in self.targets:
            character_properties["anchor_reference_prompt"] = {"type": "string"}
            character_required.append("anchor_reference_prompt")

        scene_properties: dict[str, Any] = {"scene_id": {"type": "string"}}
        scene_required = ["scene_id"]
        if "anchor" in self.targets:
            scene_properties.update(
                {
                    "visible_character_ids": {"type": "array", "items": {"type": "string"}},
                    "identity_conditioning_subject_id": string_or_null,
                    "anchor_generation_prompt": {"type": "string"},
                    "scoring_prompt": {"type": "string"},
                }
            )
            scene_required.extend(
                [
                    "visible_character_ids",
                    "identity_conditioning_subject_id",
                    "anchor_generation_prompt",
                    "scoring_prompt",
                ]
            )
        if "storydiffusion" in self.targets:
            scene_properties["storydiffusion_prompt"] = {"type": "string"}
            scene_required.append("storydiffusion_prompt")

        top_properties: dict[str, Any] = {
            "target_backends": {"type": "array", "items": {"type": "string"}},
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": character_required,
                    "properties": character_properties,
                },
            },
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": scene_required,
                    "properties": scene_properties,
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
        }
        top_required = ["target_backends", "characters", "scenes", "notes"]
        if "storydiffusion" in self.targets:
            top_properties["storydiffusion"] = {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "general_prompt",
                    "identity_reference_prompts",
                    "identity_prompts_per_character",
                    "identity_negative_prompt_extra",
                    "scene_negative_prompt_extra",
                ],
                "properties": {
                    "general_prompt": {"type": "string"},
                    "identity_reference_prompts": {"type": "array", "items": {"type": "string"}},
                    "identity_prompts_per_character": {"type": "integer"},
                    "identity_negative_prompt_extra": {"type": "string"},
                    "scene_negative_prompt_extra": {"type": "string"},
                },
            }
            top_required.append("storydiffusion")
        return {
            "name": "llm_direct_prompt_payload",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": top_required,
                "properties": top_properties,
            },
        }

    def _log(self, event: str, **metadata: Any) -> None:
        if self.event_logger:
            self.event_logger(event, **metadata)
