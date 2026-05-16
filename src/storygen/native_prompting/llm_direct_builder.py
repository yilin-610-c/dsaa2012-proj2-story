from __future__ import annotations

import json
import sys
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
        self._reset_failure_metadata()
        self._validate_targets()
        try:
            initial_response = self._generate(story)
        except Exception:
            self._mark_failed_no_payload()
            raise
        initial_payload = self._payload_from_response(initial_response, failure_status="failed_unparseable_payload")
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
            try:
                repair_response = self._repair(story, current_json, current_issues)
            except Exception as exc:
                current_issues = [
                    *current_issues,
                    ValidationIssue(
                        severity="warning",
                        code="llm_repair_api_failed",
                        path="$",
                        message="LLM repair API call failed; continuing with previous usable payload if possible",
                        suspicious_text="",
                        instruction="Retry prompt repair later if stricter validation is required.",
                    ),
                ]
                repair_records.append(
                    {
                        "attempt": attempt_index + 1,
                        "request_issues": issues_to_dicts(blocking_issues(current_issues)),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "validation_issues": issues_to_dicts(current_issues),
                        "validation_errors": issue_messages(blocking_issues(current_issues)),
                        "warnings": issues_to_dicts(warning_issues(current_issues)),
                        "repair_diff": [],
                    }
                )
                break
            try:
                repaired_payload = self._payload_from_response(repair_response, failure_status="failed_unparseable_payload")
            except Exception:
                self._mark_failed_unparseable(current_issues=current_issues, repair_records=repair_records, validation_rounds=validation_rounds)
                raise
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

        decision = self._validation_decision(
            current_payload,
            current_issues,
            story=story,
            repair_attempts_used=len(repair_records),
        )
        final_blocking = list(decision["unresolved_issue_objects"])
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
                "repair_response": repair_records[-1].get("response") if repair_records else None,
                "repair_validation_issues": issues_to_dicts(current_issues) if repair_records else [],
                "repair_validation_errors": issue_messages(final_blocking) if repair_records else [],
                "repair_diff": repair_diffs,
                "validation_rounds": validation_rounds,
                "final_validation_issues": issues_to_dicts(current_issues),
                "validation_status": decision["validation_status"],
                "unresolved_errors": issues_to_dicts(final_blocking),
                "warnings": issues_to_dicts(final_warnings),
                "generation_allowed": decision["generation_allowed"],
                "validation_policy": self._decision_metadata(decision),
            }
        )
        if decision["generation_allowed"] and final_blocking:
            self._warn_unresolved_issues(final_blocking, repair_attempts_used=len(repair_records))
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

    def _validation_decision(
        self,
        payload: NativePromptPayload,
        issues: list[ValidationIssue],
        *,
        story: Story,
        repair_attempts_used: int,
    ) -> dict[str, Any]:
        policy = str(self.direct_config.get("validation_policy", "best_effort")).strip().lower()
        allow_boundary = bool(self.direct_config.get("allow_generation_with_boundary_errors", True))
        on_hard_error = str(self.direct_config.get("on_hard_error", "fail")).strip().lower()
        usability_issues = self._backend_usability_issues(payload, story)
        hard = hard_issues(issues)
        repair = repair_issues(issues)
        warnings = warning_issues(issues)
        hard_error_action = "fail"
        if usability_issues:
            status = "failed_unparseable_payload"
            allowed = False
            hard_error_action = "skip_story" if policy == "best_effort" and on_hard_error == "skip_story" else "fail"
            hard = [*hard, *usability_issues]
        elif hard or repair:
            allowed = policy == "best_effort" and allow_boundary
            status = "best_effort_with_unresolved_issues" if allowed else "failed_unparseable_payload"
            if not allowed and policy == "best_effort" and on_hard_error == "skip_story":
                hard_error_action = "skip_story"
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
            "unresolved_issue_objects": [*hard, *repair],
        }

    def _decision_metadata(self, decision: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in decision.items() if key != "unresolved_issue_objects"}

    def _payload_from_response(self, response: LLMResponse, *, failure_status: str) -> NativePromptPayload:
        if not isinstance(response.parsed_json, dict):
            self._mark_failed_status(failure_status)
            raise NativePromptValidationError(
                [
                    ValidationIssue(
                        severity="hard_error",
                        code=failure_status,
                        path="$",
                        message="LLM response did not contain a parseable JSON object payload",
                        suspicious_text=type(response.parsed_json).__name__,
                        instruction="Return a complete JSON object matching the llm_direct schema.",
                    )
                ]
            )
        return NativePromptPayload.from_dict(response.parsed_json)

    def _backend_usability_issues(self, payload: NativePromptPayload, story: Story) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        scene_by_id = {scene.scene_id: scene for scene in payload.scenes}
        if "anchor" in self.targets:
            if not payload.characters:
                issues.append(self._fatal_issue("anchor_no_characters", "$.characters", "Anchor target has no usable characters."))
            for index, character in enumerate(payload.characters):
                if not character.character_id:
                    issues.append(self._fatal_issue("anchor_character_missing_id", f"$.characters[{index}].character_id", "Anchor character is missing character_id."))
                if not character.anchor_reference_prompt:
                    issues.append(
                        self._fatal_issue(
                            "anchor_character_missing_reference_prompt",
                            f"$.characters[{index}].anchor_reference_prompt",
                            "Anchor character is missing anchor_reference_prompt.",
                        )
                    )
            for index, parsed_scene in enumerate(story.scenes):
                scene = scene_by_id.get(parsed_scene.scene_id)
                if scene is None:
                    issues.append(self._fatal_issue("anchor_missing_scene", f"$.scenes[{index}]", f"Anchor target is missing scene {parsed_scene.scene_id}."))
                    continue
                if not scene.anchor_generation_prompt:
                    issues.append(
                        self._fatal_issue(
                            "anchor_scene_missing_generation_prompt",
                            f"$.scenes[{index}].anchor_generation_prompt",
                            f"Anchor scene {parsed_scene.scene_id} is missing anchor_generation_prompt.",
                        )
                    )
                if not scene.scoring_prompt:
                    issues.append(
                        self._fatal_issue(
                            "anchor_scene_missing_scoring_prompt",
                            f"$.scenes[{index}].scoring_prompt",
                            f"Anchor scene {parsed_scene.scene_id} is missing scoring_prompt.",
                        )
                    )
        if "storydiffusion" in self.targets:
            identity_prompts = [prompt for prompt in payload.storydiffusion.identity_reference_prompts if prompt]
            scene_prompts = [scene.storydiffusion_prompt for scene in payload.scenes if scene.storydiffusion_prompt]
            if not identity_prompts:
                issues.append(
                    self._fatal_issue(
                        "storydiffusion_no_identity_prompts",
                        "$.storydiffusion.identity_reference_prompts",
                        "StoryDiffusion target has no usable identity reference prompts.",
                    )
                )
            if not scene_prompts:
                issues.append(self._fatal_issue("storydiffusion_no_scene_prompts", "$.scenes", "StoryDiffusion target has no usable scene prompts."))
            for index, parsed_scene in enumerate(story.scenes):
                scene = scene_by_id.get(parsed_scene.scene_id)
                if scene is None:
                    issues.append(self._fatal_issue("storydiffusion_missing_scene", f"$.scenes[{index}]", f"StoryDiffusion target is missing scene {parsed_scene.scene_id}."))
                    continue
                if not scene.storydiffusion_prompt:
                    issues.append(
                        self._fatal_issue(
                            "storydiffusion_scene_missing_prompt",
                            f"$.scenes[{index}].storydiffusion_prompt",
                            f"StoryDiffusion scene {parsed_scene.scene_id} is missing storydiffusion_prompt.",
                        )
                    )
        return issues

    def _fatal_issue(self, code: str, path: str, message: str) -> ValidationIssue:
        return ValidationIssue(
            severity="hard_error",
            code=code,
            path=path,
            message=message,
            suspicious_text="",
            instruction="Return the minimum backend fields needed to construct generation prompts.",
        )

    def _reset_failure_metadata(self) -> None:
        self.last_response_record = None
        self.last_validation_errors = []
        self.last_repair_errors = []
        self.last_repair_diff = []
        self.last_validation_issues = []
        self.last_warnings = []
        self.last_unresolved_errors = []
        self.last_validation_status = "failed_no_payload"
        self.last_generation_allowed = False
        self.last_repair_attempts_used = 0
        self.last_payload = None

    def _mark_failed_no_payload(self) -> None:
        self._mark_failed_status("failed_no_payload")

    def _mark_failed_unparseable(
        self,
        *,
        current_issues: list[ValidationIssue],
        repair_records: list[dict[str, Any]],
        validation_rounds: list[dict[str, Any]],
    ) -> None:
        self._mark_failed_status("failed_unparseable_payload")
        self.last_validation_issues = issues_to_dicts(current_issues)
        self.last_unresolved_errors = issues_to_dicts(blocking_issues(current_issues))
        self.last_warnings = issues_to_dicts(warning_issues(current_issues))
        self.last_repair_attempts_used = len(repair_records)
        if self.last_response_record is not None:
            self.last_response_record.update(
                {
                    "repair_attempted": bool(repair_records),
                    "repair_attempts_used": len(repair_records),
                    "repair_attempts": repair_records,
                    "validation_rounds": validation_rounds,
                    "final_validation_issues": issues_to_dicts(current_issues),
                    "validation_status": "failed_unparseable_payload",
                    "unresolved_errors": issues_to_dicts(blocking_issues(current_issues)),
                    "warnings": issues_to_dicts(warning_issues(current_issues)),
                    "generation_allowed": False,
                }
            )

    def _mark_failed_status(self, status: str) -> None:
        self.last_validation_status = status
        self.last_generation_allowed = False
        if self.last_response_record is None:
            self.last_response_record = {
                "initial_response": None,
                "validation_issues": [],
                "validation_errors": [],
                "warnings": [],
                "repair_attempted": False,
                "repair_attempts_used": 0,
                "validation_rounds": [],
                "repair_diff": [],
                "validation_status": status,
                "unresolved_errors": [],
                "generation_allowed": False,
            }
        else:
            self.last_response_record.update({"validation_status": status, "generation_allowed": False})

    def _warn_unresolved_issues(self, issues: list[ValidationIssue], *, repair_attempts_used: int) -> None:
        summary = (
            f"[llm_direct][WARN] Continuing with unresolved validation issues after "
            f"{repair_attempts_used} repair attempt(s)."
        )
        print(summary, file=sys.stderr)
        for issue in issues[:12]:
            print(f"  - {issue.path}: {issue.code}", file=sys.stderr)
        self._log(
            "llm_direct_best_effort_unresolved_issues",
            repair_attempts_used=repair_attempts_used,
            issues=issues_to_dicts(issues),
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
                "- scoring_prompt should be short, objective, and semantic, focusing on the main subject and action; avoid mood words, style words, long identity details, camera jargon, and subjective adjectives.\n"
                "- anchor_generation_prompt is for standard diffusion / Anchor / IP-Adapter and must not require StoryDiffusion tags.\n"
                "- anchor_generation_prompt is a story scene prompt, not a reference image prompt; include the character's stable visual identity from stable_identity plus the visible scene action, object/setting, and framing. Do not rely on the character name alone for identity.\n"
                "- anchor_generation_prompt should use the same broad visual medium as anchor_reference_prompt, preferably naturalistic/cinematic story image unless the story clearly requires another medium.\n"
                "- anchor_generation_prompt must not include reference-only constraints such as simple/plain/clean/centered background, single-subject, or one-person-in-image wording.\n"
                "- anchor_reference_prompt should show only the character, not a story scene: use neutral standing pose or simple half-body/full-body pose, plain/simple background, no story location, no temporary prop, no scene action, and no other people.\n"
                "- anchor_reference_prompt is for standard diffusion / Anchor Bank, so it must not contain StoryDiffusion bracket tags such as [Ben], [Character], or [NC].\n"
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
            "Stateful Visual Prompt Planning:\n"
            "- Before writing final prompts, infer visual_continuity_anchors, one scene_visual_plan per scene, scene_change_level, and action_critical.\n"
            "- Since every panel is generated independently with text2img, final prompts must be self-contained; encode both continuity and visual progression directly in each final prompt.\n"
            "- visual_continuity_anchors capture persistent settings/objects/tasks or evolving visual states. Use prompt_phrase for persistent anchors and state_by_scene for evolving states. Every applicable anchor must appear in final scene prompts, and do not output anchors that are unused.\n"
            "- Do not carry location-specific objects across a clear location change. Character identity/clothing can persist across locations.\n"
            "- scene_visual_plan.visual_action must turn abstract verbs into visible actions. action_visibility_cue must say what visible evidence proves the action. camera_framing must be one of wide shot, medium-wide shot, medium shot, medium close-up shot, close-up shot.\n"
            "- Final anchor_generation_prompt and storydiffusion_prompt must already include applicable continuity anchors, visual_action, action_visibility_cue, and camera_framing. Local code will not combine these fields for you.\n"
            "- scene_change_level describes how much the expected image differs from the previous panel: small for minor pose/action variation, medium for visible pose/interaction change in same context, large for major pose/action/location/composition change.\n"
            "- action_critical is true when scene success depends on a specific visible pose, movement, object interaction, or spatial relation rather than only the subject identity.\n"
            "- action_prompt is required and must be a short visible-action phrase. For action-critical scenes it must distinguish success from near-miss candidates.\n"
            "- scoring_prompt is required and must be short but concrete; include the key action and important visible evidence, not only the subject and nearby setting.\n"
            "Rules:\n"
            "- Do not rely on local code to add hair, outfit, species, setting, action, framing, emotion, background, or reference constraints.\n"
            "- subject_type is your judgment and must be one of human, animal, robot, object, vehicle, unknown.\n"
            "- If subject_type is unknown, stable_identity and reference prompts must still be visually concrete and drawable.\n"
            "- stable_identity is short stable visual identity text for inspection and general prompt reasoning.\n"
            "- stable_identity, general_prompt, anchor_reference_prompt, and identity_reference_prompts must describe only stable visual identity. They must not include scene-specific locations, temporary props, actions, story events, personality, preferences, habits, or narrative background.\n"
            "- anchor_reference_prompt is a final executable identity/reference prompt for Anchor Bank; do not include bracket tags, and include single-subject and simple/centered-background reference constraints yourself.\n"
            "- anchor_reference_prompt is identity-only. Avoid scene-specific poses, locations, props, or task equipment unless they are part of the character's stable visual identity.\n"
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
        scene_visual_plan_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["visual_action", "action_visibility_cue", "camera_framing"],
            "properties": {
                "visual_action": {"type": "string"},
                "action_visibility_cue": {"type": "string"},
                "camera_framing": {
                    "type": "string",
                    "enum": ["wide shot", "medium-wide shot", "medium shot", "medium close-up shot", "close-up shot"],
                },
            },
        }
        character_properties: dict[str, Any] = {
            "character_id": {"type": "string"},
            "subject_type": {"type": "string"},
            "stable_identity": {"type": "string"},
        }
        character_required = ["character_id", "subject_type", "stable_identity"]
        if "anchor" in self.targets:
            character_properties["anchor_reference_prompt"] = {"type": "string"}
            character_required.append("anchor_reference_prompt")

        scene_properties: dict[str, Any] = {
            "scene_id": {"type": "string"},
            "scene_visual_plan": scene_visual_plan_schema,
            "scene_change_level": {"type": "string", "enum": ["small", "medium", "large"]},
            "action_critical": {"type": "boolean"},
            "action_prompt": {"type": "string"},
            "scoring_prompt": {"type": "string"},
        }
        scene_required = [
            "scene_id",
            "scene_visual_plan",
            "scene_change_level",
            "action_critical",
            "action_prompt",
            "scoring_prompt",
        ]
        if "anchor" in self.targets:
            scene_properties.update(
                {
                    "visible_character_ids": {"type": "array", "items": {"type": "string"}},
                    "identity_conditioning_subject_id": string_or_null,
                    "anchor_generation_prompt": {"type": "string"},
                }
            )
            scene_required.extend(
                [
                    "visible_character_ids",
                    "identity_conditioning_subject_id",
                    "anchor_generation_prompt",
                ]
            )
        if "storydiffusion" in self.targets:
            scene_properties["storydiffusion_prompt"] = {"type": "string"}
            scene_required.append("storydiffusion_prompt")

        top_properties: dict[str, Any] = {
            "target_backends": {"type": "array", "items": {"type": "string"}},
            "visual_continuity_anchors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["anchor_id", "type", "applies_to_scene_ids", "prompt_phrase", "state_by_scene"],
                    "properties": {
                        "anchor_id": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "persistent_setting",
                                "persistent_object",
                                "persistent_task_and_setting",
                                "evolving_visual_state",
                                "weather_or_lighting_state",
                                "vehicle_or_transport_context",
                            ],
                        },
                        "applies_to_scene_ids": {"type": "array", "items": {"type": "string"}},
                        "prompt_phrase": {"type": "string"},
                        "state_by_scene": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                },
            },
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
        top_required = ["target_backends", "visual_continuity_anchors", "characters", "scenes", "notes"]
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
