from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from storygen.native_prompting.types import ALLOWED_SUBJECT_TYPES, ALLOWED_TARGET_BACKENDS, NativePromptPayload
from storygen.types import Scene, Story


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
NARRATIVE_HABIT_TERMS = ("often", "usually", "enjoys", "likes", "drawn to", "loves to", "prefers")
MOOD_WORDS = ("thoughtful", "serene", "cozy", "calm", "joyful", "lost in thought", "contemplating", "peaceful")
UNSAFE_NEGATIVE_TERMS = ("multiple subjects", "multiple people", "crowd", "traffic", "vehicles", "complex background")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "the",
    "their",
    "they",
    "to",
    "with",
}


@dataclass(slots=True)
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str
    suspicious_text: str = ""
    instruction: str = ""

    def __str__(self) -> str:
        return self.message

    def __contains__(self, value: str) -> bool:
        return value in self.message

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "suspicious_text": self.suspicious_text,
            "instruction": self.instruction,
        }


@dataclass(slots=True)
class NativePromptValidationError(Exception):
    errors: list[ValidationIssue | str] = field(default_factory=list)

    def __str__(self) -> str:
        return "; ".join(str(error) for error in self.errors)


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
) -> list[ValidationIssue]:
    validation_config = config or NativePromptValidationConfig()
    requested_targets = normalize_targets(targets or payload.target_backends)
    issues: list[ValidationIssue] = []
    story_entities = list(story.all_entities or [])
    known_entities = set(story_entities)
    character_ids = [character.character_id for character in payload.characters]
    character_id_set = set(character_ids)
    if not known_entities:
        known_entities = set(character_ids)

    for target in requested_targets:
        if target not in ALLOWED_TARGET_BACKENDS:
            _add_issue(
                issues,
                "hard_error",
                "unknown_target_backend",
                "$.target_backends",
                f"Unknown target_backends value: {target}",
                target,
                "Use only supported target backends: anchor, storydiffusion.",
            )
    payload_targets = normalize_targets(payload.target_backends)
    for target in requested_targets:
        if target not in payload_targets:
            _add_issue(
                issues,
                "hard_error",
                "missing_requested_target",
                "$.target_backends",
                f"Payload target_backends missing requested target: {target}",
                target,
                "Return target_backends containing every requested backend.",
            )

    if story_entities:
        for entity in story_entities:
            if entity not in character_id_set:
                _add_issue(
                    issues,
                    "hard_error",
                    "missing_character",
                    "$.characters",
                    f"Missing character for story entity: {entity}",
                    entity,
                    "Include one characters[] entry for every tagged story entity.",
                )
    duplicate_characters = sorted({character_id for character_id in character_ids if character_ids.count(character_id) > 1})
    for character_id in duplicate_characters:
        _add_issue(
            issues,
            "hard_error",
            "duplicate_character",
            "$.characters",
            f"Duplicate character id: {character_id}",
            character_id,
            "Use each character_id at most once.",
        )

    story_scene_tokens = _story_scene_tokens(story)
    for index, character in enumerate(payload.characters):
        character_path = f"$.characters[{index}]"
        if not character.character_id:
            _add_issue(issues, "hard_error", "empty_character_id", f"{character_path}.character_id", "Character has empty character_id")
        elif known_entities and character.character_id not in known_entities:
            _add_issue(
                issues,
                "hard_error",
                "unknown_character_id",
                f"{character_path}.character_id",
                f"Unknown character_id: {character.character_id}",
                character.character_id,
                "Use exact character ids from the tagged story entities.",
            )
        if character.subject_type not in ALLOWED_SUBJECT_TYPES:
            _add_issue(
                issues,
                "hard_error",
                "invalid_subject_type",
                f"{character_path}.subject_type",
                f"Invalid subject_type for {character.character_id}: {character.subject_type}",
                character.subject_type,
                "Use one of: human, animal, robot, object, vehicle, unknown.",
            )
        _validate_text_field(
            issues,
            f"{character_path}.stable_identity",
            character.stable_identity,
            max_words=validation_config.max_identity_prompt_words,
            max_chars=validation_config.max_identity_prompt_chars,
        )
        _validate_identity_boundary(
            issues,
            f"{character_path}.stable_identity",
            character.stable_identity,
            story_scene_tokens=story_scene_tokens,
            require_clean_reference=False,
        )
        if character.subject_type == "unknown":
            _validate_unknown_identity(
                issues,
                f"{character_path}.stable_identity",
                character.stable_identity,
                character.character_id,
                min_words=validation_config.min_unknown_identity_words,
            )
        if "anchor" in requested_targets:
            _validate_text_field(
                issues,
                f"{character_path}.anchor_reference_prompt",
                character.anchor_reference_prompt,
                max_words=validation_config.max_identity_prompt_words,
                max_chars=validation_config.max_identity_prompt_chars,
            )
            _validate_anchor_reference_prompt(
                issues,
                f"{character_path}.anchor_reference_prompt",
                character.anchor_reference_prompt,
                story_scene_tokens=story_scene_tokens,
            )
            if character.subject_type == "unknown":
                _validate_unknown_identity(
                    issues,
                    f"{character_path}.anchor_reference_prompt",
                    character.anchor_reference_prompt,
                    character.character_id,
                    min_words=validation_config.min_unknown_identity_words,
                )

    expected_scene_ids = [scene.scene_id for scene in story.scenes]
    actual_scene_ids = [scene.scene_id for scene in payload.scenes]
    if actual_scene_ids != expected_scene_ids:
        _add_issue(
            issues,
            "hard_error",
            "scene_id_mismatch",
            "$.scenes",
            f"Scene ids do not match parsed story: expected {expected_scene_ids}, got {actual_scene_ids}",
            str(actual_scene_ids),
            "Return exactly one scene object per parsed story scene in the same order.",
        )
    scene_by_id = {scene.scene_id: scene for scene in payload.scenes}
    parsed_scene_by_id = {scene.scene_id: scene for scene in story.scenes}
    for scene_index, scene_id in enumerate(expected_scene_ids):
        scene = scene_by_id.get(scene_id)
        parsed_scene = parsed_scene_by_id[scene_id]
        if scene is None:
            continue
        scene_path = f"$.scenes[{scene_index}]"
        pronoun_participant = _resolve_leading_pronoun_entity(story, scene_index, parsed_scene)
        interaction_entities = [entity for entity in [pronoun_participant, *parsed_scene.entities] if entity]
        if "anchor" in requested_targets:
            _validate_text_field(
                issues,
                f"{scene_path}.anchor_generation_prompt",
                scene.anchor_generation_prompt,
                max_words=validation_config.max_scene_prompt_words,
                max_chars=validation_config.max_scene_prompt_chars,
            )
            _validate_scene_prompt_not_reference_only(
                issues,
                f"{scene_path}.anchor_generation_prompt",
                scene.anchor_generation_prompt,
            )
            _validate_text_field(
                issues,
                f"{scene_path}.scoring_prompt",
                scene.scoring_prompt,
                max_words=20,
                max_chars=180,
            )
            _validate_scoring_prompt(issues, f"{scene_path}.scoring_prompt", scene.scoring_prompt)
            _validate_visible_ids(issues, scene_id, f"{scene_path}.visible_character_ids", scene.visible_character_ids, known_entities)
            _validate_identity_conditioning_id(
                issues,
                scene_id,
                f"{scene_path}.identity_conditioning_subject_id",
                scene.identity_conditioning_subject_id,
                known_entities,
            )
            for entity in parsed_scene.entities:
                if entity not in scene.visible_character_ids:
                    _add_issue(
                        issues,
                        "hard_error",
                        "visible_ids_missing_explicit_entity",
                        f"{scene_path}.visible_character_ids",
                        f"scenes[{scene_id}].visible_character_ids missing explicit entity {entity}",
                        entity,
                        "Include every explicitly tagged entity that is visible in the scene.",
                    )
            if _is_pronoun_only_scene(parsed_scene.clean_text, parsed_scene.entities) and not scene.visible_character_ids:
                _add_issue(
                    issues,
                    "hard_error",
                    "visible_ids_missing_pronoun_resolution",
                    f"{scene_path}.visible_character_ids",
                    f"scenes[{scene_id}].visible_character_ids must resolve pronoun-only visual scene",
                    parsed_scene.clean_text,
                    "Resolve the pronoun to known recurring character ids when visually present.",
                )
            for entity in interaction_entities:
                if entity not in scene.visible_character_ids:
                    _add_issue(
                        issues,
                        "repair_error",
                        "interaction_visible_participant_missing",
                        f"{scene_path}.visible_character_ids",
                        f"scenes[{scene_id}] omits interaction participant {entity}",
                        parsed_scene.clean_text,
                        "Resolve pronouns in interaction scenes and include all visible participants.",
                    )
            if len(scene.visible_character_ids) > 1 and scene.identity_conditioning_subject_id is not None:
                _add_issue(
                    issues,
                    "repair_error",
                    "dual_scene_identity_conditioning_should_be_null",
                    f"{scene_path}.identity_conditioning_subject_id",
                    f"scenes[{scene_id}].identity_conditioning_subject_id should usually be null for dual-person scenes",
                    str(scene.identity_conditioning_subject_id),
                    "Set identity_conditioning_subject_id to null when multiple characters are equally visible.",
                )
        if "storydiffusion" in requested_targets:
            _validate_text_field(
                issues,
                f"{scene_path}.storydiffusion_prompt",
                scene.storydiffusion_prompt,
                max_words=validation_config.max_scene_prompt_words,
                max_chars=validation_config.max_scene_prompt_chars,
            )
            _validate_scene_prompt_not_reference_only(
                issues,
                f"{scene_path}.storydiffusion_prompt",
                scene.storydiffusion_prompt,
            )
            _validate_storydiffusion_scene_detail(
                issues,
                scene_id,
                f"{scene_path}.storydiffusion_prompt",
                scene.storydiffusion_prompt,
                min_words=validation_config.min_storydiffusion_scene_words,
            )
            _validate_storydiffusion_scene_prompt(
                issues,
                scene_id,
                f"{scene_path}.storydiffusion_prompt",
                scene.storydiffusion_prompt,
                parsed_scene.entities,
                known_entities,
            )
            tags = TAG_PATTERN.findall(scene.storydiffusion_prompt)
            if _is_pronoun_only_scene(parsed_scene.clean_text, parsed_scene.entities) and not scene.storydiffusion_prompt.startswith("[NC]"):
                if not any(tag in known_entities for tag in tags):
                    _add_issue(
                        issues,
                        "hard_error",
                        "storydiffusion_pronoun_scene_missing_known_tag",
                        f"{scene_path}.storydiffusion_prompt",
                        f"scenes[{scene_id}].storydiffusion_prompt must resolve pronoun-only scene to a known tag or [NC]",
                        scene.storydiffusion_prompt,
                        "Use the resolved character tag for pronoun-only visual scenes, or [NC] only for a true cutaway.",
                    )
            for entity in interaction_entities:
                if not scene.storydiffusion_prompt.startswith("[NC]") and entity not in tags:
                    _add_issue(
                        issues,
                        "repair_error",
                        "storydiffusion_interaction_tag_missing",
                        f"{scene_path}.storydiffusion_prompt",
                        f"scenes[{scene_id}].storydiffusion_prompt omits interaction participant [{entity}]",
                        parsed_scene.clean_text,
                        "Resolve pronouns and include exact tags for all visible interaction participants.",
                    )
            if (
                len(story_entities) == 1
                and not parsed_scene.entities
                and not _is_pronoun_only_scene(parsed_scene.clean_text, parsed_scene.entities)
                and scene.storydiffusion_prompt.startswith("[NC]")
            ):
                _add_issue(
                    issues,
                    "repair_error",
                    "nc_possible_recurring_subject",
                    f"{scene_path}.storydiffusion_prompt",
                    f"scenes[{scene_id}] uses [NC] in an ongoing single-character story",
                    scene.storydiffusion_prompt,
                    "Include the recurring character if they can naturally observe or experience the environment; use [NC] only for a true cutaway.",
                )
            _validate_scene_prompt_mood_warning(issues, f"{scene_path}.storydiffusion_prompt", scene.storydiffusion_prompt)

    if "storydiffusion" in requested_targets:
        storydiffusion = payload.storydiffusion
        _validate_text_field(
            issues,
            "$.storydiffusion.general_prompt",
            storydiffusion.general_prompt,
            max_words=max(validation_config.max_identity_prompt_words, len(payload.characters) * 24),
            max_chars=max(validation_config.max_identity_prompt_chars, len(payload.characters) * 180),
        )
        _validate_storydiffusion_general_prompt_tags(issues, storydiffusion.general_prompt, character_ids)
        _validate_identity_boundary(
            issues,
            "$.storydiffusion.general_prompt",
            storydiffusion.general_prompt,
            story_scene_tokens=story_scene_tokens,
            require_clean_reference=False,
        )
        if not storydiffusion.identity_reference_prompts:
            _add_issue(
                issues,
                "hard_error",
                "missing_storydiffusion_identity_reference_prompts",
                "$.storydiffusion.identity_reference_prompts",
                "storydiffusion.identity_reference_prompts is required",
                "",
                "Return at least one identity reference prompt for each character.",
            )
        for index, prompt in enumerate(storydiffusion.identity_reference_prompts):
            path = f"$.storydiffusion.identity_reference_prompts[{index}]"
            _validate_text_field(
                issues,
                path,
                prompt,
                max_words=validation_config.max_identity_prompt_words,
                max_chars=validation_config.max_identity_prompt_chars,
            )
            _validate_identity_reference_prompt(issues, path, prompt)
            _validate_identity_boundary(issues, path, prompt, story_scene_tokens=story_scene_tokens, require_clean_reference=False)
            _validate_storydiffusion_identity_reference_tags(issues, path, prompt, known_entities)
        if storydiffusion.identity_prompts_per_character < 1:
            _add_issue(
                issues,
                "hard_error",
                "invalid_identity_prompts_per_character",
                "$.storydiffusion.identity_prompts_per_character",
                "storydiffusion.identity_prompts_per_character must be >= 1",
                str(storydiffusion.identity_prompts_per_character),
                "Use a positive integer identity_prompts_per_character.",
            )
        expected_identity_count = len(payload.characters) * storydiffusion.identity_prompts_per_character
        if storydiffusion.identity_reference_prompts and len(storydiffusion.identity_reference_prompts) != expected_identity_count:
            _add_issue(
                issues,
                "hard_error",
                "storydiffusion_identity_prompt_count_mismatch",
                "$.storydiffusion.identity_reference_prompts",
                "storydiffusion.identity_reference_prompts count must equal "
                f"character_count * identity_prompts_per_character: expected {expected_identity_count}, "
                f"got {len(storydiffusion.identity_reference_prompts)}",
                str(len(storydiffusion.identity_reference_prompts)),
                "Return exactly character_count * identity_prompts_per_character identity reference prompts.",
            )
        _validate_scene_negative_prompt(issues, storydiffusion.scene_negative_prompt_extra, story)
    return issues


def raise_if_invalid(issues: list[ValidationIssue]) -> None:
    blocking = [issue for issue in issues if issue.severity != "warning"]
    if blocking:
        raise NativePromptValidationError(blocking)


def issues_to_dicts(issues: Iterable[ValidationIssue]) -> list[dict[str, Any]]:
    return [issue.to_dict() for issue in issues]


def issue_messages(issues: Iterable[ValidationIssue]) -> list[str]:
    return [issue.message for issue in issues]


def blocking_issues(issues: Iterable[ValidationIssue]) -> list[ValidationIssue]:
    return [issue for issue in issues if issue.severity in {"hard_error", "repair_error"}]


def hard_issues(issues: Iterable[ValidationIssue]) -> list[ValidationIssue]:
    return [issue for issue in issues if issue.severity == "hard_error"]


def repair_issues(issues: Iterable[ValidationIssue]) -> list[ValidationIssue]:
    return [issue for issue in issues if issue.severity == "repair_error"]


def warning_issues(issues: Iterable[ValidationIssue]) -> list[ValidationIssue]:
    return [issue for issue in issues if issue.severity == "warning"]


def validation_metadata(issues: list[ValidationIssue]) -> dict[str, Any]:
    return {
        "issue_count": len(issues),
        "issues": issues_to_dicts(issues),
        "errors": issue_messages([issue for issue in issues if issue.severity != "warning"]),
        "warnings": issues_to_dicts(warning_issues(issues)),
    }


def _add_issue(
    issues: list[ValidationIssue],
    severity: str,
    code: str,
    path: str,
    message: str,
    suspicious_text: str = "",
    instruction: str = "",
) -> None:
    issues.append(
        ValidationIssue(
            severity=severity,
            code=code,
            path=path,
            message=message,
            suspicious_text=suspicious_text,
            instruction=instruction,
        )
    )


def _validate_text_field(
    issues: list[ValidationIssue],
    field_name: str,
    value: str,
    *,
    max_words: int,
    max_chars: int,
) -> None:
    text = str(value or "").strip()
    if not text:
        _add_issue(
            issues,
            "hard_error",
            "missing_required_field",
            field_name,
            f"{field_name} is required",
            "",
            "Return a non-empty final value for this target-required field.",
        )
        return
    if len(text) > max_chars:
        _add_issue(issues, "hard_error", "field_too_long_chars", field_name, f"{field_name} exceeds {max_chars} chars", text)
    if len(_words(text)) > max_words:
        _add_issue(issues, "hard_error", "field_too_long_words", field_name, f"{field_name} exceeds {max_words} words", text)
    lowered = text.lower()
    for placeholder in BAD_PLACEHOLDERS:
        if placeholder in lowered:
            _add_issue(
                issues,
                "hard_error",
                "placeholder_text",
                field_name,
                f"{field_name} contains {placeholder}",
                placeholder,
                "Replace placeholder wording with concrete visual content.",
            )


def _validate_visible_ids(
    issues: list[ValidationIssue],
    scene_id: str,
    path: str,
    values: list[str],
    known_entities: set[str],
) -> None:
    if not isinstance(values, list):
        _add_issue(issues, "hard_error", "visible_ids_not_list", path, f"scenes[{scene_id}].visible_character_ids must be a list")
        return
    for value in values:
        if value not in known_entities:
            _add_issue(
                issues,
                "hard_error",
                "visible_ids_unknown_entity",
                path,
                f"scenes[{scene_id}].visible_character_ids contains unknown entity {value}",
                value,
                "Use exact character ids from story entities.",
            )


def _validate_identity_conditioning_id(
    issues: list[ValidationIssue],
    scene_id: str,
    path: str,
    value: str | None,
    known_entities: set[str],
) -> None:
    if value is not None and value not in known_entities:
        _add_issue(
            issues,
            "hard_error",
            "identity_conditioning_unknown_entity",
            path,
            f"scenes[{scene_id}].identity_conditioning_subject_id contains unknown entity {value}",
            str(value),
            "Use a known character id or null.",
        )


def _validate_anchor_reference_prompt(
    issues: list[ValidationIssue],
    path: str,
    prompt: str,
    *,
    story_scene_tokens: set[str],
) -> None:
    _validate_identity_reference_prompt(issues, path, prompt)
    _validate_identity_boundary(issues, path, prompt, story_scene_tokens=story_scene_tokens, require_clean_reference=True)
    lowered = prompt.lower()
    if not any(term in lowered for term in SINGLE_SUBJECT_TERMS):
        _add_issue(
            issues,
            "repair_error",
            "anchor_reference_missing_single_subject",
            path,
            f"{path} must request a single subject",
            prompt,
            'Rewrite as a clean single-character identity reference prompt and include wording such as "single subject" or "one person only".',
        )
    if not any(term in lowered for term in REFERENCE_FRAMING_TERMS):
        _add_issue(
            issues,
            "repair_error",
            "anchor_reference_missing_clean_background",
            path,
            f"{path} must include simple/centered reference framing",
            prompt,
            "Use plain/simple background and centered neutral identity framing.",
        )


def _validate_identity_reference_prompt(issues: list[ValidationIssue], field_name: str, prompt: str) -> None:
    lowered = prompt.lower()
    for term in IDENTITY_BANNED_TERMS:
        if term in lowered:
            _add_issue(
                issues,
                "repair_error",
                "identity_reference_banned_layout",
                field_name,
                f"{field_name} contains banned identity-reference term: {term}",
                term,
                "Remove character-sheet, turnaround, multiple-view, and duplicate-subject wording.",
            )


def _validate_identity_boundary(
    issues: list[ValidationIssue],
    path: str,
    value: str,
    *,
    story_scene_tokens: set[str],
    require_clean_reference: bool,
) -> None:
    lowered = str(value or "").lower()
    for term in NARRATIVE_HABIT_TERMS:
        if term in lowered:
            _add_issue(
                issues,
                "repair_error",
                "identity_narrative_habit",
                path,
                f"{path} contains narrative habit/personality wording: {term}",
                term,
                "Rewrite this field as stable visual identity only.",
            )
    overlap = sorted(set(_content_words(value)) & story_scene_tokens)
    if len(overlap) >= 3:
        severity = "repair_error" if require_clean_reference else "warning"
        _add_issue(
            issues,
            severity,
            "identity_scene_token_overlap",
            path,
            f"{path} overlaps with scene-specific story tokens",
            ", ".join(overlap[:8]),
            "Remove story scene locations, temporary props, actions, and events from identity/reference fields.",
        )


def _validate_scene_prompt_not_reference_only(issues: list[ValidationIssue], field_name: str, prompt: str) -> None:
    lowered = prompt.lower()
    for term in REFERENCE_ONLY_SCENE_TERMS:
        if term in lowered:
            _add_issue(
                issues,
                "repair_error",
                "scene_prompt_reference_only_phrase",
                field_name,
                f"{field_name} contains reference-only phrase: {term}",
                term,
                "Rewrite scene prompts as story scenes; keep reference-only constraints inside identity prompts.",
            )


def _validate_storydiffusion_scene_detail(
    issues: list[ValidationIssue],
    scene_id: str,
    path: str,
    prompt: str,
    *,
    min_words: int,
) -> None:
    if prompt.strip().startswith("[NC]"):
        return
    text = TAG_PATTERN.sub(" ", prompt)
    words = _words(text)
    if len(words) < min_words:
        _add_issue(
            issues,
            "repair_error",
            "storydiffusion_scene_prompt_too_short",
            path,
            f"scenes[{scene_id}].storydiffusion_prompt is too short for a visual frame prompt; expected at least {min_words} words after tags",
            prompt,
            "Add concrete visible action, object/setting, spatial relation, or framing.",
        )


def _validate_storydiffusion_general_prompt_tags(issues: list[ValidationIssue], prompt: str, character_ids: list[str]) -> None:
    tags = set(TAG_PATTERN.findall(prompt))
    for character_id in character_ids:
        if character_id not in tags:
            _add_issue(
                issues,
                "hard_error",
                "storydiffusion_general_missing_tag",
                "$.storydiffusion.general_prompt",
                f"storydiffusion.general_prompt missing tag [{character_id}]",
                character_id,
                "Include each character with exact [Character] tag in general_prompt.",
            )


def _validate_storydiffusion_identity_reference_tags(
    issues: list[ValidationIssue],
    field_name: str,
    prompt: str,
    known_entities: set[str],
) -> None:
    prompt = prompt.strip()
    tags = TAG_PATTERN.findall(prompt)
    if not tags:
        _add_issue(
            issues,
            "hard_error",
            "storydiffusion_identity_reference_missing_tag",
            field_name,
            f"{field_name} must include an exact [Character] tag",
            prompt,
            "Prefix each StoryDiffusion identity reference row with the exact character tag.",
        )
        return
    for tag in tags:
        if tag not in known_entities:
            _add_issue(
                issues,
                "hard_error",
                "storydiffusion_identity_reference_unknown_tag",
                field_name,
                f"{field_name} contains unknown tag [{tag}]",
                tag,
                "Use only exact character tags from story entities.",
            )


def _validate_storydiffusion_scene_prompt(
    issues: list[ValidationIssue],
    scene_id: str,
    path: str,
    prompt: str,
    explicit_entities: list[str],
    known_entities: set[str],
) -> None:
    prompt = prompt.strip()
    if not prompt:
        return
    if not (prompt.startswith("[NC]") or re.match(r"^(?:\[[^\[\]]+\]\s*)+", prompt)):
        _add_issue(
            issues,
            "hard_error",
            "storydiffusion_scene_prompt_bad_prefix",
            path,
            f"scenes[{scene_id}].storydiffusion_prompt must start with [Character] tags or [NC]",
            prompt,
            "Start StoryDiffusion scene prompts with exact [Character] tags or [NC].",
        )
        return
    tags = TAG_PATTERN.findall(prompt)
    for tag in tags:
        if tag != "NC" and tag not in known_entities:
            _add_issue(
                issues,
                "hard_error",
                "storydiffusion_scene_prompt_unknown_tag",
                path,
                f"scenes[{scene_id}].storydiffusion_prompt contains unknown tag [{tag}]",
                tag,
                "Use only exact character tags from story entities.",
            )
    if explicit_entities:
        for entity in explicit_entities:
            if entity not in tags:
                _add_issue(
                    issues,
                    "hard_error",
                    "storydiffusion_scene_prompt_missing_explicit_tag",
                    path,
                    f"scenes[{scene_id}].storydiffusion_prompt missing explicit tag [{entity}]",
                    entity,
                    "Include every explicitly tagged visible character in the StoryDiffusion scene prompt.",
                )


def _validate_unknown_identity(
    issues: list[ValidationIssue],
    field_name: str,
    value: str,
    character_id: str,
    *,
    min_words: int,
) -> None:
    words = _words(value)
    if len(words) < min_words:
        _add_issue(
            issues,
            "hard_error",
            "unknown_subject_identity_too_short",
            field_name,
            f"{field_name} is too short for subject_type=unknown",
            value,
            "For subject_type=unknown, provide a concrete drawable visual identity.",
        )
    stripped = TAG_PATTERN.sub("", value).strip(" ,.;")
    if stripped.lower() == character_id.lower() or not stripped:
        _add_issue(
            issues,
            "hard_error",
            "unknown_subject_identity_only_name",
            field_name,
            f"{field_name} is only a tag or character name",
            value,
            "For subject_type=unknown, describe visible shape, color, material, or other stable traits.",
        )


def _validate_scoring_prompt(issues: list[ValidationIssue], path: str, prompt: str) -> None:
    lowered = prompt.lower()
    for term in MOOD_WORDS:
        if term in lowered:
            _add_issue(
                issues,
                "warning",
                "scoring_prompt_subjective_mood",
                path,
                f"{path} contains subjective mood wording: {term}",
                term,
                "Keep scoring_prompt short, objective, and focused on subject/action.",
            )


def _validate_scene_prompt_mood_warning(issues: list[ValidationIssue], path: str, prompt: str) -> None:
    lowered = prompt.lower()
    for term in MOOD_WORDS:
        if term in lowered:
            _add_issue(
                issues,
                "warning",
                "scene_prompt_mood_heavy",
                path,
                f"{path} contains mood or inner-state wording: {term}",
                term,
                "Prefer concrete visible pose/action/object details over literary mood.",
            )
            return


def _validate_scene_negative_prompt(issues: list[ValidationIssue], prompt: str, story: Story) -> None:
    if not prompt:
        return
    lowered = prompt.lower()
    story_text = story.raw_text.lower()
    requires_complex_content = (
        len(story.all_entities) > 1
        or any(term in story_text for term in ("crowd", "traffic", "street", "cafe", "exhibition", "bus", "car", "train"))
    )
    if not requires_complex_content:
        return
    for term in UNSAFE_NEGATIVE_TERMS:
        if term in lowered:
            _add_issue(
                issues,
                "warning",
                "scene_negative_prompt_may_suppress_story_content",
                "$.storydiffusion.scene_negative_prompt_extra",
                f"scene negative prompt may suppress required story content: {term}",
                term,
                "Do not forbid multiple subjects, crowds, traffic, vehicles, props, or complex locations required by the story.",
            )


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_-]+", str(value or ""))


def _content_words(value: str) -> list[str]:
    return [word.lower() for word in _words(value) if len(word) > 2 and word.lower() not in STOPWORDS]


def _story_scene_tokens(story: Story) -> set[str]:
    entity_tokens = {token.lower() for entity in story.all_entities for token in _words(entity)}
    return {
        token
        for scene in story.scenes
        for token in _content_words(scene.clean_text)
        if token not in entity_tokens
    }


def _is_pronoun_only_scene(clean_text: str, explicit_entities: list[str]) -> bool:
    return not explicit_entities and bool(LEADING_PRONOUN_PATTERN.search(str(clean_text or "").strip()))


def _resolve_leading_pronoun_entity(story: Story, scene_index: int, scene: Scene) -> str | None:
    clean_text = str(scene.clean_text or "").strip()
    if not scene.entities or not LEADING_PRONOUN_PATTERN.search(clean_text):
        return None
    prior_entities: list[str] = []
    for prior_scene in story.scenes[:scene_index]:
        for entity in prior_scene.entities:
            if entity not in prior_entities and entity not in scene.entities:
                prior_entities.append(entity)
    return prior_entities[0] if len(prior_entities) == 1 else None
