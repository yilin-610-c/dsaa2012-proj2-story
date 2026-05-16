from __future__ import annotations

import json

import pytest

from storygen.llm_client import BaseLLMClient, LLMResponse
from storygen.anchor_bank import build_anchor_prompt
from storygen.config import resolve_config
from storygen.native_prompting import LLMDirectPromptBuilder, build_anchor_prompt_bundle, build_storydiffusion_prompt_payload
from storygen.native_prompting.types import NativePromptPayload
from storygen.native_prompting.validator import NativePromptValidationError, validate_native_prompt_payload
from storygen.types import Scene, Story


class FakeLLMClient(BaseLLMClient):
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate_structured(self, *, messages, json_schema):
        response = self.responses[self.calls]
        self.calls += 1
        return LLMResponse(raw_text=json.dumps(response), parsed_json=response, metadata={"fake_call": self.calls})


class FailingLLMClient(BaseLLMClient):
    def generate_structured(self, *, messages, json_schema):
        raise RuntimeError("quota exceeded")


class NonDictLLMClient(BaseLLMClient):
    def generate_structured(self, *, messages, json_schema):
        return LLMResponse(raw_text="[]", parsed_json=[], metadata={"fake_call": 1})


def _story_single() -> Story:
    return Story(
        source_path="01.txt",
        raw_text="[SCENE-1] <Ben> drives a car.\n[SEP]\n[SCENE-2] He stops.",
        scenes=[
            Scene("SCENE-1", 0, "<Ben> drives a car.", "Ben drives a car.", ["Ben"]),
            Scene("SCENE-2", 1, "He stops.", "He stops.", []),
        ],
        all_entities=["Ben"],
        recurring_entities=["Ben"],
        entity_to_scene_ids={"Ben": ["SCENE-1"]},
    )


def _story_double() -> Story:
    return Story(
        source_path="02.txt",
        raw_text="[SCENE-1] <Ben> meets <Sara>.\n[SEP]\n[SCENE-2] They talk.",
        scenes=[
            Scene("SCENE-1", 0, "<Ben> meets <Sara>.", "Ben meets Sara.", ["Ben", "Sara"]),
            Scene("SCENE-2", 1, "They talk.", "They talk.", []),
        ],
        all_entities=["Ben", "Sara"],
        recurring_entities=["Ben", "Sara"],
        entity_to_scene_ids={"Ben": ["SCENE-1"], "Sara": ["SCENE-1"]},
    )


def _payload(targets=("anchor", "storydiffusion")) -> dict:
    return {
        "target_backends": list(targets),
        "visual_continuity_anchors": [],
        "characters": [
            {
                "character_id": "Ben",
                "subject_type": "human",
                "stable_identity": "a young man with short brown hair and a blue jacket",
                "anchor_reference_prompt": "a single young man with short brown hair and a blue jacket, standing alone, centered, simple background, one person in the image",
            }
        ],
        "scenes": [
            {
                "scene_id": "SCENE-1",
                "visible_character_ids": ["Ben"],
                "identity_conditioning_subject_id": "Ben",
                "anchor_generation_prompt": "a young man driving a car along a quiet road, hands near the steering wheel, medium shot",
                "storydiffusion_prompt": "[Ben] driving a car along a quiet road, hands near the steering wheel, medium shot",
                "scoring_prompt": "Ben driving a car on a road",
                "action_prompt": "Ben driving with hands near the steering wheel",
                "scene_visual_plan": {
                    "visual_action": "Ben drives a car along a quiet road",
                    "action_visibility_cue": "hands near the steering wheel",
                    "camera_framing": "medium shot",
                },
                "scene_change_level": "large",
                "action_critical": True,
            },
            {
                "scene_id": "SCENE-2",
                "visible_character_ids": ["Ben"],
                "identity_conditioning_subject_id": "Ben",
                "anchor_generation_prompt": "a young man stopping the car beside a quiet road, medium shot",
                "storydiffusion_prompt": "[Ben] stopping the car beside a quiet road, medium shot",
                "scoring_prompt": "Ben stopping a car beside the road",
                "action_prompt": "Ben stopping the car beside the road",
                "scene_visual_plan": {
                    "visual_action": "Ben stops the car beside a quiet road",
                    "action_visibility_cue": "car stopped beside the road",
                    "camera_framing": "medium shot",
                },
                "scene_change_level": "medium",
                "action_critical": True,
            },
        ],
        "storydiffusion": {
            "general_prompt": "[Ben] a young man with short brown hair and a blue jacket",
            "identity_reference_prompts": [
                "[Ben] a single young man with short brown hair and a blue jacket, standing alone, centered, simple background, one person in the image"
            ],
            "identity_prompts_per_character": 1,
            "identity_negative_prompt_extra": "character sheet, multiple views",
            "scene_negative_prompt_extra": "",
        },
        "notes": {"identity_reasoning": "Ben is recurring.", "continuity_reasoning": "Ben remains visible.", "self_check": "ok"},
    }


def test_base_config_defaults_to_best_effort_one_repair() -> None:
    config = resolve_config("configs/base.yaml", "cloud_anchor_ipadapter_scene", overrides={"prompt.pipeline": "llm_direct"})
    llm_direct = config["prompt"]["llm_direct"]
    assert llm_direct["repair_attempts"] == 1
    assert llm_direct["validation_policy"] == "best_effort"
    assert llm_direct["allow_generation_with_boundary_errors"] is True


def test_anchor_adapter_copies_llm_prompt_fields_exactly() -> None:
    payload = NativePromptPayload.from_dict(_payload(("anchor",)))
    errors = validate_native_prompt_payload(payload, _story_single(), targets=["anchor"])
    assert errors == []

    bundle = build_anchor_prompt_bundle(payload, _story_single(), {"negative_prompt": "blurry"})
    first = bundle.scene_prompts["SCENE-1"]
    assert first.generation_prompt == payload.scenes[0].anchor_generation_prompt
    assert first.scoring_prompt == payload.scenes[0].scoring_prompt
    assert first.action_prompt == payload.scenes[0].action_prompt
    assert bundle.metadata["character_specs"]["Ben"]["anchor_reference_prompt"] == payload.characters[0].anchor_reference_prompt
    assert bundle.metadata["scene_route_hints"]["SCENE-1"]["identity_conditioning_subject_id"] == "Ben"
    assert bundle.metadata["scene_route_hints"]["SCENE-1"]["route_change_level"] == "large"
    assert bundle.metadata["scene_route_hints"]["SCENE-1"]["action_critical"] is True


def test_anchor_only_payload_does_not_require_storydiffusion_fields() -> None:
    raw = _payload(("anchor",))
    raw.pop("storydiffusion")
    for scene in raw["scenes"]:
        scene.pop("storydiffusion_prompt")
    payload = NativePromptPayload.from_dict(raw)
    assert validate_native_prompt_payload(payload, _story_single(), targets=["anchor"]) == []


def test_storydiffusion_only_payload_does_not_require_anchor_fields() -> None:
    raw = _payload(("storydiffusion",))
    for character in raw["characters"]:
        character.pop("anchor_reference_prompt")
    for scene in raw["scenes"]:
        scene.pop("anchor_generation_prompt")
        scene.pop("visible_character_ids")
        scene.pop("identity_conditioning_subject_id")
    payload = NativePromptPayload.from_dict(raw)
    assert validate_native_prompt_payload(payload, _story_single(), targets=["storydiffusion"]) == []


def test_json_schema_is_target_conditional() -> None:
    anchor_builder = LLMDirectPromptBuilder({"llm_direct": {"targets": ["anchor"]}})
    anchor_scene_props = anchor_builder._json_schema()["schema"]["properties"]["scenes"]["items"]["properties"]
    assert "anchor_generation_prompt" in anchor_scene_props
    assert "scoring_prompt" in anchor_scene_props
    assert "storydiffusion_prompt" not in anchor_scene_props
    assert "storydiffusion" not in anchor_builder._json_schema()["schema"]["properties"]

    storydiffusion_builder = LLMDirectPromptBuilder({"llm_direct": {"targets": ["storydiffusion"]}})
    storydiffusion_schema = storydiffusion_builder._json_schema()["schema"]
    storydiffusion_scene_props = storydiffusion_schema["properties"]["scenes"]["items"]["properties"]
    character_props = storydiffusion_schema["properties"]["characters"]["items"]["properties"]
    assert "storydiffusion_prompt" in storydiffusion_scene_props
    assert "anchor_generation_prompt" not in storydiffusion_scene_props
    assert "scoring_prompt" in storydiffusion_scene_props
    assert "action_prompt" in storydiffusion_scene_props
    assert "scene_visual_plan" in storydiffusion_scene_props
    assert "anchor_reference_prompt" not in character_props
    assert "storydiffusion" in storydiffusion_schema["properties"]


def test_anchor_bank_uses_llm_direct_anchor_reference_prompt_without_suffix() -> None:
    prompt = "a single young man with short brown hair, centered, simple background, one person in the image"
    assert build_anchor_prompt({"character_id": "Ben", "anchor_reference_prompt": prompt}, "half_body", "ignored suffix") == prompt


def test_storydiffusion_adapter_concatenates_identity_and_scene_prompts() -> None:
    payload = NativePromptPayload.from_dict(_payload(("storydiffusion",)))
    errors = validate_native_prompt_payload(payload, _story_single(), targets=["storydiffusion"])
    assert errors == []

    rendered = build_storydiffusion_prompt_payload(payload, _story_single())
    assert rendered["prompt_array"] == payload.storydiffusion.identity_reference_prompts + [
        scene.storydiffusion_prompt for scene in payload.scenes
    ]
    assert rendered["identity_prompt_count"] == 1
    assert rendered["identity_prompts_per_character"] == 1
    assert rendered["character_count"] == 1
    assert rendered["save_image_start_index"] == 1
    assert rendered["storydiffusion_id_length"] == 1


def test_storydiffusion_id_metadata_for_two_characters() -> None:
    raw = _payload(("storydiffusion",))
    raw["characters"].append(
        {
            "character_id": "Sara",
            "subject_type": "human",
            "stable_identity": "a young woman with long black hair and a green coat",
            "anchor_reference_prompt": "[Sara] a single young woman with long black hair and a green coat, standing alone, centered, simple background, one person in the image",
        }
    )
    raw["scenes"] = [
        {
            "scene_id": "SCENE-1",
            "visible_character_ids": ["Ben", "Sara"],
            "identity_conditioning_subject_id": None,
            "anchor_generation_prompt": "",
            "storydiffusion_prompt": "[Ben] [Sara] meet on a quiet street, both visible in a medium shot two-shot composition",
                "scoring_prompt": "Ben and Sara meeting on a quiet street, both visible",
                "action_prompt": "Ben and Sara meeting on a quiet street",
            "scene_visual_plan": {
                "visual_action": "Ben and Sara meet on a quiet street",
                "action_visibility_cue": "both visible in a medium two-shot",
                "camera_framing": "medium shot",
            },
            "scene_change_level": "large",
            "action_critical": False,
        },
        {
            "scene_id": "SCENE-2",
            "visible_character_ids": ["Ben", "Sara"],
            "identity_conditioning_subject_id": None,
            "anchor_generation_prompt": "",
            "storydiffusion_prompt": "[Ben] [Sara] talk together on the same quiet street, both visible in a medium shot two-shot composition",
                "scoring_prompt": "Ben and Sara talking together on the same quiet street",
                "action_prompt": "Ben and Sara talking together on the same quiet street",
            "scene_visual_plan": {
                "visual_action": "Ben and Sara talk together on the same quiet street",
                "action_visibility_cue": "both visible in a medium two-shot",
                "camera_framing": "medium shot",
            },
            "scene_change_level": "small",
            "action_critical": False,
        },
    ]
    raw["storydiffusion"]["general_prompt"] = (
        "[Ben] a young man with short brown hair and a blue jacket\n"
        "[Sara] a young woman with long black hair and a green coat"
    )
    raw["storydiffusion"]["identity_prompts_per_character"] = 2
    raw["storydiffusion"]["identity_reference_prompts"] = [
        "[Ben] a single young man with short brown hair and a blue jacket, standing alone, centered, simple background, one person in the image",
        "[Ben] a single young man with short brown hair and a blue jacket, neutral pose, centered, simple background, one person in the image",
        raw["characters"][1]["anchor_reference_prompt"],
        "[Sara] a single young woman with long black hair and a green coat, neutral pose, centered, simple background, one person in the image",
    ]
    payload = NativePromptPayload.from_dict(raw)
    assert validate_native_prompt_payload(payload, _story_double(), targets=["storydiffusion"]) == []
    rendered = build_storydiffusion_prompt_payload(payload, _story_double())
    assert rendered["identity_prompt_count"] == 4
    assert rendered["identity_prompts_per_character"] == 2
    assert rendered["character_count"] == 2
    assert rendered["story_frame_start_index"] == 4
    assert rendered["save_image_start_index"] == 4
    assert rendered["storydiffusion_id_length"] == 2


def test_subject_type_unknown_allowed_with_concrete_identity() -> None:
    raw = _payload(("anchor",))
    raw["characters"][0]["subject_type"] = "unknown"
    raw["characters"][0]["stable_identity"] = "a small red floating lantern with gold trim"
    raw["characters"][0]["anchor_reference_prompt"] = (
        "a single small red floating lantern with gold trim, centered, simple background, one subject in the image"
    )
    raw["scenes"][0]["anchor_generation_prompt"] = "a small red floating lantern with gold trim floating beside a car on a quiet road, medium shot"
    raw["scenes"][1]["anchor_generation_prompt"] = "a small red floating lantern with gold trim resting beside the stopped car, medium shot"
    raw["scenes"][0]["scene_visual_plan"] = {
        "visual_action": "the lantern floats beside a car on a quiet road",
        "action_visibility_cue": "small red lantern hovering beside the car",
        "camera_framing": "medium shot",
    }
    raw["scenes"][0]["action_prompt"] = "red lantern hovering beside the car on the road"
    raw["scenes"][0]["scoring_prompt"] = "red lantern floating beside a car on the road"
    raw["scenes"][1]["scene_visual_plan"] = {
        "visual_action": "the lantern rests beside the stopped car",
        "action_visibility_cue": "small red lantern beside a stopped car",
        "camera_framing": "medium shot",
    }
    raw["scenes"][1]["action_prompt"] = "red lantern resting beside the stopped car"
    raw["scenes"][1]["scoring_prompt"] = "red lantern beside a stopped car"
    payload = NativePromptPayload.from_dict(raw)
    assert validate_native_prompt_payload(payload, _story_single(), targets=["anchor"]) == []


def test_anchor_reference_accepts_generic_single_subject_wording() -> None:
    raw = _payload(("anchor",))
    raw["characters"][0]["anchor_reference_prompt"] = (
        "solo young man with short brown hair and a blue jacket, standing alone, centered, simple background"
    )
    payload = NativePromptPayload.from_dict(raw)
    assert validate_native_prompt_payload(payload, _story_single(), targets=["anchor"]) == []


def test_anchor_reference_rejects_storydiffusion_bracket_tags() -> None:
    raw = _payload(("anchor",))
    raw["characters"][0]["anchor_reference_prompt"] = (
        "[Ben] a single young man with short brown hair and a blue jacket, centered, simple background"
    )
    payload = NativePromptPayload.from_dict(raw)
    issues = validate_native_prompt_payload(payload, _story_single(), targets=["anchor"])
    assert any(issue.severity == "repair_error" and issue.code == "anchor_reference_contains_storydiffusion_tag" for issue in issues)


def test_anchor_generation_requires_stable_identity_context() -> None:
    raw = _payload(("anchor",))
    raw["scenes"][0]["anchor_generation_prompt"] = "Ben drives a car along a quiet road, medium shot"
    payload = NativePromptPayload.from_dict(raw)
    issues = validate_native_prompt_payload(payload, _story_single(), targets=["anchor"])
    assert any(issue.severity == "repair_error" and issue.code == "anchor_generation_missing_stable_identity" for issue in issues)


def test_scene_prompts_reject_reference_only_constraints() -> None:
    raw = _payload(("anchor", "storydiffusion"))
    raw["scenes"][0]["anchor_generation_prompt"] = (
        "a young man driving a car along a quiet road, simple background, medium shot"
    )
    raw["scenes"][1]["storydiffusion_prompt"] = (
        "[Ben] stopping the car beside a quiet road, simple background, medium shot"
    )
    payload = NativePromptPayload.from_dict(raw)
    errors = validate_native_prompt_payload(payload, _story_single(), targets=["anchor", "storydiffusion"])
    assert any(error.severity == "repair_error" for error in errors)
    assert any("$.scenes[0].anchor_generation_prompt contains reference-only phrase" in error for error in errors)
    assert any("$.scenes[1].storydiffusion_prompt contains reference-only phrase" in error for error in errors)


def test_storydiffusion_requires_tags_in_general_and_identity_references() -> None:
    raw = _payload(("storydiffusion",))
    raw["storydiffusion"]["general_prompt"] = "a young man with short brown hair and a blue jacket"
    raw["storydiffusion"]["identity_reference_prompts"] = [
        "a single young man with short brown hair and a blue jacket, centered, simple background"
    ]
    payload = NativePromptPayload.from_dict(raw)
    errors = validate_native_prompt_payload(payload, _story_single(), targets=["storydiffusion"])
    assert any("storydiffusion.general_prompt missing tag [Ben]" in error for error in errors)
    assert any("storydiffusion.identity_reference_prompts[0] must include an exact [Character] tag" in error for error in errors)


def test_storydiffusion_rejects_bare_action_scene_prompt() -> None:
    raw = _payload(("storydiffusion",))
    raw["scenes"][0]["storydiffusion_prompt"] = "[Ben] drives."
    payload = NativePromptPayload.from_dict(raw)
    errors = validate_native_prompt_payload(payload, _story_single(), targets=["storydiffusion"])
    assert any("storydiffusion_prompt is too short for a visual frame prompt" in error for error in errors)


def test_validation_rejects_bad_tags_and_banned_identity_terms() -> None:
    raw = _payload(("storydiffusion",))
    raw["scenes"][0]["storydiffusion_prompt"] = "[Alex] driving a car"
    raw["storydiffusion"]["identity_reference_prompts"][0] = "[Ben] character sheet with multiple views"
    payload = NativePromptPayload.from_dict(raw)
    errors = validate_native_prompt_payload(payload, _story_single(), targets=["storydiffusion"])
    assert any(error.severity == "hard_error" for error in errors if "unknown tag [Alex]" in error)
    assert any("unknown tag [Alex]" in error for error in errors)
    assert any("character sheet" in error for error in errors)
    assert any("multiple views" in error for error in errors)


def test_identity_leakage_is_repair_error_and_overlap_is_warning() -> None:
    raw = _payload(("anchor",))
    raw["characters"][0]["anchor_reference_prompt"] = (
        "a single young man drives a car and stops, centered, simple background"
    )
    payload = NativePromptPayload.from_dict(raw)
    issues = validate_native_prompt_payload(payload, _story_single(), targets=["anchor"])
    assert any(issue.severity == "repair_error" and issue.code == "identity_scene_token_overlap" for issue in issues)

    raw = _payload(("storydiffusion",))
    raw["storydiffusion"]["general_prompt"] = "[Ben] a young man who often drives along roads"
    payload = NativePromptPayload.from_dict(raw)
    issues = validate_native_prompt_payload(payload, _story_single(), targets=["storydiffusion"])
    assert any(issue.severity == "repair_error" and issue.code == "identity_narrative_habit" for issue in issues)


def test_pronoun_explicit_interaction_requires_both_participants() -> None:
    story = Story(
        source_path="07.txt",
        raw_text="[SCENE-1] <Nina> stands in the snow.\n[SEP]\n[SCENE-2] She meets <Leo> in a crowd.",
        scenes=[
            Scene("SCENE-1", 0, "<Nina> stands in the snow.", "Nina stands in the snow.", ["Nina"]),
            Scene("SCENE-2", 1, "She meets <Leo> in a crowd.", "She meets Leo in a crowd.", ["Leo"]),
        ],
        all_entities=["Leo", "Nina"],
        recurring_entities=[],
        entity_to_scene_ids={"Nina": ["SCENE-1"], "Leo": ["SCENE-2"]},
    )
    raw = {
        "target_backends": ["anchor", "storydiffusion"],
        "visual_continuity_anchors": [],
        "characters": [
            {
                "character_id": "Nina",
                "subject_type": "human",
                "stable_identity": "a woman with long dark hair and a warm coat",
                "anchor_reference_prompt": "a single woman with long dark hair and a warm coat, centered, simple background",
            },
            {
                "character_id": "Leo",
                "subject_type": "human",
                "stable_identity": "a young man with short brown hair and a winter jacket",
                "anchor_reference_prompt": "a single young man with short brown hair and a winter jacket, centered, simple background",
            },
        ],
        "scenes": [
            {
                "scene_id": "SCENE-1",
                "visible_character_ids": ["Nina"],
                "identity_conditioning_subject_id": "Nina",
                "anchor_generation_prompt": "Nina stands in the snow, medium shot",
                "storydiffusion_prompt": "[Nina] stands in the snow, medium shot",
                "scoring_prompt": "Nina standing in the snow",
                "action_prompt": "Nina standing in the snow",
                "scene_visual_plan": {
                    "visual_action": "Nina stands in the snow",
                    "action_visibility_cue": "Nina visible in falling snow",
                    "camera_framing": "medium shot",
                },
                "scene_change_level": "large",
                "action_critical": False,
            },
            {
                "scene_id": "SCENE-2",
                "visible_character_ids": ["Leo"],
                "identity_conditioning_subject_id": "Leo",
                "anchor_generation_prompt": "Leo stands in a crowd, medium shot",
                "storydiffusion_prompt": "[Leo] stands in a crowd, medium shot",
                "scoring_prompt": "Leo standing in a crowd",
                "action_prompt": "Leo standing in a crowd",
                "scene_visual_plan": {
                    "visual_action": "Leo stands in a crowd",
                    "action_visibility_cue": "Leo visible among people",
                    "camera_framing": "medium shot",
                },
                "scene_change_level": "medium",
                "action_critical": False,
            },
        ],
        "storydiffusion": {
            "general_prompt": "[Nina] woman with long dark hair and a warm coat\n[Leo] young man with short brown hair and a winter jacket",
            "identity_reference_prompts": [
                "[Nina] woman with long dark hair and a warm coat",
                "[Leo] young man with short brown hair and a winter jacket",
            ],
            "identity_prompts_per_character": 1,
            "identity_negative_prompt_extra": "",
            "scene_negative_prompt_extra": "",
        },
        "notes": {"identity_reasoning": "ok", "continuity_reasoning": "ok", "self_check": "ok"},
    }
    issues = validate_native_prompt_payload(NativePromptPayload.from_dict(raw), story, targets=["anchor", "storydiffusion"])
    assert any(issue.code == "interaction_visible_participant_missing" and issue.severity == "repair_error" for issue in issues)
    assert any(issue.code == "storydiffusion_interaction_tag_missing" and issue.severity == "repair_error" for issue in issues)


def test_warning_only_does_not_trigger_repair() -> None:
    warning_payload = _payload(("anchor",))
    warning_payload["scenes"][0]["scoring_prompt"] = "Ben driving a car with hands near steering wheel, thoughtful mood"
    client = FakeLLMClient([warning_payload])
    builder = LLMDirectPromptBuilder(
        {
            "llm": {"provider": "openai", "model": "fake"},
            "llm_direct": {"targets": ["anchor"], "repair_attempts": 2},
        },
        llm_client=client,
    )
    payload = builder.build(_story_single())
    assert payload.scenes[0].scoring_prompt == warning_payload["scenes"][0]["scoring_prompt"]
    assert client.calls == 1
    assert builder.last_validation_status == "passed_with_warnings"
    assert builder.last_warnings


def test_repair_runs_until_valid_and_records_field_diff() -> None:
    invalid = _payload(("storydiffusion",))
    invalid["scenes"][0]["storydiffusion_prompt"] = "[Alex] driving a car"
    still_invalid = _payload(("storydiffusion",))
    still_invalid["scenes"][0]["storydiffusion_prompt"] = "[Alex] driving a car"
    repaired = _payload(("storydiffusion",))
    client = FakeLLMClient([invalid, still_invalid, repaired])
    builder = LLMDirectPromptBuilder(
        {
            "llm": {"provider": "openai", "model": "fake"},
            "llm_direct": {"targets": ["storydiffusion"], "repair_attempts": 2},
        },
        llm_client=client,
    )

    payload = builder.build(_story_single())
    assert payload.scenes[0].storydiffusion_prompt == repaired["scenes"][0]["storydiffusion_prompt"]
    assert client.calls == 3
    assert builder.last_repair_attempts_used == 2
    assert builder.last_repair_diff
    assert any(diff["path"].endswith(".storydiffusion_prompt") for diff in builder.last_repair_diff)


def test_unresolved_validator_errors_continue_when_payload_is_usable() -> None:
    invalid = _payload(("storydiffusion",))
    invalid["scenes"][0]["storydiffusion_prompt"] = "[Alex] driving a car"
    client = FakeLLMClient([invalid, invalid])
    builder = LLMDirectPromptBuilder(
        {
            "llm": {"provider": "openai", "model": "fake"},
            "llm_direct": {"targets": ["storydiffusion"], "repair_attempts": 1},
        },
        llm_client=client,
    )
    payload = builder.build(_story_single())
    assert payload.scenes[0].storydiffusion_prompt == "[Alex] driving a car"
    assert client.calls == 2
    assert builder.last_validation_status == "best_effort_with_unresolved_issues"
    assert builder.last_generation_allowed is True
    assert builder.last_unresolved_errors


def test_best_effort_can_allow_unresolved_boundary_errors() -> None:
    invalid = _payload(("anchor",))
    invalid["characters"][0]["anchor_reference_prompt"] = (
        "a young man driving a car along a quiet road, centered, simple background"
    )
    client = FakeLLMClient([invalid])
    builder = LLMDirectPromptBuilder(
        {
            "llm": {"provider": "openai", "model": "fake"},
            "llm_direct": {
                "targets": ["anchor"],
                "repair_attempts": 0,
                "validation_policy": "best_effort",
                "allow_generation_with_boundary_errors": True,
            },
        },
        llm_client=client,
    )
    payload = builder.build(_story_single())
    assert payload.characters[0].anchor_reference_prompt == invalid["characters"][0]["anchor_reference_prompt"]
    assert builder.last_validation_status == "best_effort_with_unresolved_issues"
    assert builder.last_generation_allowed is True


def test_api_failure_before_payload_records_failed_no_payload() -> None:
    builder = LLMDirectPromptBuilder(
        {
            "llm": {"provider": "openai", "model": "fake"},
            "llm_direct": {"targets": ["anchor"]},
        },
        llm_client=FailingLLMClient(),
    )
    with pytest.raises(RuntimeError):
        builder.build(_story_single())
    assert builder.last_validation_status == "failed_no_payload"
    assert builder.last_generation_allowed is False
    assert builder.last_payload is None


def test_unparseable_payload_records_failed_unparseable_payload() -> None:
    builder = LLMDirectPromptBuilder(
        {
            "llm": {"provider": "openai", "model": "fake"},
            "llm_direct": {"targets": ["anchor"]},
        },
        llm_client=NonDictLLMClient(),
    )
    with pytest.raises(NativePromptValidationError):
        builder.build(_story_single())
    assert builder.last_validation_status == "failed_unparseable_payload"
    assert builder.last_generation_allowed is False


def test_missing_backend_minimum_fields_remains_fatal() -> None:
    invalid = _payload(("storydiffusion",))
    for scene in invalid["scenes"]:
        scene["storydiffusion_prompt"] = ""
    client = FakeLLMClient([invalid, invalid])
    builder = LLMDirectPromptBuilder(
        {
            "llm": {"provider": "openai", "model": "fake"},
            "llm_direct": {"targets": ["storydiffusion"], "repair_attempts": 1},
        },
        llm_client=client,
    )
    with pytest.raises(NativePromptValidationError):
        builder.build(_story_single())
    assert builder.last_validation_status == "failed_unparseable_payload"
    assert builder.last_generation_allowed is False


def test_stateful_planning_bird_flight_payload() -> None:
    story = Story(
        source_path="extra_06.txt",
        raw_text="[SCENE-1] <Bird> sits on a branch.\n[SEP]\n[SCENE-2] It looks around.\n[SEP]\n[SCENE-3] It flies away.",
        scenes=[
            Scene("SCENE-1", 0, "<Bird> sits on a branch.", "Bird sits on a branch.", ["Bird"]),
            Scene("SCENE-2", 1, "It looks around.", "It looks around.", []),
            Scene("SCENE-3", 2, "It flies away.", "It flies away.", []),
        ],
        all_entities=["Bird"],
        recurring_entities=["Bird"],
        entity_to_scene_ids={"Bird": ["SCENE-1"]},
    )
    raw = {
        "target_backends": ["anchor"],
        "visual_continuity_anchors": [
            {
                "anchor_id": "same_branch",
                "type": "persistent_setting",
                "applies_to_scene_ids": ["SCENE-1", "SCENE-2", "SCENE-3"],
                "prompt_phrase": "same tree branch",
                "state_by_scene": {},
            }
        ],
        "characters": [
            {
                "character_id": "Bird",
                "subject_type": "animal",
                "stable_identity": "a small colorful bird with blue orange yellow and green feathers",
                "anchor_reference_prompt": "a single small colorful bird with blue orange yellow and green feathers, full body identity reference, neutral pose, simple plain background, one bird only",
            }
        ],
        "scenes": [
            {
                "scene_id": "SCENE-1",
                "visible_character_ids": ["Bird"],
                "identity_conditioning_subject_id": "Bird",
                "anchor_generation_prompt": "a small colorful bird with blue orange yellow and green feathers perched on the same tree branch, feet gripping the branch, medium shot",
                "scene_visual_plan": {
                    "visual_action": "the bird is perched on the same tree branch",
                    "action_visibility_cue": "feet gripping the branch",
                    "camera_framing": "medium shot",
                },
                "scene_change_level": "small",
                "action_critical": False,
                "action_prompt": "bird perched on branch with feet gripping branch",
                "scoring_prompt": "colorful bird perched on same branch with feet gripping it",
            },
            {
                "scene_id": "SCENE-2",
                "visible_character_ids": ["Bird"],
                "identity_conditioning_subject_id": "Bird",
                "anchor_generation_prompt": "a small colorful bird with blue orange yellow and green feathers perched on the same tree branch, head turned as it looks around, medium shot",
                "scene_visual_plan": {
                    "visual_action": "the bird looks around while perched on the same tree branch",
                    "action_visibility_cue": "head turned to the side",
                    "camera_framing": "medium shot",
                },
                "scene_change_level": "small",
                "action_critical": True,
                "action_prompt": "bird perched on same branch with head turned to look around",
                "scoring_prompt": "colorful bird looking around on same branch with head turned",
            },
            {
                "scene_id": "SCENE-3",
                "visible_character_ids": ["Bird"],
                "identity_conditioning_subject_id": "Bird",
                "anchor_generation_prompt": "a small colorful bird with blue orange yellow and green feathers fully airborne above the same tree branch, wings fully spread, both feet lifted clear of the branch, medium-wide shot",
                "scene_visual_plan": {
                    "visual_action": "the bird is fully airborne and flying away from the same tree branch",
                    "action_visibility_cue": "wings fully spread, both feet lifted clear of the branch",
                    "camera_framing": "medium-wide shot",
                },
                "scene_change_level": "large",
                "action_critical": True,
                "action_prompt": "bird airborne in mid-flight, wings fully spread, feet not touching the branch",
                "scoring_prompt": "small colorful bird flying above the branch, wings spread, feet off the branch",
            },
        ],
    }
    payload = NativePromptPayload.from_dict(raw)
    assert validate_native_prompt_payload(payload, story, targets=["anchor"]) == []
    assert payload.scenes[2].scene_change_level == "large"
    assert payload.scenes[2].action_critical is True
    assert "branch" not in payload.characters[0].anchor_reference_prompt.lower()


def test_stateful_planning_lucy_lighting_anchor() -> None:
    story = Story(
        source_path="lucy.txt",
        raw_text="[SCENE-1] <Lucy> walks into a room.\n[SEP]\n[SCENE-2] She turns on the light.\n[SEP]\n[SCENE-3] She sits on a chair.",
        scenes=[
            Scene("SCENE-1", 0, "<Lucy> walks into a room.", "Lucy walks into a room.", ["Lucy"]),
            Scene("SCENE-2", 1, "She turns on the light.", "She turns on the light.", []),
            Scene("SCENE-3", 2, "She sits on a chair.", "She sits on a chair.", []),
        ],
        all_entities=["Lucy"],
        recurring_entities=["Lucy"],
        entity_to_scene_ids={"Lucy": ["SCENE-1"]},
    )
    raw = _payload(("anchor",))
    raw["characters"][0]["character_id"] = "Lucy"
    raw["characters"][0]["stable_identity"] = "a young woman with short black hair and a red sweater"
    raw["characters"][0]["anchor_reference_prompt"] = "a single young woman with short black hair and a red sweater, centered, simple background, one person only"
    raw["visual_continuity_anchors"] = [
        {
            "anchor_id": "room_lighting",
            "type": "evolving_visual_state",
            "applies_to_scene_ids": ["SCENE-1", "SCENE-2", "SCENE-3"],
            "prompt_phrase": "",
            "state_by_scene": {
                "SCENE-1": "dim room before the light is turned on",
                "SCENE-2": "same room as warm light turns on and fills the room",
                "SCENE-3": "same now-lit room with warm light continuing",
            },
        }
    ]
    raw["scenes"] = [
        {
            "scene_id": "SCENE-1",
            "visible_character_ids": ["Lucy"],
            "identity_conditioning_subject_id": "Lucy",
            "anchor_generation_prompt": "a young woman with short black hair and a red sweater walks through a doorway into a dim room before the light is turned on, dark room visible, medium-wide shot",
            "scene_visual_plan": {"visual_action": "Lucy walks into the dim room", "action_visibility_cue": "doorway and dark room visible", "camera_framing": "medium-wide shot"},
            "scene_change_level": "large",
            "action_critical": True,
            "action_prompt": "Lucy walking through doorway into dim room",
            "scoring_prompt": "Lucy walking into dim room through doorway",
        },
        {
            "scene_id": "SCENE-2",
            "visible_character_ids": ["Lucy"],
            "identity_conditioning_subject_id": "Lucy",
            "anchor_generation_prompt": "a young woman with short black hair and a red sweater turns on the light in the same room as warm light turns on and fills the room, medium shot",
            "scene_visual_plan": {"visual_action": "Lucy turns on the light", "action_visibility_cue": "hand near light switch and warm light filling room", "camera_framing": "medium shot"},
            "scene_change_level": "medium",
            "action_critical": True,
            "action_prompt": "Lucy turning on light switch as warm light fills room",
            "scoring_prompt": "Lucy turning on light, warm light filling room",
        },
        {
            "scene_id": "SCENE-3",
            "visible_character_ids": ["Lucy"],
            "identity_conditioning_subject_id": "Lucy",
            "anchor_generation_prompt": "a young woman with short black hair and a red sweater sits on a chair with seated posture in the same now-lit room with warm light continuing, medium shot",
            "scene_visual_plan": {"visual_action": "Lucy sits on a chair", "action_visibility_cue": "seated posture on chair in lit room", "camera_framing": "medium shot"},
            "scene_change_level": "medium",
            "action_critical": True,
            "action_prompt": "Lucy seated on chair in warm lit room",
            "scoring_prompt": "Lucy sitting on chair in warm lit room",
        },
    ]
    assert validate_native_prompt_payload(NativePromptPayload.from_dict(raw), story, targets=["anchor"]) == []


def test_stateful_planning_milo_toys_persist() -> None:
    story = Story(
        source_path="milo.txt",
        raw_text="[SCENE-1] <Milo> sits on the floor with toys.\n[SEP]\n[SCENE-2] He rolls over and laughs.\n[SEP]\n[SCENE-3] He lies down and rests.",
        scenes=[
            Scene("SCENE-1", 0, "<Milo> sits on the floor with toys.", "Milo sits on the floor with toys.", ["Milo"]),
            Scene("SCENE-2", 1, "He rolls over and laughs.", "He rolls over and laughs.", []),
            Scene("SCENE-3", 2, "He lies down and rests.", "He lies down and rests.", []),
        ],
        all_entities=["Milo"],
        recurring_entities=["Milo"],
        entity_to_scene_ids={"Milo": ["SCENE-1"]},
    )
    raw = _payload(("anchor",))
    raw["characters"][0]["character_id"] = "Milo"
    raw["characters"][0]["stable_identity"] = "a toddler boy with short brown hair and blue pajamas"
    raw["characters"][0]["anchor_reference_prompt"] = "a single toddler boy with short brown hair and blue pajamas, centered, simple background, one person only"
    raw["visual_continuity_anchors"] = [
        {
            "anchor_id": "floor_toys",
            "type": "persistent_setting",
            "applies_to_scene_ids": ["SCENE-1", "SCENE-2", "SCENE-3"],
            "prompt_phrase": "same playroom floor with colorful toys nearby",
            "state_by_scene": {},
        }
    ]
    raw["scenes"] = []
    prompts = [
        ("SCENE-1", "Milo sits on the same playroom floor with colorful toys nearby", "seated on floor among toys", "Milo sitting on floor with toys"),
        ("SCENE-2", "Milo rolls over and laughs on the same playroom floor with colorful toys nearby", "rolling body pose on floor near toys", "Milo rolling over on floor with toys nearby"),
        ("SCENE-3", "Milo lies down and rests on the same playroom floor with colorful toys nearby", "lying down posture on floor near toys", "Milo lying down on floor with toys nearby"),
    ]
    for scene_id, visual_action, cue, scoring in prompts:
        raw["scenes"].append(
            {
                "scene_id": scene_id,
                "visible_character_ids": ["Milo"],
                "identity_conditioning_subject_id": "Milo",
                "anchor_generation_prompt": f"a toddler boy with short brown hair and blue pajamas, {visual_action}, {cue}, medium shot",
                "storydiffusion_prompt": "",
                "scene_visual_plan": {"visual_action": visual_action, "action_visibility_cue": cue, "camera_framing": "medium shot"},
                "scene_change_level": "medium",
                "action_critical": True,
                "action_prompt": cue,
                "scoring_prompt": scoring,
            }
        )
    assert validate_native_prompt_payload(NativePromptPayload.from_dict(raw), story, targets=["anchor"]) == []


def test_stateful_planning_robot_task_and_scanning() -> None:
    story = Story(
        source_path="robot.txt",
        raw_text="[SCENE-1] <Robot> works in a factory.\n[SEP]\n[SCENE-2] It stops and scans the area.\n[SEP]\n[SCENE-3] It continues its task.",
        scenes=[
            Scene("SCENE-1", 0, "<Robot> works in a factory.", "Robot works in a factory.", ["Robot"]),
            Scene("SCENE-2", 1, "It stops and scans the area.", "It stops and scans the area.", []),
            Scene("SCENE-3", 2, "It continues its task.", "It continues its task.", []),
        ],
        all_entities=["Robot"],
        recurring_entities=["Robot"],
        entity_to_scene_ids={"Robot": ["SCENE-1"]},
    )
    raw = _payload(("anchor",))
    raw["characters"][0]["character_id"] = "Robot"
    raw["characters"][0]["subject_type"] = "robot"
    raw["characters"][0]["stable_identity"] = "a silver and blue service robot with round sensor eyes"
    raw["characters"][0]["anchor_reference_prompt"] = "a single silver and blue service robot with round sensor eyes, centered, simple background, one robot only"
    raw["visual_continuity_anchors"] = [
        {
            "anchor_id": "assembly_task",
            "type": "persistent_task_and_setting",
            "applies_to_scene_ids": ["SCENE-1", "SCENE-2", "SCENE-3"],
            "prompt_phrase": "same factory assembly station with metal parts on a conveyor belt",
            "state_by_scene": {},
        }
    ]
    raw["scenes"] = []
    scene_data = [
        ("SCENE-1", "the robot assembles parts at the same factory assembly station and conveyor belt", "robot arms holding metal parts over conveyor", "medium-wide shot", "robot assembling metal parts on conveyor belt"),
        ("SCENE-2", "the robot stops and scans the same factory assembly station and conveyor belt", "head turned, glowing sensor eyes projecting scanning beam", "medium close-up shot", "robot scanning factory machines with glowing sensor beam"),
        ("SCENE-3", "the robot continues assembling parts at the same factory assembly station and conveyor belt", "robot arms back on metal parts over conveyor", "medium-wide shot", "robot continuing assembly task on conveyor belt"),
    ]
    for scene_id, visual_action, cue, framing, scoring in scene_data:
        raw["scenes"].append(
            {
                "scene_id": scene_id,
                "visible_character_ids": ["Robot"],
                "identity_conditioning_subject_id": "Robot",
                "anchor_generation_prompt": f"a silver and blue service robot with round sensor eyes, {visual_action}, {cue}, {framing}",
                "storydiffusion_prompt": "",
                "scene_visual_plan": {"visual_action": visual_action, "action_visibility_cue": cue, "camera_framing": framing},
                "scene_change_level": "medium",
                "action_critical": True,
                "action_prompt": cue,
                "scoring_prompt": scoring,
            }
        )
    assert validate_native_prompt_payload(NativePromptPayload.from_dict(raw), story, targets=["anchor"]) == []
