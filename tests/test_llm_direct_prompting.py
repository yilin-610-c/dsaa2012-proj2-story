from __future__ import annotations

import json

import pytest

from storygen.llm_client import BaseLLMClient, LLMResponse
from storygen.anchor_bank import build_anchor_prompt
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
        "characters": [
            {
                "character_id": "Ben",
                "subject_type": "human",
                "stable_identity": "a young man with short brown hair and a blue jacket",
                "anchor_reference_prompt": "[Ben] a single young man with short brown hair and a blue jacket, standing alone, centered, simple background, one person in the image",
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
                "action_prompt": "",
            },
            {
                "scene_id": "SCENE-2",
                "visible_character_ids": ["Ben"],
                "identity_conditioning_subject_id": "Ben",
                "anchor_generation_prompt": "a young man stopping the car beside a quiet road, medium shot",
                "storydiffusion_prompt": "[Ben] stopping the car beside a quiet road, medium shot",
                "scoring_prompt": "Ben stopping a car",
                "action_prompt": "",
            },
        ],
        "storydiffusion": {
            "general_prompt": "[Ben] a young man with short brown hair and a blue jacket",
            "identity_reference_prompts": [
                "[Ben] a single young man with short brown hair and a blue jacket, standing alone, centered, simple background, one person in the image"
            ],
            "identity_prompts_per_character": 1,
            "negative_prompt_extra": "",
        },
        "notes": {"identity_reasoning": "Ben is recurring.", "continuity_reasoning": "Ben remains visible.", "self_check": "ok"},
    }


def test_anchor_adapter_copies_llm_prompt_fields_exactly() -> None:
    payload = NativePromptPayload.from_dict(_payload(("anchor",)))
    errors = validate_native_prompt_payload(payload, _story_single(), targets=["anchor"])
    assert errors == []

    bundle = build_anchor_prompt_bundle(payload, _story_single(), {"negative_prompt": "blurry"})
    first = bundle.scene_prompts["SCENE-1"]
    assert first.generation_prompt == payload.scenes[0].anchor_generation_prompt
    assert first.scoring_prompt == payload.scenes[0].scoring_prompt
    assert first.action_prompt == ""
    assert bundle.metadata["character_specs"]["Ben"]["anchor_reference_prompt"] == payload.characters[0].anchor_reference_prompt
    assert bundle.metadata["scene_route_hints"]["SCENE-1"]["identity_conditioning_subject_id"] == "Ben"


def test_anchor_bank_uses_llm_direct_anchor_reference_prompt_without_suffix() -> None:
    prompt = "[Ben] a single young man with short brown hair, centered, simple background, one person in the image"
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
            "storydiffusion_prompt": "[Ben] [Sara] meeting on a quiet street, both visible, medium two-shot",
            "scoring_prompt": "",
            "action_prompt": "",
        },
        {
            "scene_id": "SCENE-2",
            "visible_character_ids": ["Ben", "Sara"],
            "identity_conditioning_subject_id": None,
            "anchor_generation_prompt": "",
            "storydiffusion_prompt": "[Ben] [Sara] talking together on the same quiet street, medium two-shot",
            "scoring_prompt": "",
            "action_prompt": "",
        },
    ]
    raw["storydiffusion"]["general_prompt"] = (
        "[Ben] a young man with short brown hair and a blue jacket\n"
        "[Sara] a young woman with long black hair and a green coat"
    )
    raw["storydiffusion"]["identity_prompts_per_character"] = 2
    raw["storydiffusion"]["identity_reference_prompts"] = [
        raw["characters"][0]["anchor_reference_prompt"],
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
        "[Ben] a single small red floating lantern with gold trim, centered, simple background, one subject in the image"
    )
    payload = NativePromptPayload.from_dict(raw)
    assert validate_native_prompt_payload(payload, _story_single(), targets=["anchor"]) == []


def test_anchor_reference_accepts_generic_single_subject_wording() -> None:
    raw = _payload(("anchor",))
    raw["characters"][0]["anchor_reference_prompt"] = (
        "[Ben] solo young man with short brown hair and a blue jacket, standing alone, centered, simple background"
    )
    payload = NativePromptPayload.from_dict(raw)
    assert validate_native_prompt_payload(payload, _story_single(), targets=["anchor"]) == []


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
    assert any("scenes[SCENE-1].anchor_generation_prompt contains reference-only phrase" in error for error in errors)
    assert any("scenes[SCENE-2].storydiffusion_prompt contains reference-only phrase" in error for error in errors)


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
    assert any("unknown tag [Alex]" in error for error in errors)
    assert any("character sheet" in error for error in errors)
    assert any("multiple views" in error for error in errors)


def test_repair_runs_once_and_records_field_diff() -> None:
    invalid = _payload(("storydiffusion",))
    invalid["scenes"][0]["storydiffusion_prompt"] = "[Alex] driving a car"
    repaired = _payload(("storydiffusion",))
    client = FakeLLMClient([invalid, repaired])
    builder = LLMDirectPromptBuilder(
        {
            "llm": {"provider": "openai", "model": "fake"},
            "llm_direct": {"targets": ["storydiffusion"], "repair_attempts": 1},
        },
        llm_client=client,
    )

    payload = builder.build(_story_single())
    assert payload.scenes[0].storydiffusion_prompt == repaired["scenes"][0]["storydiffusion_prompt"]
    assert client.calls == 2
    assert builder.last_repair_diff
    assert any(diff["path"].endswith(".storydiffusion_prompt") for diff in builder.last_repair_diff)


def test_repair_failure_raises_validation_error() -> None:
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
    with pytest.raises(NativePromptValidationError):
        builder.build(_story_single())
    assert client.calls == 2
