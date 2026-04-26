import json
from pathlib import Path

import pytest

from storygen.llm_assisted_prompt_builder import LLMAssistedPromptBuilder, LLMPromptError
from storygen.llm_client import LLMResponse
from storygen.prompt_cache import build_prompt_cache_key
from storygen.prompt_pipelines import build_prompt_pipeline
from storygen.types import Scene, Story


class FakeLLMClient:
    def __init__(self, payload: dict | None = None, *, error: Exception | None = None) -> None:
        self.payload = payload or _llm_payload()
        self.error = error
        self.calls = 0

    def generate_structured(self, *, messages: list[dict[str, str]], json_schema: dict) -> LLMResponse:
        del messages, json_schema
        self.calls += 1
        if self.error is not None:
            raise self.error
        return LLMResponse(
            raw_text=json.dumps(self.payload),
            parsed_json=self.payload,
            metadata={"provider": "fake", "model": "fake-model"},
        )


def _story(raw_text: str = "Hero runs.\nHero stops.") -> Story:
    return Story(
        source_path="story.txt",
        raw_text=raw_text,
        scenes=[
            Scene("SCENE-1", 0, "<Hero> runs.", "Hero runs.", ["Hero"]),
            Scene("SCENE-2", 1, "<Hero> stops.", "Hero stops.", ["Hero"]),
        ],
        all_entities=["Hero"],
        recurring_entities=["Hero"],
        entity_to_scene_ids={"Hero": ["SCENE-1", "SCENE-2"]},
    )


def _two_character_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="Jack and Sara talk.",
        scenes=[
            Scene("SCENE-1", 0, "<Jack> and <Sara> talk.", "Jack and Sara talk.", ["Jack", "Sara"]),
            Scene("SCENE-2", 1, "They visit a cafe.", "They visit a cafe.", []),
        ],
        all_entities=["Jack", "Sara"],
        recurring_entities=[],
        entity_to_scene_ids={"Jack": ["SCENE-1"], "Sara": ["SCENE-1"]},
    )


def _two_character_rich_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="Jack and Sara sit side by side on a park bench and talk.",
        scenes=[
            Scene(
                "SCENE-1",
                0,
                "<Jack> and <Sara> sit side by side on a park bench and talk.",
                "Jack and Sara sit side by side on a park bench and talk.",
                ["Jack", "Sara"],
            ),
            Scene(
                "SCENE-2",
                1,
                "Jack and Sara face each other across a cafe table.",
                "Jack and Sara face each other across a cafe table.",
                ["Jack", "Sara"],
            ),
        ],
        all_entities=["Jack", "Sara"],
        recurring_entities=[],
        entity_to_scene_ids={"Jack": ["SCENE-1", "SCENE-2"], "Sara": ["SCENE-1", "SCENE-2"]},
    )


def _sparse_two_character_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="Jack and Sara.",
        scenes=[
            Scene("SCENE-1", 0, "<Jack> and <Sara>.", "Jack and Sara.", ["Jack", "Sara"]),
            Scene("SCENE-2", 1, "Jack and Sara.", "Jack and Sara.", ["Jack", "Sara"]),
        ],
        all_entities=["Jack", "Sara"],
        recurring_entities=[],
        entity_to_scene_ids={"Jack": ["SCENE-1", "SCENE-2"], "Sara": ["SCENE-1", "SCENE-2"]},
    )


def _bird_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="Bird flies.",
        scenes=[Scene("SCENE-1", 0, "<Bird> flies.", "Bird flies.", ["Bird"])],
        all_entities=["Bird"],
        recurring_entities=[],
        entity_to_scene_ids={"Bird": ["SCENE-1"]},
    )


def _milo_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="<Milo> sits on the floor with toys.\nHe rolls over and laughs.\nHe lies down and rests.",
        scenes=[
            Scene("SCENE-1", 0, "<Milo> sits on the floor with toys.", "Milo sits on the floor with toys.", ["Milo"]),
            Scene("SCENE-2", 1, "He rolls over and laughs.", "He rolls over and laughs.", []),
            Scene("SCENE-3", 2, "He lies down and rests.", "He lies down and rests.", []),
        ],
        all_entities=["Milo"],
        recurring_entities=["Milo"],
        entity_to_scene_ids={"Milo": ["SCENE-1"]},
    )


def _ryan_bus_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="<Ryan> walks quickly toward a bus.\nHe pauses at the door and looks ahead.",
        scenes=[
            Scene("SCENE-1", 0, "<Ryan> walks quickly toward a bus.", "Ryan walks quickly toward a bus.", ["Ryan"]),
            Scene("SCENE-2", 1, "He pauses at the door and looks ahead.", "He pauses at the door and looks ahead.", []),
        ],
        all_entities=["Ryan"],
        recurring_entities=["Ryan"],
        entity_to_scene_ids={"Ryan": ["SCENE-1"]},
    )


def _emma_city_lights_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="Emma watches city lights.",
        scenes=[
            Scene("SCENE-1", 0, "<Emma> stands on a bridge and looks down.", "Emma stands on a bridge and looks down.", ["Emma"]),
            Scene("SCENE-2", 1, "The city lights come on at night.", "The city lights come on at night.", []),
            Scene("SCENE-3", 2, "She smiles at the view.", "She smiles at the view.", []),
        ],
        all_entities=["Emma"],
        recurring_entities=[],
        entity_to_scene_ids={"Emma": ["SCENE-1"]},
    )


def _delayed_character_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="Nina meets Leo.",
        scenes=[
            Scene("SCENE-1", 0, "<Nina> stands in the snow.", "Nina stands in the snow.", ["Nina"]),
            Scene("SCENE-2", 1, "She meets <Leo> in a crowd.", "She meets Leo in a crowd.", ["Leo"]),
        ],
        all_entities=["Leo", "Nina"],
        recurring_entities=[],
        entity_to_scene_ids={"Leo": ["SCENE-2"], "Nina": ["SCENE-1"]},
    )


def _friend_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="Tom meets a friend.",
        scenes=[
            Scene("SCENE-1", 0, "<Tom> runs down a street.", "Tom runs down a street.", ["Tom"]),
            Scene("SCENE-2", 1, "He meets his friend and keeps going.", "He meets his friend and keeps going.", []),
        ],
        all_entities=["Tom"],
        recurring_entities=[],
        entity_to_scene_ids={"Tom": ["SCENE-1"]},
    )


def _prompt_config(tmp_path: Path, *, fallback: bool = True, cache_enabled: bool = True) -> dict:
    return {
        "pipeline": "llm_assisted",
        "rewriter": {"type": "rule_based"},
        "style_prompt": "cinematic illustration",
        "subject_prefix": "main subject:",
        "global_context_prefix": "shared story context:",
        "setting_prefix": "recurring setting:",
        "replace_leading_pronouns": True,
        "human_identity_prompt": "same person across all scenes",
        "animal_identity_prompt": "same animal across all scenes",
        "generic_identity_prompt": "same subject across all scenes",
        "scene_continuity_prompt": "keep the same lighting and palette",
        "action_emphasis_template": "key action: {action_phrase}",
        "default_action_prompt": "show the action clearly",
        "scene_composition_prompt": "keep the pose easy to read",
        "generation_include_style": True,
        "generation_include_global_context": False,
        "generation_include_quality_suffix": False,
        "generation_include_scene_composition": False,
        "generation_max_words": 8,
        "generation_max_chars": 80,
        "generation_template": "{subject}, {action}{setting_clause}{style_clause}",
        "scoring_template": "{subject}, {action}{setting_clause}",
        "scoring_include_style": False,
        "scoring_include_global_context": False,
        "scoring_max_words": 5,
        "scoring_max_chars": 60,
        "action_emphasis_map": {"runs": "active running pose"},
        "quality_suffix": "clean composition",
        "negative_prompt": "blurry",
        "cache": {"enabled": cache_enabled, "cache_dir": str(tmp_path / "cache")},
        "artifact": {"path": None, "export_enabled": False, "export_dir": str(tmp_path / "artifacts")},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-2024-08-06",
            "api_key_env": "OPENAI_API_KEY",
            "temperature": 0.0,
            "max_output_tokens": 800,
            "timeout_seconds": 30,
            "schema_version": "v1",
            "builder_version": "llm_assisted_v9",
            "fallback_to_rule_based": fallback,
        },
    }


def _route_factors(
    *,
    same_subject: bool = True,
    same_setting: bool = True,
    body_state_change: bool = False,
    primary_action_change: bool = False,
    new_key_objects: list[str] | None = None,
    composition_change_needed: bool = False,
) -> dict:
    return {
        "same_subject": same_subject,
        "same_setting": same_setting,
        "body_state_change": body_state_change,
        "primary_action_change": primary_action_change,
        "new_key_objects": new_key_objects or [],
        "composition_change_needed": composition_change_needed,
    }


def _llm_payload() -> dict:
    return {
        "global": {
            "main_character": "Hero",
            "identity_cues": ["same red jacket"],
            "shared_setting": ["city park"],
            "style_cues": ["cinematic illustration", "clean storyboard frame"],
            "characters": [
                {
                    "character_id": "Hero",
                    "age_band": "adult",
                    "gender_presentation": "unspecified",
                    "hair_color": "",
                    "hairstyle": "",
                    "skin_tone": "",
                    "body_build": "",
                    "signature_outfit": "red jacket",
                    "signature_accessory": "",
                    "profession_marker": "",
                }
            ],
        },
        "scenes": [
            {
                "scene_id": "SCENE-1",
                "primary_action": "runs",
                "secondary_elements": ["path"],
                "generation_prompt": "Hero runs along the path",
                "scoring_prompt": "Hero runs",
                "action_prompt": "running",
                "continuity_subject_ids": ["Hero"],
                "continuity_route_hint": "text2img",
                "route_change_level": "large",
                "route_factors": _route_factors(),
                "route_reason": "First scene establishes the frame.",
                "identity_conditioning_subject_id": "Hero",
                "primary_visible_character_ids": ["Hero"],
                "interaction_summary": "",
                "spatial_relation": "",
                "framing": "clear single-person composition",
                "setting_focus": "path in the park",
            },
            {
                "scene_id": "SCENE-2",
                "primary_action": "stops",
                "secondary_elements": ["path"],
                "generation_prompt": "Hero stops on the path",
                "scoring_prompt": "Hero stops",
                "action_prompt": "stopping",
                "continuity_subject_ids": ["Hero"],
                "continuity_route_hint": "img2img",
                "route_change_level": "small",
                "route_factors": _route_factors(),
                "route_reason": "Same subject and path, only the action changes.",
                "identity_conditioning_subject_id": "Hero",
                "primary_visible_character_ids": ["Hero"],
                "interaction_summary": "",
                "spatial_relation": "",
                "framing": "clear single-person composition",
                "setting_focus": "same park path",
            },
        ],
    }


def _two_character_llm_payload() -> dict:
    payload = _llm_payload()
    payload["global"] = {
        "main_character": "Jack",
        "identity_cues": ["Jack with black jacket", "Sara with yellow scarf"],
        "shared_setting": ["park and cafe"],
        "style_cues": ["clean storyboard frame"],
        "characters": [
            {
                "character_id": "Jack",
                "age_band": "adult",
                "gender_presentation": "man",
                "hair_color": "black",
                "hairstyle": "short hair",
                "skin_tone": "",
                "body_build": "",
                "signature_outfit": "black jacket",
                "signature_accessory": "",
                "profession_marker": "",
            },
            {
                "character_id": "Sara",
                "age_band": "adult",
                "gender_presentation": "woman",
                "hair_color": "brown",
                "hairstyle": "shoulder-length hair",
                "skin_tone": "",
                "body_build": "",
                "signature_outfit": "",
                "signature_accessory": "yellow scarf",
                "profession_marker": "",
            },
        ],
    }
    payload["scenes"] = [
        {
            "scene_id": "SCENE-1",
            "primary_action": "talk",
            "secondary_elements": ["park bench"],
            "generation_prompt": "Jack and Sara talking on a park bench",
            "scoring_prompt": "two people talking",
            "action_prompt": "talking",
            "continuity_subject_ids": ["Jack", "Sara"],
            "continuity_route_hint": "text2img",
            "route_change_level": "large",
            "route_factors": _route_factors(same_subject=True),
            "route_reason": "First scene establishes both characters.",
            "identity_conditioning_subject_id": None,
            "primary_visible_character_ids": ["Jack", "Sara"],
            "interaction_summary": "Jack and Sara talk together",
            "spatial_relation": "Jack and Sara sit side by side on the bench",
            "framing": "clear two-person composition, both characters visible",
            "setting_focus": "park bench",
        },
        {
            "scene_id": "SCENE-2",
            "primary_action": "visit cafe",
            "secondary_elements": ["cafe table"],
            "generation_prompt": "Jack and Sara sitting at a cafe table",
            "scoring_prompt": "two people at a cafe",
            "action_prompt": "sitting at cafe",
            "continuity_subject_ids": ["Jack", "Sara"],
            "continuity_route_hint": "text2img",
            "route_change_level": "large",
            "route_factors": _route_factors(same_subject=True, same_setting=False, composition_change_needed=True),
            "route_reason": "Same characters but setting changes.",
            "identity_conditioning_subject_id": None,
            "primary_visible_character_ids": ["Jack", "Sara"],
            "interaction_summary": "Jack and Sara sit together at the cafe table",
            "spatial_relation": "Jack and Sara face each other across the table",
            "framing": "clear two-person composition, both characters visible",
            "setting_focus": "cafe table",
        },
    ]
    return payload


def _bird_llm_payload() -> dict:
    return {
        "global": {
            "main_character": "Bird",
            "identity_cues": ["small blue bird", "feathered wings"],
            "shared_setting": ["tree branch"],
            "style_cues": ["clean storyboard frame"],
            "characters": [
                {
                    "character_id": "Bird",
                    "age_band": "",
                    "gender_presentation": "",
                    "hair_color": "",
                    "hairstyle": "",
                    "skin_tone": "",
                    "body_build": "small bird body",
                    "signature_outfit": "",
                    "signature_accessory": "",
                    "profession_marker": "",
                }
            ],
        },
        "scenes": [
            {
                "scene_id": "SCENE-1",
                "primary_action": "flies",
                "secondary_elements": ["branch"],
                "generation_prompt": "Bird flying from a branch",
                "scoring_prompt": "bird flying",
                "action_prompt": "flying",
                "continuity_subject_ids": ["Bird"],
                "continuity_route_hint": "text2img",
                "route_change_level": "large",
                "route_factors": _route_factors(),
                "route_reason": "First scene establishes the bird.",
                "identity_conditioning_subject_id": "Bird",
                "primary_visible_character_ids": ["Bird"],
                "interaction_summary": "",
                "spatial_relation": "",
                "framing": "clear single-subject composition",
                "setting_focus": "tree branch",
            }
        ],
    }


def _milo_llm_payload() -> dict:
    return {
        "global": {
            "main_character": "Milo",
            "identity_cues": ["dog"],
            "shared_setting": ["living room"],
            "style_cues": [],
            "characters": [
                {
                    "character_id": "Milo",
                    "age_band": "",
                    "gender_presentation": "dog",
                    "hair_color": "",
                    "hairstyle": "",
                    "skin_tone": "",
                    "body_build": "small dog body",
                    "signature_outfit": "",
                    "signature_accessory": "",
                    "profession_marker": "",
                }
            ],
        },
        "scenes": [
            {
                "scene_id": "SCENE-1",
                "primary_action": "Milo sits on the floor with toys",
                "secondary_elements": ["toys"],
                "generation_prompt": "dog sitting on the floor with toys",
                "scoring_prompt": "Milo sitting",
                "action_prompt": "sitting",
                "continuity_subject_ids": ["Milo"],
                "continuity_route_hint": "text2img",
                "route_change_level": "large",
                "route_factors": _route_factors(body_state_change=True, primary_action_change=True),
                "route_reason": "First scene establishes Milo.",
                "identity_conditioning_subject_id": "Milo",
                "primary_visible_character_ids": ["Milo"],
                "interaction_summary": "",
                "spatial_relation": "",
                "framing": "medium shot",
                "setting_focus": "floor with toys",
            },
            {
                "scene_id": "SCENE-2",
                "primary_action": "Milo rolls over and laughs",
                "secondary_elements": [],
                "generation_prompt": "dog rolling over and laughing",
                "scoring_prompt": "Milo rolling over",
                "action_prompt": "rolling over",
                "continuity_subject_ids": ["Milo"],
                "continuity_route_hint": "img2img",
                "route_change_level": "medium",
                "route_factors": _route_factors(body_state_change=True, primary_action_change=True),
                "route_reason": "Action changes.",
                "identity_conditioning_subject_id": "Milo",
                "primary_visible_character_ids": ["Milo"],
                "interaction_summary": "",
                "spatial_relation": "",
                "framing": "medium shot",
                "setting_focus": "floor with toys",
            },
            {
                "scene_id": "SCENE-3",
                "primary_action": "Milo lies down and rests",
                "secondary_elements": [],
                "generation_prompt": "dog lying down and resting",
                "scoring_prompt": "Milo resting",
                "action_prompt": "resting",
                "continuity_subject_ids": ["Milo"],
                "continuity_route_hint": "img2img",
                "route_change_level": "medium",
                "route_factors": _route_factors(body_state_change=True, primary_action_change=True),
                "route_reason": "Body state changes.",
                "identity_conditioning_subject_id": "Milo",
                "primary_visible_character_ids": ["Milo"],
                "interaction_summary": "",
                "spatial_relation": "",
                "framing": "medium shot",
                "setting_focus": "floor",
            },
        ],
    }


def _ryan_bus_payload() -> dict:
    payload = _llm_payload()
    payload["global"] = {
        "main_character": "Ryan",
        "identity_cues": ["human man"],
        "shared_setting": ["urban bus stop"],
        "style_cues": [],
        "characters": [
            {
                "character_id": "Ryan",
                "age_band": "adult",
                "gender_presentation": "man",
                "hair_color": "",
                "hairstyle": "",
                "skin_tone": "",
                "body_build": "",
                "signature_outfit": "",
                "signature_accessory": "",
                "profession_marker": "",
            }
        ],
    }
    payload["scenes"] = [
        {
            "scene_id": "SCENE-1",
            "primary_action": "Ryan walks quickly toward a bus",
            "secondary_elements": ["bus"],
            "generation_prompt": "Ryan walking quickly toward a bus",
            "scoring_prompt": "Ryan walking",
            "action_prompt": "walking",
            "continuity_subject_ids": ["Ryan"],
            "continuity_route_hint": "text2img",
            "route_change_level": "large",
            "route_factors": _route_factors(body_state_change=True, primary_action_change=True),
            "route_reason": "First scene.",
            "identity_conditioning_subject_id": "Ryan",
            "primary_visible_character_ids": ["Ryan"],
            "interaction_summary": "",
            "spatial_relation": "",
            "framing": "medium shot",
            "setting_focus": "bus stop",
        },
        {
            "scene_id": "SCENE-2",
            "primary_action": "Ryan pauses at the bus door and looks ahead",
            "secondary_elements": ["bus door"],
            "generation_prompt": "Ryan pausing at the bus door",
            "scoring_prompt": "Ryan at bus door",
            "action_prompt": "pausing",
            "continuity_subject_ids": ["Ryan"],
            "continuity_route_hint": "img2img",
            "route_change_level": "medium",
            "route_factors": _route_factors(body_state_change=True, primary_action_change=True),
            "route_reason": "Same subject but action changes.",
            "identity_conditioning_subject_id": "Ryan",
            "primary_visible_character_ids": ["Ryan"],
            "interaction_summary": "",
            "spatial_relation": "",
            "framing": "medium shot",
            "setting_focus": "bus door",
        },
    ]
    return payload


def _emma_city_lights_payload() -> dict:
    payload = _llm_payload()
    payload["global"] = {
        "main_character": "Emma",
        "identity_cues": ["human woman"],
        "shared_setting": ["bridge overlooking city"],
        "style_cues": [],
        "characters": [
            {
                "character_id": "Emma",
                "age_band": "adult",
                "gender_presentation": "woman",
                "hair_color": "",
                "hairstyle": "",
                "skin_tone": "",
                "body_build": "",
                "signature_outfit": "",
                "signature_accessory": "",
                "profession_marker": "",
            }
        ],
    }
    payload["scenes"] = [
        {
            "scene_id": "SCENE-1",
            "primary_action": "stands on bridge",
            "secondary_elements": ["bridge"],
            "generation_prompt": "Emma standing on a bridge looking down",
            "scoring_prompt": "Emma on bridge",
            "action_prompt": "standing on bridge",
            "continuity_subject_ids": ["Emma"],
            "continuity_route_hint": "img2img",
            "route_change_level": "small",
            "route_factors": _route_factors(),
            "route_reason": "Incorrect first scene reason from LLM.",
            "identity_conditioning_subject_id": "Emma",
            "primary_visible_character_ids": ["Emma"],
            "interaction_summary": "",
            "spatial_relation": "",
            "framing": "medium shot",
            "setting_focus": "bridge",
        },
        {
            "scene_id": "SCENE-2",
            "primary_action": "city lights come on",
            "secondary_elements": ["city lights", "night"],
            "generation_prompt": "wide shot of city lights turning on at night",
            "scoring_prompt": "city lights at night",
            "action_prompt": "city lights turning on",
            "continuity_subject_ids": ["Emma"],
            "continuity_route_hint": "img2img",
            "route_change_level": "small",
            "route_factors": _route_factors(same_subject=False, composition_change_needed=True),
            "route_reason": "Environment beat.",
            "identity_conditioning_subject_id": "Emma",
            "primary_visible_character_ids": [],
            "interaction_summary": "",
            "spatial_relation": "",
            "framing": "wide shot",
            "setting_focus": "city skyline from bridge",
        },
        {
            "scene_id": "SCENE-3",
            "primary_action": "smiles at view",
            "secondary_elements": ["view"],
            "generation_prompt": "Emma smiling at the view",
            "scoring_prompt": "Emma smiling",
            "action_prompt": "smiling",
            "continuity_subject_ids": ["Emma"],
            "continuity_route_hint": "img2img",
            "route_change_level": "medium",
            "route_factors": _route_factors(composition_change_needed=True),
            "route_reason": "Emma returns to foreground.",
            "identity_conditioning_subject_id": "Emma",
            "primary_visible_character_ids": ["Emma"],
            "interaction_summary": "",
            "spatial_relation": "",
            "framing": "medium shot",
            "setting_focus": "city view",
        },
    ]
    return payload


def _delayed_character_payload() -> dict:
    payload = _llm_payload()
    payload["global"] = {
        "main_character": "Nina",
        "identity_cues": ["human woman"],
        "shared_setting": ["snowy crowd"],
        "style_cues": [],
        "characters": [
            {
                "character_id": "Nina",
                "age_band": "adult",
                "gender_presentation": "woman",
                "hair_color": "",
                "hairstyle": "",
                "skin_tone": "",
                "body_build": "",
                "signature_outfit": "winter coat",
                "signature_accessory": "",
                "profession_marker": "",
            },
            {
                "character_id": "Leo",
                "age_band": "adult",
                "gender_presentation": "man",
                "hair_color": "",
                "hairstyle": "",
                "skin_tone": "",
                "body_build": "",
                "signature_outfit": "jacket",
                "signature_accessory": "",
                "profession_marker": "",
            },
        ],
    }
    payload["scenes"] = [
        {
            "scene_id": "SCENE-1",
            "primary_action": "stands in snow",
            "secondary_elements": ["snow"],
            "generation_prompt": "Nina standing in the snow",
            "scoring_prompt": "Nina in snow",
            "action_prompt": "standing",
            "continuity_subject_ids": ["Nina"],
            "continuity_route_hint": "text2img",
            "route_change_level": "large",
            "route_factors": _route_factors(),
            "route_reason": "First scene.",
            "identity_conditioning_subject_id": "Nina",
            "primary_visible_character_ids": ["Nina"],
            "interaction_summary": "",
            "spatial_relation": "",
            "framing": "medium shot",
            "setting_focus": "snow",
        },
        {
            "scene_id": "SCENE-2",
            "primary_action": "meets Leo",
            "secondary_elements": ["crowd"],
            "generation_prompt": "Nina meets Leo in a crowd",
            "scoring_prompt": "Nina and Leo in crowd",
            "action_prompt": "meeting",
            "continuity_subject_ids": ["Nina", "Leo"],
            "continuity_route_hint": "img2img",
            "route_change_level": "medium",
            "route_factors": _route_factors(composition_change_needed=False),
            "route_reason": "LLM says same scene.",
            "identity_conditioning_subject_id": None,
            "primary_visible_character_ids": ["Nina", "Leo"],
            "interaction_summary": "Nina meets Leo",
            "spatial_relation": "",
            "framing": "medium two-shot",
            "setting_focus": "crowd",
        },
    ]
    return payload


def _friend_payload() -> dict:
    payload = _llm_payload()
    payload["global"] = {
        "main_character": "Tom",
        "identity_cues": ["human man"],
        "shared_setting": ["urban street"],
        "style_cues": [],
        "characters": [
            {
                "character_id": "Tom",
                "age_band": "adult",
                "gender_presentation": "man",
                "hair_color": "",
                "hairstyle": "",
                "skin_tone": "",
                "body_build": "",
                "signature_outfit": "",
                "signature_accessory": "",
                "profession_marker": "",
            },
            {
                "character_id": "friend",
                "age_band": "adult",
                "gender_presentation": "unknown",
                "hair_color": "",
                "hairstyle": "",
                "skin_tone": "",
                "body_build": "",
                "signature_outfit": "",
                "signature_accessory": "",
                "profession_marker": "",
            },
        ],
    }
    payload["scenes"] = [
        {
            "scene_id": "SCENE-1",
            "primary_action": "runs",
            "secondary_elements": ["street"],
            "generation_prompt": "Tom running down a street",
            "scoring_prompt": "Tom running",
            "action_prompt": "running",
            "continuity_subject_ids": ["Tom"],
            "continuity_route_hint": "text2img",
            "route_change_level": "large",
            "route_factors": _route_factors(),
            "route_reason": "First scene.",
            "identity_conditioning_subject_id": "Tom",
            "primary_visible_character_ids": ["Tom"],
            "interaction_summary": "",
            "spatial_relation": "",
            "framing": "medium shot",
            "setting_focus": "urban street",
        },
        {
            "scene_id": "SCENE-2",
            "primary_action": "meets friend",
            "secondary_elements": ["urban street"],
            "generation_prompt": "Tom meets his friend and keeps going",
            "scoring_prompt": "Tom meets friend",
            "action_prompt": "meeting friend",
            "continuity_subject_ids": ["Tom"],
            "continuity_route_hint": "img2img",
            "route_change_level": "medium",
            "route_factors": _route_factors(composition_change_needed=True),
            "route_reason": "Tom meets an anonymous friend.",
            "identity_conditioning_subject_id": None,
            "primary_visible_character_ids": ["Tom", "friend"],
            "interaction_summary": "Tom meets an adult friend and they continue down the street together",
            "spatial_relation": "side by side on the urban street",
            "framing": "medium two-shot",
            "setting_focus": "urban street",
        },
    ]
    return payload


def test_llm_builder_cache_miss_calls_client_and_writes_cache(tmp_path: Path) -> None:
    client = FakeLLMClient()
    config = _prompt_config(tmp_path)
    builder = LLMAssistedPromptBuilder(config, llm_client=client)

    prompts = builder.build_story_prompts(_story())
    cache_key = build_prompt_cache_key(_story(), config)

    assert client.calls == 1
    assert (tmp_path / "cache" / f"{cache_key}.json").exists()
    assert prompts["SCENE-1"].generation_prompt == "Hero, human person, runs along the path, path"
    assert prompts["SCENE-1"].character_prompt == "Hero, human person"
    assert prompts["SCENE-1"].scoring_prompt == "Hero runs"
    assert prompts["SCENE-1"].action_prompt == "running"
    assert prompts["SCENE-1"].global_context_prompt == "city park, clean storyboard frame, keep the same lighting and palette"
    assert "cat, dog, pet animal" in prompts["SCENE-1"].negative_prompt
    assert "non-human subject" not in prompts["SCENE-1"].negative_prompt
    assert prompts["SCENE-1"].full_prompt


def test_llm_pipeline_metadata_includes_route_hints(tmp_path: Path) -> None:
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient()

    bundle = pipeline.build(_story())

    route_hints = bundle.metadata["scene_route_hints"]
    assert route_hints["SCENE-2"]["continuity_subject_ids"] == ["Hero"]
    assert route_hints["SCENE-2"]["continuity_route_hint"] == "img2img"
    assert route_hints["SCENE-2"]["route_change_level"] == "small"
    assert route_hints["SCENE-2"]["llm_route_change_level"] == "small"
    assert route_hints["SCENE-2"]["route_factors"]["same_subject"] is True
    assert route_hints["SCENE-2"]["identity_conditioning_subject_id"] == "Hero"
    assert route_hints["SCENE-2"]["primary_visible_character_ids"] == ["Hero"]


def test_llm_pipeline_metadata_includes_scene_plans(tmp_path: Path) -> None:
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient()

    bundle = pipeline.build(_story())

    scene_plans = bundle.metadata["scene_plans"]
    assert scene_plans["SCENE-2"]["framing"] == "clear single-person composition"
    assert scene_plans["SCENE-2"]["setting_focus"] == "same park path"
    assert scene_plans["SCENE-2"]["policy"]["visible_character_count"] == 1
    assert scene_plans["SCENE-2"]["policy"]["scene_focus_mode"] == "single_primary"


def test_llm_pipeline_metadata_includes_character_specs(tmp_path: Path) -> None:
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient()

    bundle = pipeline.build(_story())

    character_specs = bundle.metadata["character_specs"]
    assert character_specs["Hero"]["character_id"] == "Hero"
    assert character_specs["Hero"]["age_band"] == "adult"
    assert character_specs["Hero"]["signature_outfit"] == "red jacket"
    assert character_specs["Hero"]["metadata"]["source"] == "llm_assisted"


def test_llm_pipeline_metadata_includes_multiple_character_specs(tmp_path: Path) -> None:
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(_two_character_llm_payload())

    bundle = pipeline.build(_two_character_story())

    character_specs = bundle.metadata["character_specs"]
    assert list(character_specs) == ["Jack", "Sara"]
    assert len(character_specs) == len(set(character_specs))
    assert character_specs["Jack"]["signature_outfit"] == "black jacket"
    assert character_specs["Sara"]["signature_accessory"] == "yellow scarf"


def test_llm_pipeline_accepts_unspecified_identity_conditioning_subject(tmp_path: Path) -> None:
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(_two_character_llm_payload())

    bundle = pipeline.build(_two_character_story())

    scene_plans = bundle.metadata["scene_plans"]
    assert scene_plans["SCENE-1"]["identity_conditioning_subject_id"] is None
    assert scene_plans["SCENE-1"]["primary_visible_character_ids"] == ["Jack", "Sara"]
    assert scene_plans["SCENE-1"]["policy"]["scene_focus_mode"] == "dual_primary"


def test_llm_dual_primary_preserves_specific_scene_fields_and_flags(tmp_path: Path) -> None:
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(_two_character_llm_payload())

    bundle = pipeline.build(_two_character_story())

    scene_plan = bundle.metadata["scene_plans"]["SCENE-1"]
    assert scene_plan["interaction_summary"] == "Jack and Sara talk together"
    assert scene_plan["spatial_relation"] == "Jack and Sara sit side by side on the bench"
    assert scene_plan["framing"] == "clear two-person composition, both characters visible"
    assert scene_plan["setting_focus"] == "park bench"
    assert scene_plan["used_default_interaction_summary"] is False
    assert scene_plan["used_default_spatial_relation"] is False
    assert scene_plan["used_default_framing"] is False
    assert scene_plan["used_default_setting_focus"] is False


def test_llm_dual_primary_applies_safe_defaults_for_missing_scene_fields(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][0]["interaction_summary"] = None
    payload["scenes"][0]["spatial_relation"] = None
    payload["scenes"][0]["framing"] = None
    payload["scenes"][0]["setting_focus"] = None
    payload["scenes"][0]["generation_prompt"] = "medium two-shot of Jack and Sara sitting side by side on a park bench"
    payload["scenes"][0]["secondary_elements"] = ["park bench"]
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(payload)

    bundle = pipeline.build(_two_character_rich_story())

    scene_plan = bundle.metadata["scene_plans"]["SCENE-1"]
    assert scene_plan["interaction_summary"] == "Jack and Sara sit side by side on a park bench and talk."
    assert scene_plan["spatial_relation"] == "sitting side by side on a park bench"
    assert scene_plan["framing"] == "medium two-shot"
    assert scene_plan["setting_focus"] == "park bench"
    assert scene_plan["used_default_interaction_summary"] is False
    assert scene_plan["used_default_spatial_relation"] is False
    assert scene_plan["used_default_framing"] is False
    assert scene_plan["used_default_setting_focus"] is False


def test_llm_dual_primary_records_default_flags_when_no_specific_cues_exist(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][0]["interaction_summary"] = None
    payload["scenes"][0]["spatial_relation"] = None
    payload["scenes"][0]["framing"] = None
    payload["scenes"][0]["setting_focus"] = None
    payload["scenes"][0]["primary_action"] = ""
    payload["scenes"][0]["secondary_elements"] = []
    payload["scenes"][0]["generation_prompt"] = "Jack and Sara"
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(payload),
    )

    builder.build_story_prompts(_sparse_two_character_story())

    scene_plan = builder.metadata()["scene_plans"]["SCENE-1"]
    assert scene_plan["interaction_summary"] == "both characters are present in the same scene"
    assert scene_plan["spatial_relation"] == "jack on the left, sara on the right"
    assert scene_plan["framing"] == "medium two-shot, both characters visible"
    assert scene_plan["setting_focus"] is None
    assert scene_plan["used_default_interaction_summary"] is True
    assert scene_plan["used_default_spatial_relation"] is True
    assert scene_plan["used_default_framing"] is True
    assert scene_plan["used_default_setting_focus"] is True


def test_llm_dual_primary_rejects_identity_conditioning_subject(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][0]["identity_conditioning_subject_id"] = "Jack"
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, fallback=False, cache_enabled=False),
        llm_client=FakeLLMClient(payload),
    )

    with pytest.raises(LLMPromptError, match="must be null for dual_primary"):
        builder.build_story_prompts(_two_character_story())


def test_llm_dual_primary_prompt_uses_local_identity_snippets_and_scene_plan(tmp_path: Path) -> None:
    config = _prompt_config(tmp_path, cache_enabled=False)
    config["generation_max_words"] = 40
    config["generation_max_chars"] = 400
    config["dual_primary_generation_max_words"] = 48
    config["dual_primary_generation_max_chars"] = 420
    builder = LLMAssistedPromptBuilder(
        config,
        llm_client=FakeLLMClient(_two_character_llm_payload()),
    )

    prompts = builder.build_story_prompts(_two_character_story())

    assert "Jack, adult, man, black, short hair, black jacket" in prompts["SCENE-1"].character_prompt
    assert "Sara, adult, woman, brown, shoulder-length hair, yellow scarf" in prompts["SCENE-1"].character_prompt
    assert "Jack and Sara talk together" in prompts["SCENE-1"].local_prompt
    assert "clear two-person composition" in prompts["SCENE-1"].local_prompt
    assert prompts["SCENE-1"].generation_prompt.startswith("Jack is an adult man")
    assert "Sara is an adult woman" in prompts["SCENE-1"].generation_prompt
    assert "clear two-person composition, both characters visible" in prompts["SCENE-1"].generation_prompt
    assert "Jack, adult, man, black, short hair, black jacket, Sara" not in prompts["SCENE-1"].generation_prompt
    assert "cafe table" in prompts["SCENE-2"].generation_prompt


def test_llm_dual_primary_generation_prompt_replaces_pronoun_with_names(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][1]["interaction_summary"] = "They continue talking in a cafe."
    config = _prompt_config(tmp_path, cache_enabled=False)
    config["generation_max_words"] = 40
    config["generation_max_chars"] = 260
    config["dual_primary_generation_max_words"] = 60
    config["dual_primary_generation_max_chars"] = 420
    builder = LLMAssistedPromptBuilder(config, llm_client=FakeLLMClient(payload))

    prompts = builder.build_story_prompts(_two_character_story())

    assert "Jack and Sara continue talking in a cafe." in prompts["SCENE-2"].generation_prompt
    assert "They continue talking in a cafe." not in prompts["SCENE-2"].generation_prompt


def test_llm_dual_primary_generation_prompt_formats_setting_focus_naturally(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][0]["setting_focus"] = "park"
    payload["scenes"][1]["setting_focus"] = "cafe"
    config = _prompt_config(tmp_path, cache_enabled=False)
    config["dual_primary_generation_max_words"] = 60
    config["dual_primary_generation_max_chars"] = 420
    builder = LLMAssistedPromptBuilder(config, llm_client=FakeLLMClient(payload))

    prompts = builder.build_story_prompts(_two_character_story())

    assert prompts["SCENE-1"].generation_prompt.endswith("in a park.")
    assert prompts["SCENE-2"].generation_prompt.endswith("in a cafe.")


def test_llm_dual_primary_generation_prompt_respects_dual_limits(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][0]["interaction_summary"] = "Jack and Sara talk quietly while planning their next move together"
    payload["scenes"][0]["spatial_relation"] = "Jack sits on the left and Sara sits on the right with clear separation"
    payload["scenes"][0]["framing"] = "medium two-shot with both characters fully visible"
    payload["scenes"][0]["setting_focus"] = "park bench near a fountain with trees behind them"
    config = _prompt_config(tmp_path, cache_enabled=False)
    config["dual_primary_generation_max_words"] = 20
    config["dual_primary_generation_max_chars"] = 160
    builder = LLMAssistedPromptBuilder(config, llm_client=FakeLLMClient(payload))

    prompts = builder.build_story_prompts(_two_character_story())

    assert len(prompts["SCENE-1"].generation_prompt.split()) <= 20
    assert len(prompts["SCENE-1"].generation_prompt) <= 160
    assert prompts["SCENE-2"].generation_prompt.startswith("Jack is")


def test_llm_identity_conditioning_subject_must_match_character_specs(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][1]["identity_conditioning_subject_id"] = "Missing"
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, fallback=False, cache_enabled=False),
        llm_client=FakeLLMClient(payload),
    )

    with pytest.raises(LLMPromptError, match="identity_conditioning_subject_id"):
        builder.build_story_prompts(_two_character_story())


def test_llm_primary_visible_characters_must_match_character_specs(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][1]["primary_visible_character_ids"] = ["Jack", "Missing"]
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, fallback=False, cache_enabled=False),
        llm_client=FakeLLMClient(payload),
    )

    with pytest.raises(LLMPromptError, match="primary_visible_character_ids"):
        builder.build_story_prompts(_two_character_story())


def test_llm_main_character_must_align_with_character_specs(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["global"]["main_character"] = "Missing"
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, fallback=False, cache_enabled=False),
        llm_client=FakeLLMClient(payload),
    )

    with pytest.raises(LLMPromptError, match="main_character"):
        builder.build_story_prompts(_two_character_story())


def test_llm_non_human_character_does_not_get_human_negative_suppression(tmp_path: Path) -> None:
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(_bird_llm_payload()),
    )

    prompts = builder.build_story_prompts(_bird_story())

    assert not prompts["SCENE-1"].generation_prompt.startswith("human person")
    assert prompts["SCENE-1"].negative_prompt == "blurry"


def test_llm_named_subject_with_human_pronouns_does_not_infer_animal_from_actions(tmp_path: Path) -> None:
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(_milo_llm_payload()),
    )

    prompts = builder.build_story_prompts(_milo_story())
    character_specs = builder.metadata()["character_specs"]

    assert prompts["SCENE-1"].character_prompt == "Milo, human man"
    assert "dog" not in prompts["SCENE-1"].generation_prompt.lower()
    assert "dog" not in prompts["SCENE-2"].generation_prompt.lower()
    assert "dog" not in prompts["SCENE-3"].generation_prompt.lower()
    assert character_specs["Milo"]["gender_presentation"] == "man"
    assert "dog" not in (character_specs["Milo"]["body_build"] or "").lower()
    assert "cat, dog, pet animal" in prompts["SCENE-1"].negative_prompt


def test_llm_lightweight_identity_uses_llm_age_before_pronoun_fallback(tmp_path: Path) -> None:
    payload = _milo_llm_payload()
    payload["global"]["identity_cues"] = ["human boy"]
    payload["global"]["characters"][0].update(
        {
            "age_band": "child",
            "gender_presentation": "male",
            "body_build": "slim",
        }
    )
    payload["scenes"][0]["generation_prompt"] = "Milo sitting on the floor with toys"
    payload["scenes"][1]["generation_prompt"] = "Milo rolling over and laughing"
    payload["scenes"][2]["generation_prompt"] = "Milo lying down and resting"
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(payload),
    )

    prompts = builder.build_story_prompts(_milo_story())

    assert prompts["SCENE-1"].character_prompt == "Milo, human boy"
    assert prompts["SCENE-1"].generation_prompt.startswith("Milo, human boy")
    assert "human man" not in prompts["SCENE-1"].generation_prompt


def test_llm_no_character_scene_omits_identity_from_generation_prompt(tmp_path: Path) -> None:
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(_emma_city_lights_payload()),
    )

    prompts = builder.build_story_prompts(_emma_city_lights_story())
    scene_plan = builder.metadata()["scene_plans"]["SCENE-2"]
    route_hint = builder.metadata()["scene_route_hints"]["SCENE-2"]
    generation_prompt = prompts["SCENE-2"].generation_prompt.lower()

    assert scene_plan["policy"]["visible_character_count"] == 0
    assert scene_plan["identity_conditioning_subject_id"] is None
    assert route_hint["identity_conditioning_subject_id"] is None
    assert prompts["SCENE-2"].character_prompt == ""
    assert "emma" not in generation_prompt
    assert "human woman" not in generation_prompt
    assert "city lights" in generation_prompt
    assert "emma" not in prompts["SCENE-2"].full_prompt.lower()


def test_llm_route_metadata_forces_text2img_for_composition_change(tmp_path: Path) -> None:
    payload = _llm_payload()
    payload["scenes"][1]["continuity_route_hint"] = "img2img"
    payload["scenes"][1]["route_factors"] = _route_factors(composition_change_needed=True)
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(payload)

    bundle = pipeline.build(_story())
    route_hint = bundle.metadata["scene_route_hints"]["SCENE-2"]

    assert route_hint["continuity_route_hint"] == "text2img"
    assert route_hint["route_hint_adjustment_reason"] == "composition_change_needed"


def test_llm_route_metadata_forces_text2img_when_new_character_appears(tmp_path: Path) -> None:
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(_delayed_character_payload()),
    )

    builder.build_story_prompts(_delayed_character_story())
    route_hint = builder.metadata()["scene_route_hints"]["SCENE-2"]

    assert route_hint["continuity_route_hint"] == "text2img"
    assert route_hint["route_hint_adjustment_reason"] == "visible_character_change"


def test_llm_route_metadata_forces_text2img_for_body_action_transitions(tmp_path: Path) -> None:
    payload = _ryan_bus_payload()
    payload["scenes"][1]["generation_prompt"] = "Ryan gets inside and sits by the window"
    payload["scenes"][1]["primary_action"] = "Ryan gets inside and sits by the window"
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(payload),
    )

    builder.build_story_prompts(_ryan_bus_story())
    route_hint = builder.metadata()["scene_route_hints"]["SCENE-2"]

    assert route_hint["continuity_route_hint"] == "text2img"
    assert route_hint["route_hint_adjustment_reason"] == "body_action_transition"


def test_llm_first_scene_route_metadata_is_initial_setup(tmp_path: Path) -> None:
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(_emma_city_lights_payload())

    bundle = pipeline.build(_emma_city_lights_story())
    route_hint = bundle.metadata["scene_route_hints"]["SCENE-1"]

    assert route_hint["continuity_route_hint"] == "text2img"
    assert route_hint["route_change_level"] == "large"
    assert route_hint["route_reason"] == "Initial scene setup"


def test_llm_anonymous_friend_prompt_avoids_unknown_identity_text(tmp_path: Path) -> None:
    config = _prompt_config(tmp_path, cache_enabled=False)
    config["generation_max_words"] = 40
    config["generation_max_chars"] = 260
    config["dual_primary_generation_max_words"] = 60
    config["dual_primary_generation_max_chars"] = 420
    builder = LLMAssistedPromptBuilder(config, llm_client=FakeLLMClient(_friend_payload()))

    prompts = builder.build_story_prompts(_friend_story())
    scene_plan = builder.metadata()["scene_plans"]["SCENE-2"]
    route_hint = builder.metadata()["scene_route_hints"]["SCENE-2"]
    generation_prompt = prompts["SCENE-2"].generation_prompt.lower()

    assert prompts["SCENE-2"].character_prompt == "Tom, human man"
    assert "friend is an adult unknown" not in generation_prompt
    assert "unknown" not in generation_prompt
    assert "unknown" not in prompts["SCENE-2"].full_prompt.lower()
    assert "friend, adult" not in prompts["SCENE-2"].full_prompt.lower()
    assert "tom meets an adult friend" in generation_prompt
    assert scene_plan["primary_visible_character_ids"] == ["Tom"]
    assert scene_plan["identity_conditioning_subject_id"] == "Tom"
    assert scene_plan["continuity_subject_ids"] == ["Tom"]
    assert route_hint["identity_conditioning_subject_id"] == "Tom"
    assert route_hint["continuity_subject_ids"] == ["Tom"]


def test_llm_route_factors_adjust_small_to_medium(tmp_path: Path) -> None:
    payload = _llm_payload()
    payload["scenes"][1]["route_change_level"] = "small"
    payload["scenes"][1]["route_factors"] = _route_factors(primary_action_change=True)
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(payload)

    bundle = pipeline.build(_story())
    route_hint = bundle.metadata["scene_route_hints"]["SCENE-2"]

    assert route_hint["llm_route_change_level"] == "small"
    assert route_hint["route_change_level"] == "medium"
    assert route_hint["route_level_adjustment_reason"] == "small_inconsistent_with_route_factors"


def test_llm_route_factors_adjust_setting_and_composition_change_to_large(tmp_path: Path) -> None:
    payload = _llm_payload()
    payload["scenes"][1]["route_change_level"] = "small"
    payload["scenes"][1]["route_factors"] = _route_factors(same_setting=False, composition_change_needed=True)
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(payload)

    bundle = pipeline.build(_story())
    route_hint = bundle.metadata["scene_route_hints"]["SCENE-2"]

    assert route_hint["llm_route_change_level"] == "small"
    assert route_hint["route_change_level"] == "large"
    assert route_hint["route_level_adjustment_reason"] == "setting_change_and_composition_change_needed"


def test_llm_builder_cache_hit_skips_client(tmp_path: Path) -> None:
    config = _prompt_config(tmp_path)
    first_client = FakeLLMClient()
    LLMAssistedPromptBuilder(config, llm_client=first_client).build_story_prompts(_story())

    second_client = FakeLLMClient(error=RuntimeError("should not call"))
    prompts = LLMAssistedPromptBuilder(config, llm_client=second_client).build_story_prompts(_story())

    assert first_client.calls == 1
    assert second_client.calls == 0
    assert prompts["SCENE-2"].generation_prompt == "Hero, human person, stops on the path, path"


def test_llm_payload_missing_characters_falls_back_when_enabled(tmp_path: Path) -> None:
    payload = _llm_payload()
    del payload["global"]["characters"]
    client = FakeLLMClient(payload)
    builder = LLMAssistedPromptBuilder(_prompt_config(tmp_path, cache_enabled=False), llm_client=client)

    prompts = builder.build_story_prompts(_story())

    assert prompts["SCENE-1"].character_prompt.startswith("main subject: Hero")
    assert builder.last_character_specs["Hero"]["metadata"]["source"] == "rule_based"


def test_llm_payload_missing_characters_raises_when_fallback_disabled(tmp_path: Path) -> None:
    payload = _llm_payload()
    del payload["global"]["characters"]
    client = FakeLLMClient(payload)
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, fallback=False, cache_enabled=False),
        llm_client=client,
    )

    with pytest.raises(LLMPromptError, match="global.characters"):
        builder.build_story_prompts(_story())


def test_prompt_cache_key_changes_when_story_or_model_changes(tmp_path: Path) -> None:
    config = _prompt_config(tmp_path)
    story_key = build_prompt_cache_key(_story("Hero runs."), config)
    other_story_key = build_prompt_cache_key(_story("Different story."), config)
    other_config = _prompt_config(tmp_path)
    other_config["llm"]["model"] = "gpt-4o-mini-2024-07-18"
    other_model_key = build_prompt_cache_key(_story("Hero runs."), other_config)

    assert story_key != other_story_key
    assert story_key != other_model_key


def test_llm_builder_loads_artifact_without_calling_client(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps({"validated_output": _llm_payload()}), encoding="utf-8")
    config = _prompt_config(tmp_path)
    config["artifact"]["path"] = str(artifact_path)
    client = FakeLLMClient(error=RuntimeError("should not call"))

    prompts = LLMAssistedPromptBuilder(config, llm_client=client).build_story_prompts(_story())

    assert client.calls == 0
    assert prompts["SCENE-1"].generation_prompt == "human person, Hero, same red jacket, Hero runs"


def test_llm_builder_exports_artifact_after_api_success(tmp_path: Path) -> None:
    config = _prompt_config(tmp_path)
    config["artifact"]["export_enabled"] = True
    config["artifact"]["export_dir"] = str(tmp_path / "exports")

    LLMAssistedPromptBuilder(config, llm_client=FakeLLMClient()).build_story_prompts(_story())

    exported = list((tmp_path / "exports").glob("*.json"))
    assert len(exported) == 1
    payload = json.loads(exported[0].read_text(encoding="utf-8"))
    assert payload["validated_output"]["global"]["main_character"] == "Hero"
    assert payload["validated_output"]["global"]["characters"][0]["character_id"] == "Hero"
    assert "OPENAI_API_KEY" not in exported[0].read_text(encoding="utf-8")
    assert "api_key" not in json.dumps(payload).lower()


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"not": "json schema"},
        {**_llm_payload(), "scenes": [_llm_payload()["scenes"][0]]},
        {**_llm_payload(), "scenes": [{**_llm_payload()["scenes"][0], "scene_id": "WRONG"}, _llm_payload()["scenes"][1]]},
        {
            **_llm_payload(),
            "scenes": [{**_llm_payload()["scenes"][0], "generation_prompt": ""}, _llm_payload()["scenes"][1]],
        },
        {
            **_llm_payload(),
            "scenes": [
                {**_llm_payload()["scenes"][0], "continuity_route_hint": "bad_hint"},
                _llm_payload()["scenes"][1],
            ],
        },
        {
            **_llm_payload(),
            "scenes": [
                {**_llm_payload()["scenes"][0], "route_change_level": "tiny"},
                _llm_payload()["scenes"][1],
            ],
        },
        {
            **_llm_payload(),
            "scenes": [
                {**_llm_payload()["scenes"][0], "route_factors": "missing"},
                _llm_payload()["scenes"][1],
            ],
        },
    ],
)
def test_llm_builder_falls_back_on_invalid_payload(tmp_path: Path, bad_payload: dict) -> None:
    builder = LLMAssistedPromptBuilder(_prompt_config(tmp_path), llm_client=FakeLLMClient(payload=bad_payload))

    prompts = builder.build_story_prompts(_story())

    assert prompts["SCENE-1"].generation_prompt == "Hero, runs, cinematic illustration"


def test_llm_builder_raises_when_fallback_disabled(tmp_path: Path) -> None:
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, fallback=False),
        llm_client=FakeLLMClient(error=RuntimeError("api failed")),
    )

    with pytest.raises(LLMPromptError, match="api failed"):
        builder.build_story_prompts(_story())


def test_llm_builder_trims_long_prompts(tmp_path: Path) -> None:
    payload = _llm_payload()
    payload["scenes"][0]["generation_prompt"] = "one two three four five six seven eight nine ten"
    payload["scenes"][0]["scoring_prompt"] = "one two three four five six"
    builder = LLMAssistedPromptBuilder(_prompt_config(tmp_path, cache_enabled=False), llm_client=FakeLLMClient(payload))

    prompts = builder.build_story_prompts(_story())

    assert prompts["SCENE-1"].generation_prompt == "Hero, human person, one two three four five"
    assert prompts["SCENE-1"].scoring_prompt == "one two three four five"


def test_llm_builder_normalizes_prompt_instruction_phrasing(tmp_path: Path) -> None:
    payload = _llm_payload()
    payload["scenes"][0]["generation_prompt"] = "Illustrate Hero running through the park"
    payload["scenes"][0]["scoring_prompt"] = "Does the image show Hero running through the park?"
    payload["scenes"][0]["action_prompt"] = "Show Hero running"
    builder = LLMAssistedPromptBuilder(_prompt_config(tmp_path, cache_enabled=False), llm_client=FakeLLMClient(payload))

    prompts = builder.build_story_prompts(_story())

    assert prompts["SCENE-1"].generation_prompt == "Hero, human person, running through the park, path"
    assert prompts["SCENE-1"].scoring_prompt == "Hero running through the park"
    assert prompts["SCENE-1"].action_prompt == "Hero running"


def test_llm_single_primary_generation_prompt_removes_duplicate_leading_name(tmp_path: Path) -> None:
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(_ryan_bus_payload()),
    )

    prompts = builder.build_story_prompts(_ryan_bus_story())

    assert prompts["SCENE-1"].generation_prompt == "Ryan, human man, walking quickly toward a bus"
    assert "Ryan, human man, Ryan walking" not in prompts["SCENE-1"].generation_prompt


def test_llm_local_prompt_prefers_resolved_text_over_pronoun_scene_text(tmp_path: Path) -> None:
    builder = LLMAssistedPromptBuilder(
        _prompt_config(tmp_path, cache_enabled=False),
        llm_client=FakeLLMClient(_ryan_bus_payload()),
    )

    prompts = builder.build_story_prompts(_ryan_bus_story())

    assert "Ryan pauses at the bus door" in prompts["SCENE-2"].local_prompt
    assert "He pauses at the door" not in prompts["SCENE-2"].local_prompt
    assert "He pauses at the door" not in prompts["SCENE-2"].full_prompt


def test_llm_builder_preserves_explicit_human_identity_without_duplicate_prefix(tmp_path: Path) -> None:
    payload = _llm_payload()
    payload["global"]["identity_cues"] = ["human woman", "long brown hair", "blue pajamas"]
    payload["scenes"][0]["generation_prompt"] = "Lily gazing out the window"
    builder = LLMAssistedPromptBuilder(_prompt_config(tmp_path, cache_enabled=False), llm_client=FakeLLMClient(payload))

    prompts = builder.build_story_prompts(_story())

    assert prompts["SCENE-1"].character_prompt == "Hero, human woman"
    assert prompts["SCENE-1"].generation_prompt.startswith("Hero, human woman")
    assert "long brown hair" not in prompts["SCENE-1"].generation_prompt
    assert "blue pajamas" not in prompts["SCENE-1"].generation_prompt
