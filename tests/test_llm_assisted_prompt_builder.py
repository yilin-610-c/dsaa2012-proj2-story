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


def _bird_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="Bird flies.",
        scenes=[Scene("SCENE-1", 0, "<Bird> flies.", "Bird flies.", ["Bird"])],
        all_entities=["Bird"],
        recurring_entities=[],
        entity_to_scene_ids={"Bird": ["SCENE-1"]},
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
            "builder_version": "llm_assisted_v7",
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


def test_llm_builder_cache_miss_calls_client_and_writes_cache(tmp_path: Path) -> None:
    client = FakeLLMClient()
    config = _prompt_config(tmp_path)
    builder = LLMAssistedPromptBuilder(config, llm_client=client)

    prompts = builder.build_story_prompts(_story())
    cache_key = build_prompt_cache_key(_story(), config)

    assert client.calls == 1
    assert (tmp_path / "cache" / f"{cache_key}.json").exists()
    assert prompts["SCENE-1"].generation_prompt == "human person, Hero, same red jacket, Hero runs"
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


def test_llm_dual_primary_applies_safe_defaults_for_missing_scene_fields(tmp_path: Path) -> None:
    payload = _two_character_llm_payload()
    payload["scenes"][0]["interaction_summary"] = None
    payload["scenes"][0]["spatial_relation"] = None
    payload["scenes"][0]["framing"] = None
    pipeline = build_prompt_pipeline(_prompt_config(tmp_path, cache_enabled=False), event_logger=None)
    pipeline.builder.llm_client = FakeLLMClient(payload)

    bundle = pipeline.build(_two_character_story())

    scene_plan = bundle.metadata["scene_plans"]["SCENE-1"]
    assert scene_plan["interaction_summary"] == "both characters are present in the same scene"
    assert scene_plan["spatial_relation"] == "the two characters are clearly separated and not merged"
    assert scene_plan["framing"] == "clear two-person composition, both characters visible"


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
    builder = LLMAssistedPromptBuilder(
        config,
        llm_client=FakeLLMClient(_two_character_llm_payload()),
    )

    prompts = builder.build_story_prompts(_two_character_story())

    assert "Jack, adult, man, black, short hair, black jacket" in prompts["SCENE-1"].character_prompt
    assert "Sara, adult, woman, brown, shoulder-length hair, yellow scarf" in prompts["SCENE-1"].character_prompt
    assert "Jack and Sara talk together" in prompts["SCENE-1"].local_prompt
    assert "clear two-person composition" in prompts["SCENE-1"].local_prompt


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
    assert prompts["SCENE-2"].generation_prompt == "human person, Hero, same red jacket, Hero stops"


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

    assert prompts["SCENE-1"].generation_prompt == "human person, Hero, same red jacket, one two"
    assert prompts["SCENE-1"].scoring_prompt == "one two three four five"


def test_llm_builder_normalizes_prompt_instruction_phrasing(tmp_path: Path) -> None:
    payload = _llm_payload()
    payload["scenes"][0]["generation_prompt"] = "Illustrate Hero running through the park"
    payload["scenes"][0]["scoring_prompt"] = "Does the image show Hero running through the park?"
    payload["scenes"][0]["action_prompt"] = "Show Hero running"
    builder = LLMAssistedPromptBuilder(_prompt_config(tmp_path, cache_enabled=False), llm_client=FakeLLMClient(payload))

    prompts = builder.build_story_prompts(_story())

    assert prompts["SCENE-1"].generation_prompt == "human person, Hero, same red jacket, Hero running"
    assert prompts["SCENE-1"].scoring_prompt == "Hero running through the park"
    assert prompts["SCENE-1"].action_prompt == "Hero running"


def test_llm_builder_preserves_explicit_human_identity_without_duplicate_prefix(tmp_path: Path) -> None:
    payload = _llm_payload()
    payload["global"]["identity_cues"] = ["human woman", "long brown hair", "blue pajamas"]
    payload["scenes"][0]["generation_prompt"] = "Lily gazing out the window"
    builder = LLMAssistedPromptBuilder(_prompt_config(tmp_path, cache_enabled=False), llm_client=FakeLLMClient(payload))

    prompts = builder.build_story_prompts(_story())

    assert prompts["SCENE-1"].character_prompt == "Hero, human woman, long brown hair, blue pajamas"
    assert prompts["SCENE-1"].generation_prompt.startswith("Hero, human woman, long brown hair")
