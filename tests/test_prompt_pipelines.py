import pytest

from storygen.prompt_pipelines import ApiPromptPipeline, LLMAssistedPromptPipeline, RuleBasedPromptPipeline, build_prompt_pipeline
from storygen.parser import parse_story_file
from storygen.types import Scene, Story


def _story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="",
        scenes=[
            Scene("SCENE-1", 0, "<Hero> runs.", "Hero runs.", ["Hero"]),
            Scene("SCENE-2", 1, "<Hero> stops.", "Hero stops.", ["Hero"]),
        ],
        all_entities=["Hero"],
        recurring_entities=["Hero"],
        entity_to_scene_ids={"Hero": ["SCENE-1", "SCENE-2"]},
    )


def _multi_character_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="",
        scenes=[
            Scene("SCENE-1", 0, "<Jack> and <Sara> sit.", "Jack and Sara sit.", ["Jack", "Sara"]),
            Scene("SCENE-2", 1, "They talk.", "They talk.", []),
        ],
        all_entities=["Jack", "Sara"],
        recurring_entities=[],
        entity_to_scene_ids={"Jack": ["SCENE-1"], "Sara": ["SCENE-1"]},
    )


def _no_entity_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="",
        scenes=[Scene("SCENE-1", 0, "A quiet room.", "A quiet room.", [])],
        all_entities=[],
        recurring_entities=[],
        entity_to_scene_ids={},
    )


def _prompt_config(pipeline: str = "rule_based") -> dict:
    return {
        "pipeline": pipeline,
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
        "generation_max_words": 28,
        "generation_max_chars": 220,
        "generation_template": "{subject}, {action}{setting_clause}{style_clause}",
        "scoring_template": "{subject}, {action}{setting_clause}",
        "scoring_include_style": False,
        "scoring_include_global_context": False,
        "scoring_max_words": 20,
        "scoring_max_chars": 160,
        "action_emphasis_map": {"runs": "active running pose"},
        "quality_suffix": "clean composition",
        "negative_prompt": "blurry",
        "cache": {"enabled": False, "cache_dir": ".cache/prompt_builder"},
        "artifact": {"path": None, "export_enabled": False, "export_dir": "prompt_artifacts/llm_assisted_v6"},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-2024-08-06",
            "api_key_env": "OPENAI_API_KEY",
            "temperature": 0.0,
            "max_output_tokens": 800,
            "timeout_seconds": 30,
            "schema_version": "v1",
            "builder_version": "llm_assisted_v6",
            "fallback_to_rule_based": True,
        },
    }


def test_rule_based_prompt_pipeline_builds_scene_prompt_bundle() -> None:
    pipeline = build_prompt_pipeline(_prompt_config())

    bundle = pipeline.build(_story())

    assert isinstance(pipeline, RuleBasedPromptPipeline)
    assert set(bundle.scene_prompts) == {"SCENE-1", "SCENE-2"}
    assert bundle.story_prompt is None
    assert bundle.metadata["pipeline"] == "rule_based"
    assert bundle.metadata["character_specs"]["Hero"]["metadata"]["source"] == "rule_based"


def test_rule_based_character_specs_empty_when_story_has_no_entities() -> None:
    pipeline = build_prompt_pipeline(_prompt_config())

    bundle = pipeline.build(_no_entity_story())

    assert bundle.metadata["character_specs"] == {}


def test_rule_based_character_specs_include_all_entities_when_no_recurring_entities() -> None:
    pipeline = build_prompt_pipeline(_prompt_config())

    bundle = pipeline.build(_multi_character_story())

    assert list(bundle.metadata["character_specs"]) == ["Jack", "Sara"]
    assert bundle.metadata["character_specs"]["Jack"]["metadata"]["scene_ids"] == ["SCENE-1"]
    assert bundle.metadata["character_specs"]["Sara"]["metadata"]["scene_ids"] == ["SCENE-1"]


def test_rule_based_character_specs_cover_test_set_06_multi_character_story() -> None:
    pipeline = build_prompt_pipeline(_prompt_config())

    bundle = pipeline.build(parse_story_file("test_set/06.txt"))

    assert list(bundle.metadata["character_specs"]) == ["Jack", "Sara"]


def test_llm_assisted_prompt_pipeline_is_selected() -> None:
    pipeline = build_prompt_pipeline(_prompt_config("llm_assisted"))

    assert isinstance(pipeline, LLMAssistedPromptPipeline)
    assert pipeline.metadata()["implemented"] is True


def test_api_prompt_pipeline_is_alias_for_llm_assisted() -> None:
    pipeline = build_prompt_pipeline(_prompt_config("api"))

    assert isinstance(pipeline, ApiPromptPipeline)
    assert isinstance(pipeline, LLMAssistedPromptPipeline)


def test_unknown_prompt_pipeline_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported prompt pipeline"):
        build_prompt_pipeline(_prompt_config("missing"))
