import json
from pathlib import Path

import pytest

from storygen.llm_assisted_prompt_builder import LLMAssistedPromptBuilder, LLMPromptError
from storygen.llm_client import LLMResponse
from storygen.prompt_cache import build_prompt_cache_key
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
            "builder_version": "llm_assisted_v3",
            "fallback_to_rule_based": fallback,
        },
    }


def _llm_payload() -> dict:
    return {
        "global": {
            "main_character": "Hero",
            "identity_cues": ["same red jacket"],
            "shared_setting": ["city park"],
            "style_cues": ["cinematic illustration", "clean storyboard frame"],
        },
        "scenes": [
            {
                "scene_id": "SCENE-1",
                "primary_action": "runs",
                "secondary_elements": ["path"],
                "generation_prompt": "Hero runs along the path",
                "scoring_prompt": "Hero runs",
                "action_prompt": "running",
            },
            {
                "scene_id": "SCENE-2",
                "primary_action": "stops",
                "secondary_elements": ["path"],
                "generation_prompt": "Hero stops on the path",
                "scoring_prompt": "Hero stops",
                "action_prompt": "stopping",
            },
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
    assert "cat, dog, animal, pet, non-human subject" in prompts["SCENE-1"].negative_prompt
    assert prompts["SCENE-1"].full_prompt


def test_llm_builder_cache_hit_skips_client(tmp_path: Path) -> None:
    config = _prompt_config(tmp_path)
    first_client = FakeLLMClient()
    LLMAssistedPromptBuilder(config, llm_client=first_client).build_story_prompts(_story())

    second_client = FakeLLMClient(error=RuntimeError("should not call"))
    prompts = LLMAssistedPromptBuilder(config, llm_client=second_client).build_story_prompts(_story())

    assert first_client.calls == 1
    assert second_client.calls == 0
    assert prompts["SCENE-2"].generation_prompt == "human person, Hero, same red jacket, Hero stops"


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
