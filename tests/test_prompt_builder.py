from storygen.prompt_builder import PromptBuilder
from storygen.types import Scene, Story


def test_prompt_builder_builds_separated_prompt_fields() -> None:
    story = Story(
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
    builder = PromptBuilder(
        {
            "style_prompt": "cinematic illustration",
            "character_prefix": "recurring characters:",
            "global_context_prefix": "shared story context:",
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[0])

    assert prompt_spec.style_prompt == "cinematic illustration"
    assert prompt_spec.character_prompt == "recurring characters: Hero"
    assert prompt_spec.global_context_prompt == "shared story context: Hero"
    assert prompt_spec.local_prompt == "Hero runs."
    assert "cinematic illustration" in prompt_spec.full_prompt
    assert prompt_spec.negative_prompt == "blurry"
