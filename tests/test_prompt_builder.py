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
            "action_emphasis_map": {"runs": "active running pose"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[0])

    assert prompt_spec.style_prompt == "cinematic illustration"
    assert prompt_spec.character_prompt == "main subject: Hero, same person across all scenes"
    assert prompt_spec.global_context_prompt == "shared story context: Hero, keep the same lighting and palette"
    assert prompt_spec.local_prompt == "Hero runs., key action: active running pose, keep the pose easy to read"
    assert "cinematic illustration" in prompt_spec.full_prompt
    assert prompt_spec.negative_prompt == "blurry"


def test_prompt_builder_reuses_primary_entity_for_pronoun_only_scenes() -> None:
    story = Story(
        source_path="story.txt",
        raw_text="",
        scenes=[
            Scene(
                "SCENE-1",
                0,
                "<Lily> makes breakfast in the kitchen.",
                "Lily makes breakfast in the kitchen.",
                ["Lily"],
            ),
            Scene(
                "SCENE-2",
                1,
                "She looks out the window quietly.",
                "She looks out the window quietly.",
                [],
            ),
        ],
        all_entities=["Lily"],
        recurring_entities=["Lily"],
        entity_to_scene_ids={"Lily": ["SCENE-1", "SCENE-2"]},
    )
    builder = PromptBuilder(
        {
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
            "action_emphasis_map": {"looks out": "looking out through the window"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[1])

    assert prompt_spec.character_prompt == "main subject: Lily, same person across all scenes"
    assert prompt_spec.global_context_prompt == (
        "shared story context: Lily, recurring setting: the kitchen, keep the same lighting and palette"
    )
    assert prompt_spec.local_prompt == (
        "Lily looks out the window quietly., key action: looking out through the window, keep the pose easy to read"
    )


def test_prompt_builder_uses_animal_continuity_rules() -> None:
    story = Story(
        source_path="story.txt",
        raw_text="",
        scenes=[
            Scene("SCENE-1", 0, "<Dog> runs across a field.", "Dog runs across a field.", ["Dog"]),
            Scene("SCENE-2", 1, "It chases a ball.", "It chases a ball.", []),
        ],
        all_entities=["Dog"],
        recurring_entities=["Dog"],
        entity_to_scene_ids={"Dog": ["SCENE-1", "SCENE-2"]},
    )
    builder = PromptBuilder(
        {
            "style_prompt": "storybook illustration",
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
            "action_emphasis_map": {"chases": "chasing motion"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[1])

    assert prompt_spec.character_prompt == "main subject: Dog, same animal across all scenes"
    assert prompt_spec.local_prompt == "Dog chases a ball., key action: chasing motion, keep the pose easy to read"


def test_prompt_builder_adds_action_prompt_for_sitting_scene() -> None:
    story = Story(
        source_path="story.txt",
        raw_text="",
        scenes=[Scene("SCENE-3", 2, "Lily sits down to eat.", "Lily sits down to eat.", ["Lily"])],
        all_entities=["Lily"],
        recurring_entities=["Lily"],
        entity_to_scene_ids={"Lily": ["SCENE-1", "SCENE-2", "SCENE-3"]},
    )
    builder = PromptBuilder(
        {
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
            "action_emphasis_map": {"sits down": "clearly seated pose"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[0])

    assert "key action: clearly seated pose" in prompt_spec.local_prompt
