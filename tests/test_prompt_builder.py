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
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[0])

    assert prompt_spec.style_prompt == "cinematic illustration"
    assert prompt_spec.character_prompt.startswith("main subject: Hero,")
    assert "same person across all scenes" in prompt_spec.character_prompt
    assert prompt_spec.global_context_prompt == "shared story context: Hero, keep the same lighting and palette"
    assert prompt_spec.action_prompt == "runs"
    assert prompt_spec.generation_prompt.startswith("Hero, runs, cinematic illustration")
    assert prompt_spec.scoring_prompt == "Hero, runs"
    assert "Hero runs." in prompt_spec.local_prompt
    assert "key action: active running pose" in prompt_spec.local_prompt
    assert "keep the pose easy to read" in prompt_spec.local_prompt
    assert "cinematic illustration" in prompt_spec.full_prompt
    assert len(prompt_spec.generation_prompt) < len(prompt_spec.full_prompt)
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
            "action_emphasis_map": {"looks out": "looking out through the window"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[1])

    assert prompt_spec.character_prompt.startswith("main subject: Lily,")
    assert "same person across all scenes" in prompt_spec.character_prompt
    assert prompt_spec.global_context_prompt == (
        "shared story context: Lily, recurring setting: the kitchen, keep the same lighting and palette"
    )
    assert prompt_spec.action_prompt == "looks out the window quietly"
    assert prompt_spec.scoring_prompt == "Lily, looks out the window quietly"
    assert prompt_spec.generation_prompt.startswith("Lily, looks out the window quietly, cinematic illustration")
    assert "Lily looks out the window quietly." in prompt_spec.local_prompt
    assert "key action: looking out through the window" in prompt_spec.local_prompt
    assert "keep the pose easy to read" in prompt_spec.local_prompt


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
            "rewriter": {"type": "rule_based"},
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
            "action_emphasis_map": {"chases": "chasing motion"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[1])

    assert prompt_spec.character_prompt.startswith("main subject: Dog,")
    assert "same animal across all scenes" in prompt_spec.character_prompt
    assert prompt_spec.action_prompt == "chases a ball"
    assert prompt_spec.generation_prompt.startswith("Dog, chases a ball, storybook illustration")
    assert prompt_spec.scoring_prompt == "Dog, chases a ball"
    assert "Dog chases a ball." in prompt_spec.local_prompt
    assert "key action: chasing motion" in prompt_spec.local_prompt
    assert "keep the pose easy to read" in prompt_spec.local_prompt


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
            "action_emphasis_map": {"sits down": "clearly seated pose"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[0])

    assert "key action: clearly seated pose" in prompt_spec.local_prompt
    assert prompt_spec.action_prompt == "sits down to eat"
    assert prompt_spec.generation_prompt.startswith("Lily, sits down to eat, cinematic illustration")
    assert prompt_spec.scoring_prompt == "Lily, sits down to eat"


def test_prompt_builder_adds_short_setting_to_scoring_prompt() -> None:
    story = Story(
        source_path="story.txt",
        raw_text="",
        scenes=[Scene("SCENE-1", 0, "<Lily> makes breakfast in the kitchen.", "Lily makes breakfast in the kitchen.", ["Lily"])],
        all_entities=["Lily"],
        recurring_entities=["Lily"],
        entity_to_scene_ids={"Lily": ["SCENE-1"]},
    )
    builder = PromptBuilder(
        {
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
            "action_emphasis_map": {"makes breakfast": "preparing breakfast"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[0])

    assert prompt_spec.action_prompt == "makes breakfast"
    assert "Lily, makes breakfast" in prompt_spec.generation_prompt
    assert "kitchen" in prompt_spec.generation_prompt.lower()
    assert "cinematic illustration" in prompt_spec.generation_prompt
    assert prompt_spec.scoring_prompt == "Lily, makes breakfast, in the kitchen"


def test_generation_prompt_is_shorter_and_avoids_verbose_continuity_phrases() -> None:
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
            )
        ],
        all_entities=["Lily"],
        recurring_entities=["Lily"],
        entity_to_scene_ids={"Lily": ["SCENE-1"]},
    )
    builder = PromptBuilder(
        {
            "rewriter": {"type": "rule_based"},
            "style_prompt": "cinematic story illustration, coherent visual style",
            "subject_prefix": "main subject:",
            "global_context_prefix": "shared story context:",
            "setting_prefix": "recurring setting:",
            "replace_leading_pronouns": True,
            "human_identity_prompt": "same person across all scenes, consistent face, hairstyle, outfit, accessories, and body proportions",
            "animal_identity_prompt": "same animal across all scenes",
            "generic_identity_prompt": "same subject across all scenes",
            "scene_continuity_prompt": "maintain a consistent background identity, lighting direction, color palette, and visual tone across the sequence unless the text clearly changes them",
            "action_emphasis_template": "key action: {action_phrase}",
            "default_action_prompt": "show the action clearly with an unambiguous pose",
            "scene_composition_prompt": "keep the full pose and scene relationship easy to read",
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
            "action_emphasis_map": {"makes breakfast": "preparing breakfast with food and kitchen tools visible"},
            "quality_suffix": "clean composition, readable action, consistent framing",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[0])

    assert len(prompt_spec.generation_prompt) < len(prompt_spec.full_prompt)
    assert len(prompt_spec.scoring_prompt) < len(prompt_spec.generation_prompt)
    assert "consistent background identity" not in prompt_spec.generation_prompt
    assert "clean composition" not in prompt_spec.generation_prompt
    assert "full pose and scene relationship" not in prompt_spec.generation_prompt


def test_generation_prompt_respects_budget() -> None:
    story = Story(
        source_path="story.txt",
        raw_text="",
        scenes=[Scene("SCENE-1", 0, "<Artist> paints in front of a canvas.", "Artist paints in front of a canvas.", ["Artist"])],
        all_entities=["Artist"],
        recurring_entities=["Artist"],
        entity_to_scene_ids={"Artist": ["SCENE-1"]},
    )
    builder = PromptBuilder(
        {
            "rewriter": {"type": "rule_based"},
            "style_prompt": "cinematic story illustration, coherent visual style",
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
            "generation_max_words": 4,
            "generation_max_chars": 24,
            "generation_template": "{subject}, {action}{setting_clause}{style_clause}",
            "scoring_template": "{subject}, {action}{setting_clause}",
            "scoring_include_style": False,
            "scoring_include_global_context": False,
            "scoring_max_words": 20,
            "scoring_max_chars": 160,
            "action_emphasis_map": {"paints": "painting action clearly shown"},
            "quality_suffix": "clean composition",
            "negative_prompt": "blurry",
        }
    )

    prompt_spec = builder.build_prompt_for_scene(story, story.scenes[0])

    assert len(prompt_spec.generation_prompt.split()) <= 4
    assert len(prompt_spec.generation_prompt) <= 24


def test_bus_narrative_door_scene_not_marked_as_new_venue() -> None:
    """Regression: vehicle continuity + 'at the door' must not become a false 'new scene setting'."""
    from pathlib import Path

    from storygen.parser import parse_story_file

    root = Path(__file__).resolve().parents[1]
    story = parse_story_file(root / "test_set" / "02.txt")
    cfg = {
        "rewriter": {"type": "rule_based"},
        "style_prompt": "cinematic story illustration, coherent visual style",
        "subject_prefix": "main subject:",
        "global_context_prefix": "shared story context:",
        "setting_prefix": "recurring setting:",
        "replace_leading_pronouns": True,
        "human_identity_prompt": "same person across all scenes",
        "animal_identity_prompt": "same animal across all scenes",
        "generic_identity_prompt": "same subject across all scenes",
        "scene_continuity_prompt": "maintain a consistent background identity",
        "action_emphasis_template": "key action: {action_phrase}",
        "default_action_prompt": "show the action clearly",
        "scene_composition_prompt": "keep the pose easy to read",
        "generation_include_style": True,
        "generation_include_global_context": False,
        "generation_include_scene_consistency": True,
        "generation_include_quality_suffix": False,
        "generation_include_scene_composition": False,
        "generation_max_words": 28,
        "generation_max_chars": 280,
        "generation_scene_consistency_max_words": 18,
        "generation_scene_consistency_max_chars": 160,
        "generation_trim_truncation_artifacts": True,
        "generation_template": "{subject}, {action}{setting_clause}{style_clause}",
        "scoring_template": "{subject}, {action}{setting_clause}",
        "scoring_include_style": False,
        "scoring_include_global_context": False,
        "scoring_max_words": 20,
        "scoring_max_chars": 160,
        "action_emphasis_map": {},
        "quality_suffix": "clean composition",
        "negative_prompt": "blurry",
    }
    builder = PromptBuilder(cfg)
    spec = builder.build_prompt_for_scene(story, story.scenes[1])
    low = spec.scene_consistency_prompt.lower()
    assert "new scene setting: at the door" not in low
    assert "bus door" in low
    assert not spec.generation_prompt.rstrip().endswith(":")


def test_cinematography_and_dynamic_negative_switches() -> None:
    story = Story(
        source_path="story.txt",
        raw_text="",
        scenes=[
            Scene(
                "SCENE-1",
                0,
                "<Lily> walks the alley at night.",
                "Lily walks the alley at night.",
                ["Lily"],
            )
        ],
        all_entities=["Lily"],
        recurring_entities=["Lily"],
        entity_to_scene_ids={"Lily": ["SCENE-1"]},
    )
    cfg = {
        "rewriter": {"type": "rule_based"},
        "style_prompt": "cinematic story illustration, coherent visual style",
        "subject_prefix": "main subject:",
        "global_context_prefix": "shared story context:",
        "setting_prefix": "recurring setting:",
        "replace_leading_pronouns": True,
        "human_identity_prompt": "same person across all scenes",
        "animal_identity_prompt": "same animal across all scenes",
        "generic_identity_prompt": "same subject across all scenes",
        "scene_continuity_prompt": "maintain a consistent background identity",
        "action_emphasis_template": "key action: {action_phrase}",
        "default_action_prompt": "show the action clearly",
        "scene_composition_prompt": "keep the pose easy to read",
        "generation_include_style": True,
        "generation_include_global_context": False,
        "generation_include_scene_consistency": True,
        "generation_include_quality_suffix": False,
        "generation_include_scene_composition": False,
        "generation_max_words": 40,
        "generation_max_chars": 400,
        "generation_scene_consistency_max_words": 20,
        "generation_scene_consistency_max_chars": 200,
        "generation_template": "{subject}, {action}{setting_clause}{style_clause}",
        "scoring_template": "{subject}, {action}{setting_clause}",
        "scoring_include_style": False,
        "scoring_include_global_context": False,
        "scoring_max_words": 20,
        "scoring_max_chars": 160,
        "action_emphasis_map": {},
        "quality_suffix": "clean composition",
        "negative_prompt": "blurry",
        "cinematography": {
            "enabled": True,
            "night_keywords": ["night", "alley"],
            "night_phrase": "rim light test phrase",
            "cozy_indoor_keywords": [],
            "cozy_phrase": "",
            "default_phrase": "",
        },
        "dynamic_negative": {
            "enabled": True,
            "night_substrings": ["night", "alley"],
            "night_extra": "daylight washout",
            "single_subject_extra": "crowd duplicate",
        },
    }
    spec = PromptBuilder(cfg).build_prompt_for_scene(story, story.scenes[0])
    assert "rim light test phrase" in spec.generation_prompt
    assert "daylight washout" in spec.negative_prompt
    assert "crowd duplicate" in spec.negative_prompt
