from __future__ import annotations

from pathlib import Path

import yaml

from storygen.parser import parse_story_file
from storygen.prompt_builder import PromptBuilder
from storygen.prompt_stack.facade import ModularPromptBuilder
from storygen.prompt_stack.factory import build_rule_prompt_builder, merge_prompt_config_with_pack
from storygen.types import PromptSpec, Scene, Story


def _dual_story_config() -> dict:
    return {
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
        "dual_identity_prompt_enabled": True,
        "dual_spatial_relation_prompt_enabled": True,
        "dual_spatial_relation_template": "{left} on the left, {right} on the right",
        "dual_identity_separation_prompt": "keep both characters visually distinct; do not swap hairstyles, outfits, accessories, or positions",
        "generation_include_style": True,
        "generation_include_global_context": False,
        "generation_include_scene_consistency": True,
        "generation_include_quality_suffix": False,
        "generation_include_scene_composition": False,
        "generation_max_words": 28,
        "generation_max_chars": 220,
        "dual_primary_generation_max_words": 80,
        "dual_primary_generation_max_chars": 520,
        "generation_template": "{subject}, {action}{setting_clause}{style_clause}",
        "scoring_template": "{subject}, {action}{setting_clause}",
        "scoring_include_style": False,
        "scoring_include_global_context": False,
        "scoring_max_words": 40,
        "scoring_max_chars": 260,
        "action_emphasis_map": {},
        "quality_suffix": "clean composition",
        "negative_prompt": "blurry",
    }


def _dual_story() -> Story:
    return Story(
        source_path="story.txt",
        raw_text="",
        scenes=[
            Scene(
                "SCENE-1",
                0,
                "<Jack> and <Sara> talk at a cafe.",
                "Jack and Sara talk at a cafe.",
                ["Jack", "Sara"],
            )
        ],
        all_entities=["Jack", "Sara"],
        recurring_entities=["Jack", "Sara"],
        entity_to_scene_ids={"Jack": ["SCENE-1"], "Sara": ["SCENE-1"]},
    )


def test_modular_sdxl_matches_legacy_prompts() -> None:
    cfg = _dual_story_config()
    legacy = PromptBuilder(dict(cfg))
    modular = ModularPromptBuilder(dict(cfg), backend="sdxl", template_pack="default")
    story = _dual_story()
    assert legacy.build_story_prompts(story) == modular.build_story_prompts(story)


def test_build_rule_prompt_builder_legacy_default() -> None:
    cfg = {**_dual_story_config(), "builder": "legacy"}
    b = build_rule_prompt_builder(cfg)
    assert type(b).__name__ == "PromptBuilder"


def test_build_rule_prompt_builder_modular() -> None:
    cfg = {
        **_dual_story_config(),
        "builder": "modular",
        "modular": {"backend": "sdxl", "template_pack": "default"},
    }
    b = build_rule_prompt_builder(cfg)
    assert type(b).__name__ == "ModularPromptBuilder"


def test_storydiffusion_backend_strips_spatial_hacks() -> None:
    cfg = {
        **_dual_story_config(),
        "builder": "modular",
        "modular": {"backend": "storydiffusion", "template_pack": "default"},
    }
    b = build_rule_prompt_builder(cfg)
    story = _dual_story()
    spec = b.build_story_prompts(story)["SCENE-1"]
    assert "on the left" not in spec.character_prompt.lower()
    assert "on the right" not in spec.character_prompt.lower()
    assert "wide environmental two-shot" not in spec.generation_prompt.lower()
    assert "full bodies visible" not in spec.generation_prompt.lower()


def test_prompt_builder_public_story_context() -> None:
    story = _dual_story()
    cfg = _dual_story_config()
    ctx = PromptBuilder(cfg).build_story_context(story)
    assert ctx.get("is_dual_subject_story") is True


def test_merge_template_pack_loads_default_yaml() -> None:
    merged, post = merge_prompt_config_with_pack({"style_prompt": "x"}, "default")
    assert merged["style_prompt"] == "x"
    assert "segment_bans" in post


def test_default_template_yaml_is_valid() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "prompt_templates" / "default.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data.get("storydiffusion_postprocess"), dict)


def test_parse_story_file_with_modular_profile_config() -> None:
    """Smoke: real file parses; modular builder produces PromptSpecs."""
    root = Path(__file__).resolve().parents[1]
    story = parse_story_file(root / "test_set" / "01.txt")
    cfg = {"style_prompt": "test", "negative_prompt": "", "rewriter": {"type": "rule_based"}}
    specs = ModularPromptBuilder({**cfg, "builder": "modular", "modular": {"backend": "sdxl"}}, backend="sdxl").build_story_prompts(story)
    assert specs
    for spec in specs.values():
        assert isinstance(spec, PromptSpec)
