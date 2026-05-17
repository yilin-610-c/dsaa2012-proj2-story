from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml

from storygen.prompt_stack.renderers.storydiffusion import (
    render_clean_native_storydiffusion_prompts,
    render_clean_v2_native_storydiffusion_prompts,
    render_natural_native_storydiffusion_prompts,
)
from storygen.types import PromptBundle, PromptSpec, Scene, Story


def _load_run_test_set() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "storydiffusion_gradio_probe" / "run_test_set.py"
    spec = importlib.util.spec_from_file_location("run_test_set_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_double_story(tmp_path: Path) -> Path:
    path = tmp_path / "07.txt"
    path.write_text(
        "[SCENE-1] <Nina> stands in the snow.\n\n"
        "[SEP]\n\n"
        "[SCENE-2] She meets <Leo> in a crowd.\n\n"
        "[SEP]\n\n"
        "[SCENE-3] They look at each other quietly.",
        encoding="utf-8",
    )
    return path


def _prompt_spec(scene_id: str, clean_text: str, action: str, consistency: str) -> PromptSpec:
    return PromptSpec(
        scene_id=scene_id,
        style_prompt="cinematic story illustration",
        character_prompt="same person across all scenes, consistent face, consistent outfit",
        global_context_prompt="maintain a consistent background identity",
        scene_consistency_prompt=consistency,
        local_prompt=clean_text,
        action_prompt=action,
        generation_prompt=f"LLM optimized {clean_text}, medium shot, story-specific setting, same person across all scenes, maintain the same background identity",
        scoring_prompt=clean_text,
        full_prompt=clean_text,
        negative_prompt="blurry",
    )


class FakePromptPipeline:
    def build(self, story):
        specs = {
            "SCENE-1": _prompt_spec("SCENE-1", "Nina stands in the snow", "stands", "new scene setting: in the snow, same scene entities: Nina, maintain the same location identity"),
            "SCENE-2": _prompt_spec("SCENE-2", "Nina meets Leo in a crowd", "meets Leo", "new scene setting: in a crowd, same scene entities: Nina and Leo"),
            "SCENE-3": _prompt_spec("SCENE-3", "Nina and Leo look at each other quietly", "look at each other quietly", "same setting: in a crowd, keep the same background and setting cues"),
        }
        return PromptBundle(
            scene_prompts=specs,
            metadata={
                "pipeline": "llm_assisted",
                "builder_version": "llm_assisted_v9",
                "character_specs": {
                    "Nina": {
                        "gender_presentation": "female",
                        "age_band": "adult",
                        "hair_color": "blonde",
                        "hairstyle": "long hair",
                        "signature_outfit": "green school uniform",
                        "signature_accessory": "backpack",
                        "metadata": {"source": "llm_assisted"},
                    },
                    "Leo": {
                        "gender_presentation": "male",
                        "age_band": "adult",
                        "hair_color": "dark",
                        "hairstyle": "wavy hair",
                        "signature_outfit": "navy vest and white shirt",
                        "metadata": {"source": "llm_assisted"},
                    },
                }
            },
        )


class FakeStudentPromptPipeline:
    def build(self, story):
        specs = {
            "SCENE-1": _prompt_spec("SCENE-1", "Student reads in the library", "reads in the library", ""),
            "SCENE-2": _prompt_spec("SCENE-2", "Student takes notes", "takes notes", ""),
            "SCENE-3": _prompt_spec("SCENE-3", "Student closes the book and leaves", "closes the book and leaves", ""),
        }
        return PromptBundle(
            scene_prompts=specs,
            metadata={
                "pipeline": "llm_assisted",
                "builder_version": "llm_assisted_v9",
                "scene_plans": {
                    "SCENE-1": {"interaction_summary": "Student reads in the library", "framing": "medium shot", "setting_focus": "library"},
                    "SCENE-2": {"interaction_summary": "Student takes notes", "framing": "medium shot", "setting_focus": "library table"},
                    "SCENE-3": {"interaction_summary": "Student closes the book and leaves", "framing": "wide shot", "setting_focus": "library exit"},
                },
                "character_specs": {
                    "student": {
                        "character_id": "student",
                        "subject_type": "human",
                        "gender_presentation": "male",
                        "age_band": "young adult",
                        "hair_color": "brown",
                        "hairstyle": "short",
                        "signature_outfit": "casual shirt",
                        "signature_accessory": "glasses",
                        "profession_marker": "student",
                        "metadata": {"source": "llm_assisted"},
                    },
                },
            },
        )


def _patch_prompt_pipeline(monkeypatch) -> None:
    import storygen.prompt_pipelines as prompt_pipelines

    monkeypatch.setattr(prompt_pipelines, "build_prompt_pipeline", lambda prompt_config: FakePromptPipeline())


def _patch_student_prompt_pipeline(monkeypatch) -> None:
    import storygen.prompt_pipelines as prompt_pipelines

    monkeypatch.setattr(prompt_pipelines, "build_prompt_pipeline", lambda prompt_config: FakeStudentPromptPipeline())


def _args(tmp_path: Path, *, mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=tmp_path,
        glob="07.txt",
        config_dir=tmp_path / "configs",
        output_root=tmp_path / "outputs",
        unwrap_output_dir=True,
        storydiffusion_root=Path("/opt/StoryDiffusion"),
        run=False,
        limit=1,
        use_reference_images=False,
        reference_image=[],
        anchor_run_dir=None,
        ensure_anchors=False,
        device="cuda",
        torch_dtype="float16",
        sd_type="Unstable",
        style="(No style)",
        seed=0,
        num_steps=20,
        guidance_scale=5.0,
        sa32=0.5,
        sa64=0.5,
        id_length=1,
        height=512,
        width=512,
        prompt_profile="cloud_storydiffusion_debug",
        prompt_builder="modular",
        prompt_modular_backend="storydiffusion",
        prompt_template_pack="",
        prompt_generation_max_words=60,
        prompt_generation_max_chars=420,
        storydiffusion_prompt_mode=mode,
        save_identity_images=False,
    )


def test_current_mode_keeps_existing_prompt_shape(tmp_path: Path) -> None:
    module = _load_run_test_set()
    story = _write_double_story(tmp_path)

    config = module.build_probe_config(
        story,
        output_root=tmp_path / "outputs",
        unwrap_output_dir=True,
        storydiffusion_root=Path("/opt/StoryDiffusion"),
        use_reference_images=False,
        reference_images=[],
        id_length=1,
        sd_type="Unstable",
        style="(No style)",
        seed=0,
        num_steps=20,
        guidance_scale=5.0,
        sa32=0.5,
        sa64=0.5,
        height=512,
        width=512,
        prompt_profile="cloud_storydiffusion_debug",
        prompt_generation_max_words=60,
        prompt_generation_max_chars=420,
        anchor_bank_summary=None,
        prompt_builder_kind="modular",
        prompt_modular_backend="storydiffusion",
        storydiffusion_prompt_mode="current",
    )

    assert "storydiffusion_prompt_debug" not in config
    assert config["save_image_start_index"] == 2
    assert config["prompts"]["prompt_array"][0].startswith("[Nina] a person, full body")
    assert "storydiffusion_clean_renderer" not in config["prompt_debug"]["prompt_pipeline"]


def test_clean_mode_writes_debug_and_renders_native_prompts(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_test_set()
    _patch_prompt_pipeline(monkeypatch)
    _write_double_story(tmp_path)

    config_paths = module.write_configs(_args(tmp_path, mode="clean"))

    assert len(config_paths) == 1
    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    debug_path = config_paths[0].with_name("storydiffusion_prompt_debug.json")
    debug = json.loads(debug_path.read_text(encoding="utf-8"))

    general_prompt = config["prompts"]["general_prompt"]
    assert "[Nina] human woman" in general_prompt
    assert "blonde long hair" in general_prompt
    assert "green school uniform" in general_prompt
    assert "backpack" in general_prompt
    assert "[Leo] human man" in general_prompt
    assert "navy vest and white shirt" in general_prompt

    assert config["save_image_start_index"] == 2
    prompt_array = config["prompts"]["prompt_array"]
    assert prompt_array[:2] == debug["identity_prompts"]
    negative_prompt = config["prompts"]["negative_prompt"]
    assert "blurry" in negative_prompt
    assert "character sheet" in negative_prompt
    assert "turnaround" in negative_prompt
    assert "multiple views" in negative_prompt
    assert "duplicate person" in negative_prompt
    assert "triptych" in negative_prompt
    assert "same person across all scenes" not in "\n".join(prompt_array).lower()
    assert "maintain the same background identity" not in "\n".join(prompt_array).lower()
    nina_identity = next(prompt for prompt in prompt_array[:2] if prompt.startswith("[Nina]"))
    assert "a single human woman" in nina_identity
    assert "one person in the image" in nina_identity
    assert "plain background" in nina_identity
    assert "character reference" not in nina_identity

    scene_prompts = debug["scene_prompts"]
    assert scene_prompts[0].startswith("[Nina] LLM optimized Nina stands in the snow")
    assert "story-specific setting" in scene_prompts[0]
    assert "same person across all scenes" not in scene_prompts[0].lower()
    assert "maintain the same background identity" not in scene_prompts[0].lower()
    assert "Leo" not in scene_prompts[0]
    assert scene_prompts[1].startswith("[Nina] [Leo] LLM optimized Nina meets Leo in a crowd")
    assert "both characters visible" in scene_prompts[1]
    assert scene_prompts[2].startswith("[Nina] [Leo] LLM optimized Nina and Leo look at each other quietly")
    assert "crowd" in scene_prompts[2]
    assert debug["source_fields"][1]["resolved_entities"] == ["Nina", "Leo"]


def test_clean_v2_mode_writes_identity_mapping_debug(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_test_set()
    _patch_prompt_pipeline(monkeypatch)
    _write_double_story(tmp_path)
    args = _args(tmp_path, mode="clean_v2")
    args.id_length = 2

    config_paths = module.write_configs(args)

    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    debug = json.loads(config_paths[0].with_name("storydiffusion_prompt_debug.json").read_text(encoding="utf-8"))
    identity_reference_prompts = debug["identity_reference_prompts"]
    assert debug["mode"] == "clean_v2"
    assert debug["identity_prompts_per_character"] == 2
    assert debug["identity_prompt_count"] == len(identity_reference_prompts)
    assert debug["story_frame_start_index"] == len(identity_reference_prompts)
    assert config["save_image_start_index"] == len(identity_reference_prompts)
    assert config["generation"]["id_length"] == len(identity_reference_prompts)
    assert config["generation"]["storydiffusion_internal_id_length"] == 2
    assert debug["saved_image_prompt_map"]["image_000.png"]["story_scene_prompt_index"] == 0
    assert debug["saved_image_prompt_map"]["image_000.png"]["prompt"] == debug["story_scene_prompts"][0]
    assert config["prompts"]["prompt_array"][: len(identity_reference_prompts)] == identity_reference_prompts


def test_natural_mode_writes_storyboard_debug(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_test_set()
    _patch_prompt_pipeline(monkeypatch)
    _write_double_story(tmp_path)
    args = _args(tmp_path, mode="natural")
    args.id_length = 2

    config_paths = module.write_configs(args)

    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    debug = json.loads(config_paths[0].with_name("storydiffusion_prompt_debug.json").read_text(encoding="utf-8"))
    identity_reference_prompts = debug["identity_reference_prompts"]
    assert debug["mode"] == "natural"
    assert debug["identity_prompt_count"] == len(identity_reference_prompts)
    assert config["save_image_start_index"] == len(identity_reference_prompts)
    assert config["generation"]["id_length"] == len(identity_reference_prompts)
    assert config["generation"]["storydiffusion_internal_id_length"] == 2
    assert "character sheet" in config["prompts"]["negative_prompt"]
    assert "multiple views" in config["prompts"]["negative_prompt"]
    assert debug["natural_scene_prompt"] == debug["story_scene_prompts"]
    assert len(debug["structured_source_fields"]) == len(debug["story_scene_prompts"])
    assert config["prompt_debug"]["probe_overrides"]["prompt.llm.max_output_tokens"] == 2400
    assert debug["validation_warnings"] == [
        {"scene_id": "SCENE-1", "warnings": []},
        {"scene_id": "SCENE-2", "warnings": []},
        {"scene_id": "SCENE-3", "warnings": []},
    ]
    assert debug["final_prompt_array"] == config["prompts"]["prompt_array"]


def test_natural_mode_writes_identity_image_debug_map(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_test_set()
    _patch_prompt_pipeline(monkeypatch)
    _write_double_story(tmp_path)
    args = _args(tmp_path, mode="natural")
    args.id_length = 2
    args.save_identity_images = True

    config_paths = module.write_configs(args)

    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    debug = json.loads(config_paths[0].with_name("storydiffusion_prompt_debug.json").read_text(encoding="utf-8"))
    assert config["save_identity_images"] is True
    mapping = debug["identity_image_prompt_map"]
    assert sorted(mapping) == ["identity_000.png", "identity_001.png", "identity_002.png", "identity_003.png"]
    assert mapping["identity_000.png"]["path"].endswith("identity_refs/identity_000.png")
    assert mapping["identity_000.png"]["prompt"] == debug["identity_reference_prompts"][0]
    assert debug["saved_image_prompt_map"]["image_000.png"]["prompt"] == debug["story_scene_prompts"][0]


def test_natural_student_prompt_uses_case_insensitive_character_spec(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_test_set()
    _patch_student_prompt_pipeline(monkeypatch)
    story = tmp_path / "19.txt"
    story.write_text(
        "[SCENE-1] <Student> reads in the library.\n\n"
        "[SEP]\n\n"
        "[SCENE-2] He takes notes.\n\n"
        "[SEP]\n\n"
        "[SCENE-3] He closes the book and leaves.",
        encoding="utf-8",
    )
    args = _args(tmp_path, mode="natural")
    args.glob = "19.txt"

    config_paths = module.write_configs(args)

    debug = json.loads(config_paths[0].with_name("storydiffusion_prompt_debug.json").read_text(encoding="utf-8"))
    general_prompt = debug["general_prompt"]
    identity_prompt = debug["identity_reference_prompts"][0]
    assert "[Student] human man" in general_prompt
    assert "short brown hair" in general_prompt
    assert "casual shirt" in general_prompt
    assert "glasses" in general_prompt
    assert "student" in general_prompt
    assert "[Student] a single human man" in identity_prompt
    assert "one person in the image" in identity_prompt
    assert "no character sheet" in identity_prompt
    assert "no multiple views" in identity_prompt
    assert "character reference" not in identity_prompt
    assert "human person" not in general_prompt
    assert "human person" not in identity_prompt
    assert debug["story_scene_prompts"][0] == "[Student] reading in the library, medium shot"
    assert debug["story_scene_prompts"][1] == "[Student] taking notes, at a library table, medium shot"
    assert debug["saved_image_prompt_map"]["image_000.png"]["prompt"] == debug["story_scene_prompts"][0]


def test_clean_renderer_uses_type_aware_animal_and_robot_prompts() -> None:
    story = Story(
        source_path="story.txt",
        raw_text="<Cat> hides.\n<Cat> meets <Dog>.\n<Robot> helps.",
        scenes=[
            Scene("SCENE-1", 0, "<Cat> hides.", "Cat hides.", ["Cat"]),
            Scene("SCENE-2", 1, "<Cat> meets <Dog>.", "Cat meets Dog.", ["Cat", "Dog"]),
            Scene("SCENE-3", 2, "<Robot> helps.", "Robot helps.", ["Robot"]),
        ],
        all_entities=["Cat", "Dog", "Robot"],
        recurring_entities=["Cat", "Dog", "Robot"],
        entity_to_scene_ids={"Cat": ["SCENE-1", "SCENE-2"], "Dog": ["SCENE-2"], "Robot": ["SCENE-3"]},
    )
    specs = {
        "SCENE-1": _prompt_spec("SCENE-1", "Cat hides in a corner", "hiding", "same setting: room"),
        "SCENE-2": _prompt_spec("SCENE-2", "Cat is. Dog is. Cat looks at Dog", "looking", "same setting: room"),
        "SCENE-3": _prompt_spec("SCENE-3", "Robot helps", "helping", "same setting: workshop"),
    }
    specs["SCENE-2"].generation_prompt = "Cat is. Dog is. Cat looks at Dog, corner of the room"
    rendered = render_clean_native_storydiffusion_prompts(
        story,
        specs,
        {
            "Cat": {
                "subject_type": "animal",
                "species": "cat",
                "fur_color": "gray",
                "fur_pattern": "short fur",
                "body_size": "small",
            },
            "Dog": {
                "subject_type": "animal",
                "species": "dog",
                "fur_color": "brown",
                "fur_pattern": "short coat",
                "markings": "floppy ears",
                "body_size": "medium",
            },
            "Robot": {
                "subject_type": "robot",
                "material": "metal",
                "color_scheme": "silver",
                "shape_features": "boxy body",
            },
        },
    )

    joined = "\n".join([rendered.general_prompt, *rendered.identity_prompts, *rendered.scene_prompts]).lower()
    assert "[cat] small gray cat, short fur" in rendered.general_prompt.lower()
    assert "[dog] medium brown dog, short coat, floppy ears" in rendered.general_prompt.lower()
    assert "[robot] silver metal robot, boxy body" in rendered.general_prompt.lower()
    assert "human person" not in joined
    assert "hairstyle" not in joined
    assert "complete outfit visible" not in next(prompt for prompt in rendered.identity_prompts if prompt.startswith("[Cat]")).lower()
    assert "one animal in the image" in next(prompt for prompt in rendered.identity_prompts if prompt.startswith("[Cat]")).lower()
    assert "one robot in the image" in next(prompt for prompt in rendered.identity_prompts if prompt.startswith("[Robot]")).lower()
    assert "cat is. dog is." not in joined
    assert "both animals visible" in rendered.scene_prompts[1].lower()
    assert "two-animal composition" in rendered.scene_prompts[1].lower()


def test_clean_v2_renderer_uses_lightweight_animal_scene_prompts() -> None:
    story = Story(
        source_path="03.txt",
        raw_text="<Cat> hides in a corner.\n<Cat> looks at <Dog>.",
        scenes=[
            Scene("SCENE-1", 0, "<Cat> hides in a corner.", "Cat hides in a corner.", ["Cat"]),
            Scene("SCENE-2", 1, "<Cat> looks at <Dog>.", "Cat looks at Dog.", ["Cat", "Dog"]),
        ],
        all_entities=["Cat", "Dog"],
        recurring_entities=["Cat", "Dog"],
        entity_to_scene_ids={"Cat": ["SCENE-1", "SCENE-2"], "Dog": ["SCENE-2"]},
    )
    specs = {
        "SCENE-1": _prompt_spec("SCENE-1", "Cat hides in a corner", "hiding in a corner", "same setting: room"),
        "SCENE-2": _prompt_spec("SCENE-2", "Cat looks at Dog", "Cat looks at Dog", "same setting: room"),
    }
    specs["SCENE-2"].generation_prompt = "Cat is a small gray cat, striped. Dog is a medium brown dog, spotted, floppy ears. Cat looks at Dog. medium two-shot"
    rendered = render_clean_v2_native_storydiffusion_prompts(
        story,
        specs,
        {
            "Cat": {"subject_type": "animal", "species": "cat", "fur_color": "gray", "fur_pattern": "striped", "body_size": "small"},
            "Dog": {
                "subject_type": "animal",
                "species": "dog",
                "fur_color": "brown",
                "fur_pattern": "spotted",
                "markings": "floppy ears",
                "body_size": "medium",
            },
        },
        scene_plans={
            "SCENE-1": {"interaction_summary": "Cat hides in a corner", "framing": "medium shot", "setting_focus": "room corner"},
            "SCENE-2": {
                "interaction_summary": "Cat looks at Dog",
                "spatial_relation": "Cat on the left, Dog on the right",
                "framing": "medium two-shot",
                "setting_focus": "living room",
            },
        },
        identity_prompts_per_character=2,
    )

    joined_scene_prompts = "\n".join(rendered.scene_prompts).lower()
    joined_identity_prompts = "\n".join(rendered.identity_reference_prompts).lower()
    assert "[dog] medium brown dog, visible spotted coat pattern, floppy ears" in rendered.general_prompt.lower()
    assert "visible spotted coat pattern" in joined_identity_prompts
    assert "a single small gray cat" in joined_identity_prompts
    assert "a single medium brown dog" in joined_identity_prompts
    assert "one animal in the image" in joined_identity_prompts
    assert "no character sheet" in joined_identity_prompts
    assert "no multiple views" in joined_identity_prompts
    assert "side view animal reference" not in joined_identity_prompts
    assert "three-quarter animal reference" not in joined_identity_prompts
    assert "cat is a small gray cat" not in joined_scene_prompts
    assert "dog is a medium brown dog" not in joined_scene_prompts
    assert "complete outfit visible" not in joined_identity_prompts
    assert "person" not in joined_identity_prompts
    assert rendered.scene_prompts[1].lower().count("medium two-shot") == 0
    assert "two-animal composition" in rendered.scene_prompts[1].lower()
    assert "both animals visible" in rendered.scene_prompts[1].lower()
    assert rendered.identity_prompts_per_character == 2
    assert len(rendered.identity_reference_prompts) == 4
    assert rendered.save_image_start_index == 4
    assert rendered.saved_image_prompt_map["image_000.png"]["story_scene_prompt_index"] == 0
    assert rendered.saved_image_prompt_map["image_000.png"]["prompt"] == rendered.story_scene_prompts[0]


def test_clean_v2_human_dual_scene_stays_story_faithful_to_cafe_text() -> None:
    story = Story(
        source_path="06.txt",
        raw_text="<Jack> and <Sara> sit in a park and talk.\nThey continue talking in a cafe.",
        scenes=[
            Scene("SCENE-1", 0, "<Jack> and <Sara> sit in a park and talk.", "Jack and Sara sit in a park and talk.", ["Jack", "Sara"]),
            Scene("SCENE-2", 1, "They continue talking in a cafe.", "They continue talking in a cafe.", []),
        ],
        all_entities=["Jack", "Sara"],
        recurring_entities=["Jack", "Sara"],
        entity_to_scene_ids={"Jack": ["SCENE-1"], "Sara": ["SCENE-1"]},
    )
    specs = {
        "SCENE-1": _prompt_spec("SCENE-1", "Jack and Sara talk in a park", "talking together", ""),
        "SCENE-2": _prompt_spec("SCENE-2", "Jack and Sara talking at a cafe table", "talking together", ""),
    }
    rendered = render_clean_v2_native_storydiffusion_prompts(
        story,
        specs,
        {
            "Jack": {
                "subject_type": "human",
                "gender_presentation": "male",
                "age_band": "adult",
                "hair_color": "brown",
                "hairstyle": "short",
                "signature_outfit": "casual shirt",
            },
            "Sara": {
                "subject_type": "human",
                "gender_presentation": "female",
                "age_band": "adult",
                "hair_color": "blonde",
                "hairstyle": "long",
                "signature_outfit": "summer dress",
            },
        },
        scene_plans={
            "SCENE-1": {"interaction_summary": "Jack and Sara are talking together", "spatial_relation": "side by side", "framing": "medium two-shot", "setting_focus": "park"},
            "SCENE-2": {"interaction_summary": "Jack and Sara are talking together", "spatial_relation": "across the table", "framing": "medium two-shot", "setting_focus": "cafe table"},
        },
        identity_prompts_per_character=1,
    )

    cafe_prompt = rendered.scene_prompts[1].lower()
    assert "talking together" in cafe_prompt
    assert "cafe" in cafe_prompt
    assert "drinking coffee" not in cafe_prompt
    assert cafe_prompt.count("medium two-shot") == 1
    assert "jack is an adult" not in cafe_prompt
    assert "[jack] human man, short brown hair, casual shirt" in rendered.general_prompt.lower()


def test_natural_renderer_uses_short_animal_storyboard_prompts() -> None:
    story = Story(
        source_path="03.txt",
        raw_text="<Cat> hides in a corner and watches.\n<Cat> looks at <Dog>.\n<Cat> and <Dog> move forward together.",
        scenes=[
            Scene("SCENE-1", 0, "<Cat> hides in a corner and watches.", "Cat hides in a corner and watches.", ["Cat"]),
            Scene("SCENE-2", 1, "<Cat> looks at <Dog>.", "Cat looks at Dog.", ["Cat", "Dog"]),
            Scene("SCENE-3", 2, "<Cat> and <Dog> move forward together.", "Cat and Dog move forward together.", ["Cat", "Dog"]),
        ],
        all_entities=["Cat", "Dog"],
        recurring_entities=["Cat", "Dog"],
        entity_to_scene_ids={"Cat": ["SCENE-1", "SCENE-2", "SCENE-3"], "Dog": ["SCENE-2", "SCENE-3"]},
    )
    specs = {
        "SCENE-1": _prompt_spec("SCENE-1", "Cat hides in a corner and watches", "hiding in a corner", "same setting: room"),
        "SCENE-2": _prompt_spec("SCENE-2", "Cat looks at Dog", "Cat looks at Dog", "same setting: room"),
        "SCENE-3": _prompt_spec("SCENE-3", "Cat and Dog move forward together", "moving forward together", ""),
    }
    specs["SCENE-2"].generation_prompt = "Cat is a small gray cat, striped. Dog is a medium brown dog, spotted, floppy ears. Cat looks at Dog. medium two-shot"
    rendered = render_natural_native_storydiffusion_prompts(
        story,
        specs,
        {
            "Cat": {"subject_type": "animal", "species": "cat", "fur_color": "gray", "fur_pattern": "striped", "body_size": "small"},
            "Dog": {
                "subject_type": "animal",
                "species": "dog",
                "fur_color": "brown",
                "fur_pattern": "spotted",
                "markings": "floppy ears",
                "body_size": "medium",
            },
        },
        scene_plans={
            "SCENE-1": {"interaction_summary": "Cat hides in a corner", "framing": "medium shot", "setting_focus": "room corner"},
            "SCENE-2": {
                "interaction_summary": "Cat looks at Dog",
                "spatial_relation": "Cat on the left, Dog on the right",
                "framing": "medium two-shot",
                "setting_focus": "living room",
            },
            "SCENE-3": {
                "interaction_summary": "Cat and Dog move forward together",
                "framing": "wide shot",
                "setting_focus": "open space",
            },
        },
        identity_prompts_per_character=2,
    )

    joined_scene_prompts = "\n".join(rendered.scene_prompts).lower()
    assert rendered.scene_prompts[0] == "[Cat] hiding and watching from a corner, medium shot"
    assert rendered.scene_prompts[1] == "[Cat] [Dog] Cat looks at Dog, in a living room"
    assert rendered.scene_prompts[2] == "[Cat] [Dog] Cat and Dog move forward together, in an open space, wide shot"
    assert "cat is a small gray cat" not in joined_scene_prompts
    assert "dog is a medium brown dog" not in joined_scene_prompts
    assert "both animals visible" not in joined_scene_prompts
    assert "two-animal composition" not in joined_scene_prompts
    assert "action readable" not in joined_scene_prompts
    assert "left" not in joined_scene_prompts
    assert "right" not in joined_scene_prompts
    assert rendered.source_fields[0]["validation_warnings"] == []
    assert rendered.source_fields[0]["natural_scene_prompt"] == rendered.scene_prompts[0]
    assert "a single small gray cat" in "\n".join(rendered.identity_reference_prompts).lower()
    assert "side view animal reference" not in "\n".join(rendered.identity_reference_prompts).lower()
    assert rendered.identity_prompts_per_character == 2
    assert len(rendered.identity_reference_prompts) == 4
    assert rendered.save_image_start_index == 4
    assert rendered.saved_image_prompt_map["image_000.png"]["prompt"] == rendered.story_scene_prompts[0]


def test_natural_human_dual_scene_stays_story_faithful_to_cafe_text() -> None:
    story = Story(
        source_path="06.txt",
        raw_text="<Jack> and <Sara> sit in a park and talk.\nThey continue talking in a cafe.",
        scenes=[
            Scene("SCENE-1", 0, "<Jack> and <Sara> sit in a park and talk.", "Jack and Sara sit in a park and talk.", ["Jack", "Sara"]),
            Scene("SCENE-2", 1, "They continue talking in a cafe.", "They continue talking in a cafe.", []),
        ],
        all_entities=["Jack", "Sara"],
        recurring_entities=["Jack", "Sara"],
        entity_to_scene_ids={"Jack": ["SCENE-1"], "Sara": ["SCENE-1"]},
    )
    specs = {
        "SCENE-1": _prompt_spec("SCENE-1", "Jack and Sara talk in a park", "talking together", ""),
        "SCENE-2": _prompt_spec("SCENE-2", "Jack and Sara talking at a cafe table", "talking together", ""),
    }
    rendered = render_natural_native_storydiffusion_prompts(
        story,
        specs,
        {
            "Jack": {
                "subject_type": "human",
                "gender_presentation": "male",
                "age_band": "adult",
                "hair_color": "brown",
                "hairstyle": "short",
                "signature_outfit": "casual shirt",
            },
            "Sara": {
                "subject_type": "human",
                "gender_presentation": "female",
                "age_band": "adult",
                "hair_color": "blonde",
                "hairstyle": "long",
                "signature_outfit": "summer dress",
            },
        },
        scene_plans={
            "SCENE-1": {"interaction_summary": "Jack and Sara are talking together", "spatial_relation": "side by side", "framing": "medium two-shot", "setting_focus": "park"},
            "SCENE-2": {"interaction_summary": "Jack and Sara are talking together", "spatial_relation": "across the table", "framing": "medium two-shot", "setting_focus": "cafe table"},
        },
        identity_prompts_per_character=1,
    )

    cafe_prompt = rendered.scene_prompts[1].lower()
    assert cafe_prompt.startswith("[jack] [sara] jack and sara continue talking in a cafe")
    assert "across the table" in cafe_prompt
    assert "medium two-shot" in cafe_prompt
    assert "drinking coffee" not in cafe_prompt
    assert "jack is an adult" not in cafe_prompt
    assert "both characters visible" not in cafe_prompt
    joined_identity_prompts = "\n".join(rendered.identity_reference_prompts).lower()
    assert "a single human man" in joined_identity_prompts
    assert "a single human woman" in joined_identity_prompts
    assert "one person in the image" in joined_identity_prompts
    assert "character reference" not in joined_identity_prompts


def test_natural_robot_scene_prompt_avoids_human_terms() -> None:
    story = Story(
        source_path="13.txt",
        raw_text="<Robot> repairs a signal tower.",
        scenes=[
            Scene("SCENE-1", 0, "<Robot> repairs a signal tower.", "Robot repairs a signal tower.", ["Robot"]),
        ],
        all_entities=["Robot"],
        recurring_entities=["Robot"],
        entity_to_scene_ids={"Robot": ["SCENE-1"]},
    )
    specs = {
        "SCENE-1": _prompt_spec("SCENE-1", "Robot repairs a signal tower", "repairing the signal tower", ""),
    }
    rendered = render_natural_native_storydiffusion_prompts(
        story,
        specs,
        {
            "Robot": {
                "subject_type": "robot",
                "material": "metal",
                "color_scheme": "silver",
                "shape_features": "boxy body",
            },
        },
        scene_plans={"SCENE-1": {"framing": "medium shot", "setting_focus": "signal tower"}},
    )

    joined = "\n".join(rendered.scene_prompts).lower()
    assert rendered.scene_prompts[0] == "[Robot] repairing a signal tower, medium shot"
    assert "person" not in joined
    assert "outfit" not in joined
    identity_prompt = rendered.identity_reference_prompts[0].lower()
    assert "a single silver metal robot" in identity_prompt
    assert "one robot in the image" in identity_prompt
    assert "no character sheet" in identity_prompt
    assert "character reference" not in identity_prompt
    assert rendered.source_fields[0]["validation_warnings"] == []
