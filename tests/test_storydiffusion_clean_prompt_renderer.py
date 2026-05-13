from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml

from storygen.types import PromptBundle, PromptSpec


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


def _patch_prompt_pipeline(monkeypatch) -> None:
    import storygen.prompt_pipelines as prompt_pipelines

    monkeypatch.setattr(prompt_pipelines, "build_prompt_pipeline", lambda prompt_config: FakePromptPipeline())


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
    assert "same person across all scenes" not in "\n".join(prompt_array).lower()
    assert "maintain the same background identity" not in "\n".join(prompt_array).lower()
    nina_identity = next(prompt for prompt in prompt_array[:2] if prompt.startswith("[Nina]"))
    assert "full body character reference of Nina" in nina_identity
    assert "clear face" in nina_identity
    assert "simple background" in nina_identity

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
