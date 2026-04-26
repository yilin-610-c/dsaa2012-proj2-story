from storygen.config import resolve_config
import pytest


def test_resolve_config_applies_runtime_profile_and_overrides() -> None:
    config = resolve_config(
        "configs/base.yaml",
        "demo_run",
        overrides={
            "runtime.input_path": "test_set/01.txt",
            "generation.base_seed": 999,
        },
    )

    assert config["runtime"]["profile"] == "demo_run"
    assert config["generation"]["candidate_count"] == 4
    assert config["generation"]["base_seed"] == 999
    assert config["model"]["width"] == 768
    assert config["scoring"]["type"] == "clip_consistency"


def test_resolve_config_supports_extension_profiles() -> None:
    strong = resolve_config("configs/base.yaml", "cloud_strong_backbone")
    story = resolve_config("configs/base.yaml", "cloud_storydiffusion")
    guided = resolve_config("configs/base.yaml", "llm_prompt_img2img_guided")
    ip_adapter = resolve_config("configs/base.yaml", "llm_prompt_ip_adapter_text2img")
    hybrid = resolve_config("configs/base.yaml", "llm_prompt_hybrid_identity")
    two_character = resolve_config("configs/base.yaml", "llm_prompt_two_character_text2img")

    assert strong["prompt"]["pipeline"] == "rule_based"
    assert strong["model"]["backend"] == "diffusers_text2img"
    assert strong["model"]["granularity"] == "scene"
    assert story["model"]["backend"] == "storydiffusion_direct"
    assert story["model"]["granularity"] == "story"
    assert guided["prompt"]["pipeline"] == "llm_assisted"
    assert guided["generation"]["routing"]["route_policy"] == "llm_guided_conservative"
    assert guided["generation"]["routing"]["strength_by_change_level"]["medium"] == 0.65
    assert guided["generation"]["routing"]["text2img_when_composition_change_needed"] is True
    assert guided["prompt"]["llm"]["builder_version"] == "llm_assisted_v9"
    assert guided["prompt"]["artifact"]["export_dir"] == "prompt_artifacts/llm_assisted_v9"
    assert guided["scoring"]["route_aware"]["enabled"] is True
    assert ip_adapter["generation"]["identity_conditioning"]["enabled"] is True
    assert ip_adapter["generation"]["identity_conditioning"]["anchor_type"] == "half_body"
    assert ip_adapter["generation"]["identity_conditioning"]["apply_to_modes"] == ["text2img"]
    assert ip_adapter["generation"]["routing"]["img2img_enabled"] is False
    assert ip_adapter["model"]["enable_attention_slicing"] is False
    assert hybrid["generation"]["identity_conditioning"]["enabled"] is True
    assert hybrid["generation"]["routing"]["route_policy"] == "llm_guided_conservative"
    assert hybrid["model"]["enable_attention_slicing"] is False
    assert two_character["prompt"]["pipeline"] == "llm_assisted"
    assert two_character["generation"]["routing"]["img2img_enabled"] is False
    assert two_character["generation"]["routing"]["route_policy"] == "disabled"
    assert two_character["generation"]["anchor_bank"]["enabled"] is True
    assert two_character["generation"]["identity_conditioning"]["enabled"] is True
    assert two_character["generation"]["identity_conditioning"]["max_primary_visible_characters_for_ip_adapter"] == 1


def test_resolve_config_unknown_profile_lists_available_profiles() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_config("configs/base.yaml", "missing_profile")

    message = str(exc_info.value)
    assert "missing_profile" in message
    assert "smoke_test" in message
    assert "cloud_storydiffusion" in message
