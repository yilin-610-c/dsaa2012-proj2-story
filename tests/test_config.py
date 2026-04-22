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

    assert strong["prompt"]["pipeline"] == "rule_based"
    assert strong["model"]["backend"] == "diffusers_text2img"
    assert strong["model"]["granularity"] == "scene"
    assert story["model"]["backend"] == "storydiffusion_direct"
    assert story["model"]["granularity"] == "story"


def test_resolve_config_unknown_profile_lists_available_profiles() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_config("configs/base.yaml", "missing_profile")

    message = str(exc_info.value)
    assert "missing_profile" in message
    assert "smoke_test" in message
    assert "cloud_storydiffusion" in message
