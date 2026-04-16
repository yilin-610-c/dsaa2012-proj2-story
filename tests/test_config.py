from storygen.config import resolve_config


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
