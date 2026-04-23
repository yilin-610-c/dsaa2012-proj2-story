from pathlib import Path

from storygen.anchor_bank import build_anchor_bank_plan, build_anchor_prompt
from storygen.types import RunContext


def _run_context(tmp_path: Path) -> RunContext:
    return RunContext(
        run_name="test",
        output_root=tmp_path,
        run_directory=tmp_path / "run",
        scenes_directory=tmp_path / "run" / "scenes",
        logs_directory=tmp_path / "run" / "logs",
    )


def _anchor_config(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "generate": False,
        "anchor_types": ["portrait", "half_body"],
        "output_dir_name": "anchors",
        "base_seed_offset": 900000,
        "prompt_suffix": "clean identity reference image",
    }


def _character_specs() -> dict:
    return {
        "Jane Doe": {
            "character_id": "Jane Doe",
            "age_band": "adult",
            "gender_presentation": "woman",
            "hair_color": "brown hair",
            "hairstyle": "short bob",
            "skin_tone": "warm skin tone",
            "body_build": "slim build",
            "signature_outfit": "green jacket",
            "signature_accessory": "round glasses",
            "profession_marker": "",
            "metadata": {"source": "test"},
        }
    }


def test_anchor_bank_plan_empty_without_character_specs(tmp_path: Path) -> None:
    plan = build_anchor_bank_plan(
        character_specs={},
        anchor_config=_anchor_config(),
        run_context=_run_context(tmp_path),
        prompt_config={"negative_prompt": "blurry"},
    )

    assert plan["enabled"] is True
    assert plan["characters"] == {}


def test_anchor_bank_plan_builds_portrait_and_half_body(tmp_path: Path) -> None:
    plan = build_anchor_bank_plan(
        character_specs=_character_specs(),
        anchor_config=_anchor_config(),
        run_context=_run_context(tmp_path),
        prompt_config={"negative_prompt": "blurry"},
    )

    character = plan["characters"]["Jane Doe"]
    assert character["safe_character_id"] == "Jane_Doe"
    assert set(character["anchors"]) == {"portrait", "half_body"}
    assert character["anchor_spec"]["portrait_path"].endswith("anchors/Jane_Doe/portrait.png")
    assert character["anchor_spec"]["half_body_path"].endswith("anchors/Jane_Doe/half_body.png")


def test_anchor_prompt_uses_stable_identity_fields_only() -> None:
    prompt = build_anchor_prompt(
        {
            **_character_specs()["Jane Doe"],
            "route_reason": "scene action should not appear",
            "primary_action": "running through rain",
        },
        "portrait",
        "clean identity reference image",
    )

    assert "Jane Doe" in prompt
    assert "green jacket" in prompt
    assert "round glasses" in prompt
    assert "running through rain" not in prompt
    assert "scene action should not appear" not in prompt
