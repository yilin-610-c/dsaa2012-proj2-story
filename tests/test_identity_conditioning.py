from pathlib import Path

import pytest

from storygen.identity_conditioning import select_identity_anchor
from storygen.types import Scene


def _scene(entities: list[str] | None = None) -> Scene:
    return Scene("SCENE-1", 0, "raw", "clean", entities or [])


def _identity_config(*, fail_on_missing_anchor: bool = True) -> dict:
    return {
        "enabled": True,
        "adapter_type": "ip_adapter",
        "anchor_source": "character_anchor_bank",
        "anchor_type": "half_body",
        "apply_to_modes": ["text2img"],
        "scale": 0.6,
        "adapter_model_id": "h94/IP-Adapter",
        "adapter_subfolder": "sdxl_models",
        "adapter_weight_name": "ip-adapter_sdxl.bin",
        "fail_on_missing_anchor": fail_on_missing_anchor,
    }


def _anchor_bank(tmp_path: Path) -> dict:
    jack = tmp_path / "anchors" / "Jack" / "half_body.png"
    sara = tmp_path / "anchors" / "Sara" / "half_body.png"
    jack.parent.mkdir(parents=True)
    sara.parent.mkdir(parents=True)
    jack.write_bytes(b"fake")
    sara.write_bytes(b"fake")
    return {
        "characters": {
            "Jack": {"anchors": {"half_body": {"image_path": str(jack)}}},
            "Sara": {"anchors": {"half_body": {"image_path": str(sara)}}},
        }
    }


def test_select_identity_anchor_uses_route_hint_subject(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene(["Sara"]),
        route_hint={"continuity_subject_ids": ["Jack"]},
        generation_mode="text2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(),
    )

    assert result["identity_conditioning_enabled"] is True
    assert result["identity_anchor_character_id"] == "Jack"
    assert result["identity_anchor_type"] == "half_body"
    assert result["identity_conditioning_reason"] == "route_hint_subject"


def test_select_identity_anchor_falls_back_to_scene_entity(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene(["Sara"]),
        route_hint={},
        generation_mode="text2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(),
    )

    assert result["identity_anchor_character_id"] == "Sara"
    assert result["identity_conditioning_reason"] == "scene_entity"


def test_select_identity_anchor_skips_mode_not_enabled(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene(["Jack"]),
        route_hint={"continuity_subject_ids": ["Jack"]},
        generation_mode="img2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(),
    )

    assert result["identity_conditioning_enabled"] is False
    assert result["identity_conditioning_reason"] == "generation_mode_not_enabled:img2img"


def test_select_identity_anchor_single_character_fallback(tmp_path: Path) -> None:
    anchor = tmp_path / "anchors" / "Lily" / "half_body.png"
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(b"fake")
    result = select_identity_anchor(
        scene=_scene([]),
        route_hint={},
        generation_mode="text2img",
        anchor_bank_summary={"characters": {"Lily": {"anchors": {"half_body": {"image_path": str(anchor)}}}}},
        identity_config=_identity_config(),
    )

    assert result["identity_anchor_character_id"] == "Lily"
    assert result["identity_conditioning_reason"] == "single_anchor_character"


def test_select_identity_anchor_ambiguous_multi_character_can_skip(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene([]),
        route_hint={},
        generation_mode="text2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(fail_on_missing_anchor=False),
    )

    assert result["identity_conditioning_enabled"] is False
    assert result["identity_conditioning_reason"] == "ambiguous_or_missing_scene_character"


def test_select_identity_anchor_missing_file_raises_when_configured(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing_anchor_file"):
        select_identity_anchor(
            scene=_scene(["Jack"]),
            route_hint={},
            generation_mode="text2img",
            anchor_bank_summary={"characters": {"Jack": {"anchors": {"half_body": {"image_path": str(tmp_path / "missing.png")}}}}},
            identity_config=_identity_config(),
        )
