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
        "scale_by_subject_type": {
            "animal": 0.1,
            "robot": 0.1,
            "object": 0.1,
            "vehicle": 0.1,
        },
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
            "Jack": {"character_spec": {"subject_type": "human"}, "anchors": {"half_body": {"image_path": str(jack)}}},
            "Sara": {"character_spec": {"subject_type": "human"}, "anchors": {"half_body": {"image_path": str(sara)}}},
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


def test_select_identity_anchor_prefers_identity_subject_id(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene(["Sara"]),
        route_hint={
            "identity_conditioning_subject_id": "Jack",
            "continuity_subject_ids": ["Sara"],
        },
        generation_mode="text2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(),
    )

    assert result["identity_conditioning_enabled"] is True
    assert result["identity_anchor_character_id"] == "Jack"
    assert result["identity_conditioning_reason"] == "identity_subject_id"


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


def test_select_identity_anchor_ambiguous_multi_character_with_visible_characters_skips(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene([]),
        route_hint={"primary_visible_character_ids": ["Jack", "Sara"], "identity_conditioning_subject_id": None},
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


def test_select_identity_anchor_prefers_canonical_half_body_path(tmp_path: Path) -> None:
    canonical = tmp_path / "anchors" / "Jack" / "canonical_half_body.png"
    legacy = tmp_path / "anchors" / "Jack" / "half_body.png"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"fake")
    legacy.write_bytes(b"legacy")

    result = select_identity_anchor(
        scene=_scene(["Jack"]),
        route_hint={},
        generation_mode="text2img",
        anchor_bank_summary={
            "characters": {
                "Jack": {
                    "anchors": {
                        "half_body": {
                            "image_path": str(legacy),
                            "canonical_image_path": str(canonical),
                        }
                    }
                }
            }
        },
        identity_config=_identity_config(),
    )

    assert result["identity_anchor_path"] == str(canonical)


def test_non_human_subject_type_uses_configured_ip_adapter_scale(tmp_path: Path) -> None:
    anchor = tmp_path / "anchors" / "Bird" / "half_body.png"
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(b"fake")

    result = select_identity_anchor(
        scene=_scene(["Bird"]),
        route_hint={},
        generation_mode="text2img",
        anchor_bank_summary={
            "characters": {
                "Bird": {
                    "character_spec": {"subject_type": "animal"},
                    "anchors": {"half_body": {"image_path": str(anchor)}},
                }
            }
        },
        identity_config=_identity_config(),
    )

    assert result["ip_adapter_scale"] == 0.1
    assert result["identity_anchor_subject_type"] == "animal"
    assert result["ip_adapter_scale_reason"] == "scale_by_subject_type:animal"


def test_human_subject_type_keeps_default_ip_adapter_scale(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene(["Jack"]),
        route_hint={},
        generation_mode="text2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(),
    )

    assert result["ip_adapter_scale"] == 0.6
    assert result["identity_anchor_subject_type"] == "human"
    assert result["ip_adapter_scale_reason"] == "default_scale"


def test_subject_type_scale_can_be_overridden(tmp_path: Path) -> None:
    anchor = tmp_path / "anchors" / "Robot" / "half_body.png"
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(b"fake")
    config = _identity_config()
    config["scale_by_subject_type"]["robot"] = 0.2

    result = select_identity_anchor(
        scene=_scene(["Robot"]),
        route_hint={},
        generation_mode="text2img",
        anchor_bank_summary={
            "characters": {
                "Robot": {
                    "character_spec": {"subject_type": "robot"},
                    "anchors": {"half_body": {"image_path": str(anchor)}},
                }
            }
        },
        identity_config=config,
    )

    assert result["ip_adapter_scale"] == 0.2
    assert result["ip_adapter_scale_reason"] == "scale_by_subject_type:robot"
