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


def test_select_identity_anchor_uses_route_hint_subject_for_single_primary_scene(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene(["Sara"]),
        route_hint={
            "continuity_subject_ids": ["Jack"],
            "primary_visible_character_ids": ["Jack"],
            "policy": {"visible_character_count": 1, "scene_focus_mode": "single_primary"},
        },
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
            "primary_visible_character_ids": ["Jack"],
            "continuity_subject_ids": ["Sara"],
            "policy": {"visible_character_count": 1, "scene_focus_mode": "single_primary"},
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
        route_hint={
            "primary_visible_character_ids": ["Sara"],
            "policy": {"visible_character_count": 1, "scene_focus_mode": "single_primary"},
        },
        generation_mode="text2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(),
    )

    assert result["identity_anchor_character_id"] == "Sara"
    assert result["identity_conditioning_reason"] == "scene_entity"


def test_select_identity_anchor_skips_mode_not_enabled(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene(["Jack"]),
        route_hint={"continuity_subject_ids": ["Jack"], "primary_visible_character_ids": ["Jack"]},
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
        route_hint={"primary_visible_character_ids": ["Lily"]},
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
    assert result["identity_conditioning_reason"] == "policy_skip:no_clear_primary_visible_character"


def test_select_identity_anchor_ambiguous_multi_character_with_visible_characters_skips(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene([]),
        route_hint={
            "primary_visible_character_ids": ["Jack", "Sara"],
            "identity_conditioning_subject_id": None,
            "policy": {"visible_character_count": 2, "scene_focus_mode": "dual_primary"},
        },
        generation_mode="text2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(fail_on_missing_anchor=False),
    )

    assert result["identity_conditioning_enabled"] is False
    assert result["identity_conditioning_reason"] == "policy_skip:dual_primary_scene"


def test_select_identity_anchor_single_primary_prefers_visible_character(tmp_path: Path) -> None:
    result = select_identity_anchor(
        scene=_scene([]),
        route_hint={
            "primary_visible_character_ids": ["Sara"],
            "identity_conditioning_subject_id": None,
            "policy": {"visible_character_count": 1, "scene_focus_mode": "single_primary"},
        },
        generation_mode="text2img",
        anchor_bank_summary=_anchor_bank(tmp_path),
        identity_config=_identity_config(),
    )

    assert result["identity_conditioning_enabled"] is True
    assert result["identity_anchor_character_id"] == "Sara"
    assert result["identity_conditioning_reason"] == "primary_visible_character"


def test_select_identity_anchor_missing_file_raises_when_configured(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing_anchor_file"):
        select_identity_anchor(
            scene=_scene(["Jack"]),
            route_hint={"primary_visible_character_ids": ["Jack"]},
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
        route_hint={"primary_visible_character_ids": ["Jack"]},
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
