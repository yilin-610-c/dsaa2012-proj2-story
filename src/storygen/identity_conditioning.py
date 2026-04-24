from __future__ import annotations

from pathlib import Path
from typing import Any

from storygen.types import Scene


def _normalized_anchor_characters(anchor_bank_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    characters = anchor_bank_summary.get("characters", {})
    if not isinstance(characters, dict):
        return {}
    return {str(character_id): payload for character_id, payload in characters.items() if isinstance(payload, dict)}


def _single_matching_character(
    candidates: list[Any],
    anchor_characters: dict[str, dict[str, Any]],
) -> str | None:
    normalized = {character_id.lower(): character_id for character_id in anchor_characters}
    matches = []
    seen = set()
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            match = normalized[key]
            match_key = match.lower()
            if match_key not in seen:
                seen.add(match_key)
                matches.append(match)
    return matches[0] if len(matches) == 1 else None


def _resolve_anchor_image_path(anchor_payload: dict[str, Any] | None) -> str:
    if not isinstance(anchor_payload, dict):
        return ""
    canonical_anchor = anchor_payload.get("canonical_anchor")
    if isinstance(canonical_anchor, dict):
        canonical_selected_path = str(canonical_anchor.get("selected_image_path") or "").strip()
        if canonical_selected_path:
            return canonical_selected_path
    canonical_path = str(anchor_payload.get("canonical_image_path") or "").strip()
    if canonical_path:
        return canonical_path
    return str(anchor_payload.get("image_path") or "").strip()


def select_identity_anchor(
    *,
    scene: Scene,
    route_hint: dict[str, Any] | None,
    generation_mode: str,
    anchor_bank_summary: dict[str, Any],
    identity_config: dict[str, Any],
) -> dict[str, Any]:
    if not identity_config.get("enabled", False):
        return {"identity_conditioning_enabled": False, "identity_conditioning_reason": "identity_conditioning_disabled"}

    apply_to_modes = {str(mode) for mode in identity_config.get("apply_to_modes", ["text2img"])}
    if generation_mode not in apply_to_modes:
        return {
            "identity_conditioning_enabled": False,
            "identity_conditioning_reason": f"generation_mode_not_enabled:{generation_mode}",
        }
    if identity_config.get("adapter_type", "ip_adapter") != "ip_adapter":
        raise ValueError(f"Unsupported identity conditioning adapter_type: {identity_config.get('adapter_type')}")
    if identity_config.get("anchor_source", "character_anchor_bank") != "character_anchor_bank":
        raise ValueError(f"Unsupported identity anchor_source: {identity_config.get('anchor_source')}")

    anchor_characters = _normalized_anchor_characters(anchor_bank_summary)
    if not anchor_characters:
        return _missing_anchor_result(identity_config, "no_anchor_characters")

    route_hint = route_hint or {}
    selected_character_id = _single_matching_character(
        [route_hint.get("identity_conditioning_subject_id")],
        anchor_characters,
    )
    reason = "identity_subject_id"
    if selected_character_id is None:
        selected_character_id = _single_matching_character(
            list(route_hint.get("continuity_subject_ids", [])),
            anchor_characters,
        )
        reason = "route_hint_subject"
    if selected_character_id is None:
        selected_character_id = _single_matching_character(scene.entities, anchor_characters)
        reason = "scene_entity"
    if selected_character_id is None and len(anchor_characters) == 1:
        selected_character_id = next(iter(anchor_characters))
        reason = "single_anchor_character"
    if selected_character_id is None:
        return _missing_anchor_result(identity_config, "ambiguous_or_missing_scene_character")

    anchor_type = str(identity_config.get("anchor_type", "half_body"))
    character_payload = anchor_characters[selected_character_id]
    anchors = character_payload.get("anchors", {})
    anchor_payload = anchors.get(anchor_type) if isinstance(anchors, dict) else None
    anchor_path = _resolve_anchor_image_path(anchor_payload)
    if not anchor_path:
        return _missing_anchor_result(identity_config, f"missing_anchor_type:{selected_character_id}:{anchor_type}")
    if not Path(anchor_path).exists():
        return _missing_anchor_result(identity_config, f"missing_anchor_file:{anchor_path}")

    return {
        "identity_conditioning_enabled": True,
        "identity_anchor_character_id": selected_character_id,
        "identity_anchor_type": anchor_type,
        "identity_anchor_path": anchor_path,
        "identity_conditioning_reason": reason,
        "ip_adapter_scale": float(identity_config.get("scale", 0.6)),
        "ip_adapter_model_id": identity_config.get("adapter_model_id"),
        "ip_adapter_subfolder": identity_config.get("adapter_subfolder"),
        "ip_adapter_weight_name": identity_config.get("adapter_weight_name"),
        "identity_apply_to_modes": list(identity_config.get("apply_to_modes", ["text2img"])),
    }


def _missing_anchor_result(identity_config: dict[str, Any], reason: str) -> dict[str, Any]:
    if identity_config.get("fail_on_missing_anchor", True):
        raise ValueError(f"Identity conditioning anchor unavailable: {reason}")
    return {
        "identity_conditioning_enabled": False,
        "identity_conditioning_reason": reason,
    }
