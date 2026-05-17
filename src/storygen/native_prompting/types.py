from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_TARGET_BACKENDS = {"anchor", "storydiffusion"}
ALLOWED_SUBJECT_TYPES = {"human", "animal", "robot", "object", "vehicle", "unknown"}
ALLOWED_SCENE_CHANGE_LEVELS = {"small", "medium", "large"}
ALLOWED_CAMERA_FRAMINGS = {
    "wide shot",
    "medium-wide shot",
    "medium shot",
    "medium close-up shot",
    "close-up shot",
}


@dataclass(slots=True)
class NativeVisualContinuityAnchor:
    anchor_id: str
    type: str
    applies_to_scene_ids: list[str] = field(default_factory=list)
    prompt_phrase: str = ""
    state_by_scene: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class NativeCharacterPrompt:
    character_id: str
    subject_type: str
    stable_identity: str
    anchor_reference_prompt: str = ""


@dataclass(slots=True)
class NativeScenePrompt:
    scene_id: str
    visible_character_ids: list[str] = field(default_factory=list)
    identity_conditioning_subject_id: str | None = None
    anchor_generation_prompt: str = ""
    storydiffusion_prompt: str = ""
    scoring_prompt: str = ""
    action_prompt: str = ""
    scene_visual_plan: dict[str, str] = field(default_factory=dict)
    scene_change_level: str = ""
    action_critical: bool = False


@dataclass(slots=True)
class NativeStoryDiffusionPromptPayload:
    general_prompt: str = ""
    identity_reference_prompts: list[str] = field(default_factory=list)
    identity_prompts_per_character: int = 1
    identity_negative_prompt_extra: str = ""
    scene_negative_prompt_extra: str = ""

    @property
    def negative_prompt_extra(self) -> str:
        return self.scene_negative_prompt_extra


@dataclass(slots=True)
class NativePromptPayload:
    target_backends: list[str]
    characters: list[NativeCharacterPrompt]
    scenes: list[NativeScenePrompt]
    visual_continuity_anchors: list[NativeVisualContinuityAnchor] = field(default_factory=list)
    storydiffusion: NativeStoryDiffusionPromptPayload = field(default_factory=NativeStoryDiffusionPromptPayload)
    notes: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NativePromptPayload":
        characters = [
            NativeCharacterPrompt(
                character_id=str(item.get("character_id", "")).strip(),
                subject_type=str(item.get("subject_type", "")).strip().lower(),
                stable_identity=str(item.get("stable_identity", "")).strip(),
                anchor_reference_prompt=str(item.get("anchor_reference_prompt", "")).strip(),
            )
            for item in payload.get("characters", [])
            if isinstance(item, dict)
        ]
        scenes = [
            NativeScenePrompt(
                scene_id=str(item.get("scene_id", "")).strip(),
                visible_character_ids=[str(value).strip() for value in item.get("visible_character_ids", []) if str(value).strip()]
                if isinstance(item.get("visible_character_ids", []), list)
                else [],
                identity_conditioning_subject_id=(
                    str(item.get("identity_conditioning_subject_id")).strip()
                    if item.get("identity_conditioning_subject_id") is not None
                    else None
                ),
                anchor_generation_prompt=str(item.get("anchor_generation_prompt", "")).strip(),
                storydiffusion_prompt=str(item.get("storydiffusion_prompt", "")).strip(),
                scoring_prompt=str(item.get("scoring_prompt", "")).strip(),
                action_prompt=str(item.get("action_prompt", "")).strip(),
                scene_visual_plan=_scene_visual_plan_from_dict(item.get("scene_visual_plan")),
                scene_change_level=str(item.get("scene_change_level", "")).strip().lower(),
                action_critical=bool(item.get("action_critical", False)),
            )
            for item in payload.get("scenes", [])
            if isinstance(item, dict)
        ]
        visual_continuity_anchors = [
            NativeVisualContinuityAnchor(
                anchor_id=str(item.get("anchor_id", "")).strip(),
                type=str(item.get("type", "")).strip(),
                applies_to_scene_ids=[
                    str(value).strip()
                    for value in item.get("applies_to_scene_ids", [])
                    if str(value).strip()
                ]
                if isinstance(item.get("applies_to_scene_ids", []), list)
                else [],
                prompt_phrase=str(item.get("prompt_phrase", "")).strip(),
                state_by_scene=_state_by_scene_from_value(item.get("state_by_scene")),
            )
            for item in payload.get("visual_continuity_anchors", [])
            if isinstance(item, dict)
        ]
        storydiffusion_payload = payload.get("storydiffusion") if isinstance(payload.get("storydiffusion"), dict) else {}
        legacy_negative_prompt_extra = str(storydiffusion_payload.get("negative_prompt_extra", "")).strip()
        storydiffusion = NativeStoryDiffusionPromptPayload(
            general_prompt=str(storydiffusion_payload.get("general_prompt", "")).strip(),
            identity_reference_prompts=[
                str(value).strip()
                for value in storydiffusion_payload.get("identity_reference_prompts", [])
                if str(value).strip()
            ]
            if isinstance(storydiffusion_payload.get("identity_reference_prompts", []), list)
            else [],
            identity_prompts_per_character=_safe_int(storydiffusion_payload.get("identity_prompts_per_character", 1), default=0),
            identity_negative_prompt_extra=str(
                storydiffusion_payload.get("identity_negative_prompt_extra", legacy_negative_prompt_extra)
            ).strip(),
            scene_negative_prompt_extra=str(storydiffusion_payload.get("scene_negative_prompt_extra", "")).strip(),
        )
        return cls(
            target_backends=[str(value).strip().lower() for value in payload.get("target_backends", [])],
            characters=characters,
            scenes=scenes,
            visual_continuity_anchors=visual_continuity_anchors,
            storydiffusion=storydiffusion,
            notes=payload.get("notes") if isinstance(payload.get("notes"), dict) else {},
            raw_payload=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_backends": list(self.target_backends),
            "characters": [
                {
                    "character_id": character.character_id,
                    "subject_type": character.subject_type,
                    "stable_identity": character.stable_identity,
                    "anchor_reference_prompt": character.anchor_reference_prompt,
                }
                for character in self.characters
            ],
            "scenes": [
                {
                    "scene_id": scene.scene_id,
                    "visible_character_ids": list(scene.visible_character_ids),
                    "identity_conditioning_subject_id": scene.identity_conditioning_subject_id,
                    "anchor_generation_prompt": scene.anchor_generation_prompt,
                    "storydiffusion_prompt": scene.storydiffusion_prompt,
                    "scoring_prompt": scene.scoring_prompt,
                    "action_prompt": scene.action_prompt,
                    "scene_visual_plan": dict(scene.scene_visual_plan),
                    "scene_change_level": scene.scene_change_level,
                    "action_critical": scene.action_critical,
                }
                for scene in self.scenes
            ],
            "visual_continuity_anchors": [
                {
                    "anchor_id": anchor.anchor_id,
                    "type": anchor.type,
                    "applies_to_scene_ids": list(anchor.applies_to_scene_ids),
                    "prompt_phrase": anchor.prompt_phrase,
                    "state_by_scene": dict(anchor.state_by_scene),
                }
                for anchor in self.visual_continuity_anchors
            ],
            "storydiffusion": {
                "general_prompt": self.storydiffusion.general_prompt,
                "identity_reference_prompts": list(self.storydiffusion.identity_reference_prompts),
                "identity_prompts_per_character": self.storydiffusion.identity_prompts_per_character,
                "identity_negative_prompt_extra": self.storydiffusion.identity_negative_prompt_extra,
                "scene_negative_prompt_extra": self.storydiffusion.scene_negative_prompt_extra,
            },
            "notes": dict(self.notes),
        }


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scene_visual_plan_from_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        "visual_action": str(value.get("visual_action", "")).strip(),
        "action_visibility_cue": str(value.get("action_visibility_cue", "")).strip(),
        "camera_framing": str(value.get("camera_framing", "")).strip(),
    }


def _state_by_scene_from_value(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            str(key).strip(): str(item).strip()
            for key, item in value.items()
            if str(key).strip() and str(item).strip()
        }
    if isinstance(value, list):
        result: dict[str, str] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            scene_id = str(item.get("scene_id", "")).strip()
            phrase = str(item.get("prompt_phrase", item.get("state", ""))).strip()
            if scene_id and phrase:
                result[scene_id] = phrase
        return result
    return {}
