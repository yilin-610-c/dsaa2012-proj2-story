from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_TARGET_BACKENDS = {"anchor", "storydiffusion"}
ALLOWED_SUBJECT_TYPES = {"human", "animal", "robot", "object", "vehicle", "unknown"}


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


@dataclass(slots=True)
class NativeStoryDiffusionPromptPayload:
    general_prompt: str = ""
    identity_reference_prompts: list[str] = field(default_factory=list)
    identity_prompts_per_character: int = 1
    negative_prompt_extra: str = ""


@dataclass(slots=True)
class NativePromptPayload:
    target_backends: list[str]
    characters: list[NativeCharacterPrompt]
    scenes: list[NativeScenePrompt]
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
            )
            for item in payload.get("scenes", [])
            if isinstance(item, dict)
        ]
        storydiffusion_payload = payload.get("storydiffusion") if isinstance(payload.get("storydiffusion"), dict) else {}
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
            negative_prompt_extra=str(storydiffusion_payload.get("negative_prompt_extra", "")).strip(),
        )
        return cls(
            target_backends=[str(value).strip().lower() for value in payload.get("target_backends", [])],
            characters=characters,
            scenes=scenes,
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
                }
                for scene in self.scenes
            ],
            "storydiffusion": {
                "general_prompt": self.storydiffusion.general_prompt,
                "identity_reference_prompts": list(self.storydiffusion.identity_reference_prompts),
                "identity_prompts_per_character": self.storydiffusion.identity_prompts_per_character,
                "negative_prompt_extra": self.storydiffusion.negative_prompt_extra,
            },
            "notes": dict(self.notes),
        }


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
