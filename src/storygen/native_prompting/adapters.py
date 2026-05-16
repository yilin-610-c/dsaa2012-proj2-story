from __future__ import annotations

from typing import Any

from storygen.native_prompting.types import NativePromptPayload
from storygen.types import PromptBundle, PromptSpec, Story


def build_anchor_prompt_bundle(payload: NativePromptPayload, story: Story, prompt_config: dict[str, Any]) -> PromptBundle:
    scene_by_id = {scene.scene_id: scene for scene in payload.scenes}
    scene_prompts: dict[str, PromptSpec] = {}
    for scene in story.scenes:
        native_scene = scene_by_id[scene.scene_id]
        scene_prompts[scene.scene_id] = PromptSpec(
            scene_id=scene.scene_id,
            style_prompt="",
            character_prompt="",
            global_context_prompt="",
            scene_consistency_prompt="",
            local_prompt=native_scene.anchor_generation_prompt,
            action_prompt=native_scene.action_prompt,
            generation_prompt=native_scene.anchor_generation_prompt,
            scoring_prompt=native_scene.scoring_prompt,
            full_prompt=native_scene.anchor_generation_prompt,
            negative_prompt=str(prompt_config.get("negative_prompt", "")).strip(),
        )
    character_specs = {
        character.character_id: {
            "character_id": character.character_id,
            "subject_type": character.subject_type,
            "stable_identity": character.stable_identity,
            "anchor_reference_prompt": character.anchor_reference_prompt,
            "metadata": {"source": "llm_direct"},
        }
        for character in payload.characters
    }
    scene_route_hints = {
        scene.scene_id: {
            "primary_visible_character_ids": list(scene.visible_character_ids),
            "visible_character_ids": list(scene.visible_character_ids),
            "identity_conditioning_subject_id": scene.identity_conditioning_subject_id,
        }
        for scene in payload.scenes
    }
    return PromptBundle(
        scene_prompts=scene_prompts,
        story_prompt=None,
        metadata={
            "pipeline": "llm_direct",
            "implemented": True,
            "target_backends": list(payload.target_backends),
            "character_specs": character_specs,
            "scene_route_hints": scene_route_hints,
            "native_prompt_payload": payload.to_dict(),
        },
    )


def build_storydiffusion_prompt_payload(payload: NativePromptPayload, story: Story) -> dict[str, Any]:
    story_scene_prompts = [scene.storydiffusion_prompt for scene in payload.scenes]
    identity_reference_prompts = list(payload.storydiffusion.identity_reference_prompts)
    identity_prompt_count = len(identity_reference_prompts)
    identity_prompts_per_character = int(payload.storydiffusion.identity_prompts_per_character)
    character_count = len(payload.characters)
    prompt_array = identity_reference_prompts + story_scene_prompts
    save_image_start_index = identity_prompt_count
    saved_image_prompt_map = {
        f"image_{index:03d}.png": {
            "prompt_array_index": save_image_start_index + index,
            "story_scene_prompt_index": index,
            "scene_id": story.scenes[index].scene_id if index < len(story.scenes) else None,
            "prompt": prompt,
        }
        for index, prompt in enumerate(story_scene_prompts)
    }
    debug = {
        "mode": "llm_direct",
        "general_prompt": payload.storydiffusion.general_prompt,
        "identity_reference_prompts": identity_reference_prompts,
        "story_scene_prompts": story_scene_prompts,
        "identity_prompt_count": identity_prompt_count,
        "identity_prompts_per_character": identity_prompts_per_character,
        "character_count": character_count,
        "story_frame_start_index": save_image_start_index,
        "save_image_start_index": save_image_start_index,
        "storydiffusion_id_length": identity_prompts_per_character,
        "saved_image_prompt_map": saved_image_prompt_map,
        "final_prompt_array": prompt_array,
        "identity_negative_prompt_extra": payload.storydiffusion.identity_negative_prompt_extra,
        "scene_negative_prompt_extra": payload.storydiffusion.scene_negative_prompt_extra,
        "native_prompt_payload": payload.to_dict(),
    }
    return {
        "general_prompt": payload.storydiffusion.general_prompt,
        "identity_prompts": identity_reference_prompts,
        "scene_prompts": story_scene_prompts,
        "identity_reference_prompts": identity_reference_prompts,
        "story_scene_prompts": story_scene_prompts,
        "identity_prompt_count": identity_prompt_count,
        "identity_prompts_per_character": identity_prompts_per_character,
        "character_count": character_count,
        "story_frame_start_index": save_image_start_index,
        "saved_image_prompt_map": saved_image_prompt_map,
        "prompt_array": prompt_array,
        "save_image_start_index": save_image_start_index,
        "storydiffusion_id_length": identity_prompts_per_character,
        "identity_negative_prompt_extra": payload.storydiffusion.identity_negative_prompt_extra,
        "scene_negative_prompt_extra": payload.storydiffusion.scene_negative_prompt_extra,
        "negative_prompt_extra": payload.storydiffusion.scene_negative_prompt_extra,
        "debug": debug,
    }
