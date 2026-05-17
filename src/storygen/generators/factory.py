from __future__ import annotations

from typing import Any

from .base import BaseSceneGenerator, BaseStoryGenerator
from .diffusers_text2img import DiffusersTextToImageGenerator
from .diffusers_text2img_consistent import DiffusersTextToImageConsistentAttentionGenerator
from .dit_text2img import DitTextToImageGenerator
from .dit_story_joint import DitStoryJointGenerator
from .storydiffusion_direct import StoryDiffusionDirectGenerator


def build_generation_backend(
    model_config: dict[str, Any],
    runtime_config: dict[str, Any],
) -> BaseSceneGenerator | BaseStoryGenerator:
    backend_type = model_config.get("backend")
    granularity = model_config.get("granularity", "scene")

    if backend_type == "diffusers_text2img" and granularity == "scene":
        if bool(model_config.get("consistent_attention", {}).get("enabled", False)):
            return DiffusersTextToImageConsistentAttentionGenerator(model_config, runtime_config)
        return DiffusersTextToImageGenerator(model_config, runtime_config)
    if backend_type == "dit_text2img" and granularity == "scene":
        return DitTextToImageGenerator(model_config, runtime_config)
    if backend_type == "storydiffusion_direct" and granularity == "story":
        return StoryDiffusionDirectGenerator(model_config, runtime_config)
    if backend_type == "dit_story_joint" and granularity == "story":
        return DitStoryJointGenerator(model_config, runtime_config)
    if backend_type == "diffusers_text2img":
        raise ValueError("diffusers_text2img only supports model.granularity='scene'")
    if backend_type == "dit_text2img":
        raise ValueError("dit_text2img only supports model.granularity='scene'")
    if backend_type == "storydiffusion_direct":
        raise ValueError("storydiffusion_direct only supports model.granularity='story'")
    if backend_type == "dit_story_joint":
        raise ValueError("dit_story_joint only supports model.granularity='story'")
    raise ValueError(f"Unsupported generator backend: {backend_type}")


def build_backend_metadata(model_config: dict[str, Any], runtime_config: dict[str, Any]) -> dict[str, Any]:
    backend = model_config.get("backend")
    return {
        "backend": backend,
        "granularity": model_config.get("granularity", "scene"),
        "implemented": backend in {"diffusers_text2img", "storydiffusion_direct", "dit_text2img", "dit_story_joint"},
        "model_id": model_config.get("model_id"),
        "device": runtime_config.get("device"),
        "torch_dtype": runtime_config.get("torch_dtype"),
    }
