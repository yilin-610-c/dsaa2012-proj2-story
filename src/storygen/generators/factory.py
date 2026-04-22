from __future__ import annotations

from typing import Any

from .base import BaseSceneGenerator, BaseStoryGenerator
from .diffusers_text2img import DiffusersTextToImageGenerator
from .storydiffusion_direct import StoryDiffusionDirectGenerator


def build_generation_backend(
    model_config: dict[str, Any],
    runtime_config: dict[str, Any],
) -> BaseSceneGenerator | BaseStoryGenerator:
    backend_type = model_config.get("backend")
    granularity = model_config.get("granularity", "scene")

    if backend_type == "diffusers_text2img" and granularity == "scene":
        return DiffusersTextToImageGenerator(model_config, runtime_config)
    if backend_type == "storydiffusion_direct" and granularity == "story":
        return StoryDiffusionDirectGenerator(model_config, runtime_config)
    if backend_type == "diffusers_text2img":
        raise ValueError("diffusers_text2img only supports model.granularity='scene'")
    if backend_type == "storydiffusion_direct":
        raise ValueError("storydiffusion_direct only supports model.granularity='story'")
    raise ValueError(f"Unsupported generator backend: {backend_type}")


def build_backend_metadata(model_config: dict[str, Any], runtime_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": model_config.get("backend"),
        "granularity": model_config.get("granularity", "scene"),
        "implemented": model_config.get("backend") == "diffusers_text2img",
        "model_id": model_config.get("model_id"),
        "device": runtime_config.get("device"),
        "torch_dtype": runtime_config.get("torch_dtype"),
    }
