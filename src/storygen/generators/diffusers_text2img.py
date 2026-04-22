from __future__ import annotations

import time
from typing import Any

from storygen.types import GenerationCandidate, GenerationRequest

from .base import BaseImageGenerator


class DiffusersTextToImageGenerator(BaseImageGenerator):
    """Pure text-to-image wrapper around diffusers."""

    def __init__(self, model_config: dict[str, Any], runtime_config: dict[str, Any]) -> None:
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.pipeline = None
        self.device = runtime_config.get("device", "cuda")
        self.dtype = runtime_config.get("torch_dtype", "float16")

    def load(self) -> None:
        if self.pipeline is not None:
            return

        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise ImportError(
                "Diffusers backend dependencies are missing. Install requirements.txt first."
            ) from exc

        dtype = getattr(torch, self.dtype)
        pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_config["model_id"],
            torch_dtype=dtype,
        )
        pipeline = pipeline.to(self.device)
        if self.model_config.get("enable_attention_slicing", False):
            pipeline.enable_attention_slicing()
        self.pipeline = pipeline

    def generate_scene(self, request: GenerationRequest) -> GenerationCandidate:
        self.load()

        import torch

        started_at = time.time()
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        result = self.pipeline(
            prompt=request.prompt_spec.generation_prompt,
            negative_prompt=request.prompt_spec.negative_prompt or None,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator,
        )
        image = result.images[0]
        elapsed = time.time() - started_at

        return GenerationCandidate(
            scene_id=request.scene_id,
            candidate_index=request.candidate_index,
            seed=request.seed,
            prompt_spec=request.prompt_spec,
            image=image,
            metadata={
                "backend": "diffusers_text2img",
                "model_id": self.model_config["model_id"],
                "device": self.device,
                "torch_dtype": self.dtype,
                "width": request.width,
                "height": request.height,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "reference_image_path": request.reference_image_path,
                "previous_selected_image_path": request.previous_selected_image_path,
                "elapsed_seconds": round(elapsed, 4),
                "extra_options": request.extra_options,
            },
        )
