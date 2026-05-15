from __future__ import annotations

import time
from typing import Any

from storygen.types import GenerationCandidate, GenerationRequest

from .base import BaseImageGenerator


class DitTextToImageGenerator(BaseImageGenerator):
    """Scene-level DiT-based text-to-image generator using PixArt-Alpha."""

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
            from diffusers import PixArtAlphaPipeline
        except ImportError as exc:
            raise ImportError(
                "DiT backend requires diffusers>=0.24 with PixArtAlphaPipeline. "
                "Install with: pip install diffusers>=0.24 transformers accelerate"
            ) from exc

        dtype = getattr(torch, self.dtype)
        model_id = self.model_config.get("model_id", "PixArt-alpha/PixArt-XL-2-1024-MS")

        pipeline = PixArtAlphaPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        if self.model_config.get("enable_attention_slicing", False) and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        if self.model_config.get("enable_vae_slicing", False) and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()

        if self.model_config.get("enable_vae_tiling", False) and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

        lora_path = self.model_config.get("lora_path")
        if lora_path:
            self._load_lora(pipeline, lora_path)

        if self.model_config.get("enable_model_cpu_offload", True):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(self.device)

        self.pipeline = pipeline

    def _load_lora(self, pipeline, lora_path: str) -> None:
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("LoRA loading requires peft. Install with: pip install peft")
        import os
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora_path)
        pipeline.transformer = pipeline.transformer.merge_and_unload()
        te_lora_path = os.path.join(lora_path, "text_encoder")
        if os.path.isdir(te_lora_path):
            pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, te_lora_path)
            pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()

    def generate_scene(self, request: GenerationRequest) -> GenerationCandidate:
        self.load()

        import torch

        started_at = time.time()
        generator = torch.Generator(device=self.device).manual_seed(request.seed)

        # Inject LoRA trigger word into prompt if configured
        prompt = request.prompt_spec.generation_prompt
        lora_trigger = str(self.model_config.get("lora_trigger") or "").strip()
        if lora_trigger:
            prompt = f"{lora_trigger} {prompt}"

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": request.prompt_spec.negative_prompt or None,
            "width": request.width,
            "height": request.height,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "generator": generator,
            "max_sequence_length": int(self.model_config.get("max_sequence_length", 120)),
        }

        result = self.pipeline(**call_kwargs)
        image = result.images[0]
        elapsed = time.time() - started_at

        lora_dir = self.model_config.get("lora_path")
        meta: dict[str, Any] = {
            "backend": "dit_text2img",
            "model_id": self.model_config["model_id"],
            "device": self.device,
            "torch_dtype": self.dtype,
            "width": request.width,
            "height": request.height,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "elapsed_seconds": round(elapsed, 4),
            "extra_options": request.extra_options,
            "generation_mode": "text2img",
            "identity_conditioning_enabled": False,
            "identity_conditioning_applied": False,
            # Verifiable LoRA application (Peft merge is done in load(); weights are fused into transformer)
            "lora_checkpoint_dir": str(lora_dir) if lora_dir else None,
            "lora_peft_merged_into_transformer": bool(lora_dir),
            "lora_trigger_prefix": lora_trigger if lora_trigger else None,
            "pixart_prompt": prompt,
        }

        return GenerationCandidate(
            scene_id=request.scene_id,
            candidate_index=request.candidate_index,
            seed=request.seed,
            prompt_spec=request.prompt_spec,
            image=image,
            metadata=meta,
        )
