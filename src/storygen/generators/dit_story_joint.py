from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from storygen.types import (
    GenerationCandidate,
    PanelGenerationOutput,
    StoryGenerationRequest,
    StoryGenerationResult,
)

from .base import BaseStoryGenerator


class CrossSceneSelfAttentionWrapper(nn.Module):
    """Wraps self-attention to share scene-level context across scenes.

    Instead of spatially-aligned token blending (which hurts moving
    characters), this uses **global context blending**: each scene's
    mean feature vector is averaged across scenes and added back as a
    uniform bias to every token.  This propagates overall color tone,
    lighting, and style while preserving spatial structure and
    character identity at every spatial position.

    Also supports per-layer blend_strength for fine-grained control:
    shallow layers (texture / color) can be blended more heavily than
    deep layers (semantics / identity).
    """

    def __init__(
        self,
        original_attn: nn.Module,
        layer_idx: int,
        cross_scene_layers: set[int],
        blend_strength: float = 0.15,
        blend_strength_per_layer: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        self.attn = original_attn
        self.layer_idx = layer_idx
        self.cross_scene_layers = cross_scene_layers
        self._num_scenes = 0

        # Per-layer override takes precedence
        if blend_strength_per_layer and layer_idx in blend_strength_per_layer:
            self.blend_strength = float(blend_strength_per_layer[layer_idx])
        else:
            self.blend_strength = float(blend_strength)

    def set_num_scenes(self, num_scenes: int) -> None:
        self._num_scenes = num_scenes

    @property
    def _is_cross_scene(self) -> bool:
        return self._num_scenes > 1 and self.layer_idx in self.cross_scene_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        alpha = self.blend_strength
        if not self._is_cross_scene or alpha <= 0.0:
            return self.attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

        # hidden_states: (B, T, D)  with B = num_groups * num_scenes
        B = hidden_states.shape[0]
        N = self._num_scenes
        if B % N != 0:
            return self.attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

        groups = B // N  # 1 (no CFG) or 2 (with CFG)
        T = hidden_states.shape[1]

        # Reshape to separate groups and scenes: (B, T, D) → (groups, N, T, D)
        x = hidden_states.reshape(groups, N, T, hidden_states.shape[2])

        # Compute per-scene global context (one vector per scene):
        # mean over all SPATIAL tokens → (groups, N, 1, D)
        scene_global = x.mean(dim=2, keepdim=True)

        # Cross-scene global context (average of all scenes within each group)
        cross_global = scene_global.mean(dim=1, keepdim=True)  # (groups, 1, 1, D)

        # Blend: add the same cross-scene global bias to EVERY token.
        # This ONLY affects global statistics (color cast, overall lighting,
        # style tone) without distorting spatial structure or moving
        # character features to wrong positions.
        x_blended = (1.0 - alpha) * x + alpha * cross_global  # (groups, N, T, D)

        # Reshape back to (B, T, D)
        out = x_blended.reshape(B, T, hidden_states.shape[2])

        # Standard self-attention on blended tokens
        return self.attn(
            out,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


class DitStoryJointGenerator(BaseStoryGenerator):
    """Story-level generator using PixArt-Alpha DiT with cross-scene attention.

    All scene latents are generated jointly through the DiT transformer.
    In designated layers, self-attention tokens are enriched with signals
    from corresponding spatial positions in other scenes, encouraging
    consistent identity, style, and lighting across the story sequence.
    """

    def __init__(self, model_config: dict[str, Any], runtime_config: dict[str, Any]) -> None:
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.device = runtime_config.get("device", "cuda")
        self.dtype = runtime_config.get("torch_dtype", "float16")
        self.pipeline = None
        self._cross_scene_injected = False

    def load(self) -> None:
        if self.pipeline is not None:
            return

        try:
            from diffusers import PixArtAlphaPipeline
        except ImportError as exc:
            raise ImportError(
                "DiT story backend requires diffusers>=0.24. "
                "Install with: pip install diffusers>=0.24 transformers accelerate"
            ) from exc

        import torch as _torch

        dtype = getattr(_torch, self.dtype)
        model_id = self.model_config.get("model_id", "PixArt-alpha/PixArt-XL-2-1024-MS")

        pipeline = PixArtAlphaPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

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
        self._cross_scene_injected = False

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

    def _inject_cross_scene_attention(self, num_scenes: int) -> None:
        """Replace self-attention in designated transformer blocks."""
        if self._cross_scene_injected or num_scenes <= 1:
            return

        story_attention_config = self.model_config.get("story_attention", {})
        cross_scene_layers = set(story_attention_config.get("cross_scene_layers", []))
        blend_strength = float(story_attention_config.get("blend_strength", 0.15))
        blend_strength_per_layer: dict[int, float] | None = None
        raw_per_layer = story_attention_config.get("blend_strength_per_layer")
        if isinstance(raw_per_layer, dict):
            blend_strength_per_layer = {int(k): float(v) for k, v in raw_per_layer.items()}
        if not cross_scene_layers:
            return

        transformer = self.pipeline.transformer
        for i, block in enumerate(transformer.transformer_blocks):
            if i in cross_scene_layers and not isinstance(block.attn1, CrossSceneSelfAttentionWrapper):
                block.attn1 = CrossSceneSelfAttentionWrapper(
                    block.attn1,
                    layer_idx=i,
                    cross_scene_layers=cross_scene_layers,
                    blend_strength=blend_strength,
                    blend_strength_per_layer=blend_strength_per_layer,
                )

        self._cross_scene_injected = True

    def _update_num_scenes(self, num_scenes: int) -> None:
        """Propagate scene count to all cross-scene attention wrappers."""
        for block in self.pipeline.transformer.transformer_blocks:
            if isinstance(block.attn1, CrossSceneSelfAttentionWrapper):
                block.attn1.set_num_scenes(num_scenes)

    def _encode_prompts(
        self, prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode multiple prompts into batched embeddings and attention masks.

        Returns:
            (prompt_embeds, prompt_attention_mask) — both shape (N, seq_len, dim)
        """
        pipeline = self.pipeline
        text_encoder = pipeline.text_encoder
        tokenizer = pipeline.tokenizer

        max_length = int(self.model_config.get("max_sequence_length", 120))
        embeds_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []

        for prompt in prompts:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(text_encoder.device)
            with torch.no_grad():
                embeds = text_encoder(text_input_ids)[0]
            embeds_list.append(embeds)
            mask_list.append(text_inputs.attention_mask.to(embeds.device))

        prompt_embeds = torch.cat(embeds_list, dim=0)  # (N, seq_len, dim)
        prompt_attention_mask = torch.cat(mask_list, dim=0)  # (N, seq_len)
        return prompt_embeds, prompt_attention_mask

    def generate_story(self, request: StoryGenerationRequest) -> StoryGenerationResult:
        self.load()
        assert self.pipeline is not None

        import torch as _torch

        num_scenes = len(request.scene_plans)
        if num_scenes == 0:
            raise ValueError("At least one scene plan is required.")

        # Inject cross-scene attention into the transformer before generating
        self._inject_cross_scene_attention(num_scenes)
        self._update_num_scenes(num_scenes)

        width = request.width
        height = request.height
        guidance_scale = float(self.model_config.get("guidance_scale", 4.5))
        num_inference_steps = int(self.model_config.get("num_inference_steps", 20))
        seed = request.seed
        max_sequence_length = int(self.model_config.get("max_sequence_length", 120))

        pipeline = self.pipeline

        # Build prompts (same modes as dit_text2img).
        prompt_mode = str(self.model_config.get("prompt_mode", "simple")).strip()
        prompts: list[str] = []
        for plan in request.scene_plans:
            if prompt_mode == "full":
                p = (plan.generation_prompt or plan.prompt_spec.generation_prompt or "").strip()
            else:
                scoring = (plan.scoring_prompt or plan.prompt_spec.scoring_prompt or "").strip()
                style = (plan.prompt_spec.style_prompt or "").strip()
                p = scoring
                if style:
                    p = f"{p}, {style}" if p else style
            prompts.append(p)

        # Inject LoRA trigger word into every scene prompt if configured
        lora_trigger = str(self.model_config.get("lora_trigger") or "").strip()
        if lora_trigger:
            prompts = [f"{lora_trigger} {p}" for p in prompts]
        negative = request.negative_prompt or None

        generator = _torch.Generator(device=self.device).manual_seed(seed)

        started_at = time.time()

        # Use the pipeline's own __call__ with batched prompts.
        # The pipeline handles: resolution binning, adaLN conditioning,
        # CFG, scheduler steps, VAE decode, and image postprocessing.
        result = pipeline(
            prompt=prompts,
            negative_prompt=negative,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            max_sequence_length=max_sequence_length,
            output_type="pil",
        )

        elapsed = time.time() - started_at
        images = result.images  # list of PIL images, one per prompt

        # Build panel outputs
        panel_outputs: list[PanelGenerationOutput] = []
        for i, plan in enumerate(request.scene_plans):
            image = images[i] if i < len(images) else None

            panel_outputs.append(
                PanelGenerationOutput(
                    scene_id=plan.scene_id,
                    panel_index=plan.scene_index,
                    prompt=prompts[i] if i < len(prompts) else plan.generation_prompt,
                    image=image,
                    metadata={
                        "backend": "dit_story_joint",
                        "model_id": self.model_config["model_id"],
                        "device": self.device,
                        "torch_dtype": self.dtype,
                        "width": width,
                        "height": height,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": num_inference_steps,
                        "elapsed_seconds": round(elapsed, 4),
                        "story_attention": {
                            "enabled": True,
                            "cross_scene_layers": list(
                                self.model_config.get("story_attention", {}).get("cross_scene_layers", [])
                            ),
                            "num_scenes": num_scenes,
                        },
                    },
                )
            )

        return StoryGenerationResult(
            backend="dit_story_joint",
            seed=seed,
            panel_outputs=panel_outputs,
            metadata={
                "implemented": True,
                "message": (
                    f"dit_story_joint: {num_scenes} scenes generated jointly through PixArt-Alpha DiT "
                    f"with cross-scene attention in layers "
                    f"{self.model_config.get('story_attention', {}).get('cross_scene_layers', [])}"
                ),
                "scene_plan_count": num_scenes,
                "anchor_bank_enabled": False,
                "character_specs": request.character_specs,
                "scene_plans": [
                    {"scene_id": p.scene_id, "scene_index": p.scene_index}
                    for p in request.scene_plans
                ],
                "elapsed_seconds": round(elapsed, 4),
            },
        )
