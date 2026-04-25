from __future__ import annotations

from typing import Any

from storygen.types import GenerationRequest, PanelGenerationOutput, StoryGenerationRequest, StoryGenerationResult

from .base import BaseStoryGenerator
from .diffusers_text2img import DiffusersTextToImageGenerator


class StoryDiffusionDirectGenerator(BaseStoryGenerator):
    """Story-level backend that currently delegates to scene-level diffusers generation."""

    def __init__(self, model_config: dict[str, Any], runtime_config: dict[str, Any]) -> None:
        self.model_config = model_config
        self.runtime_config = runtime_config
        self._scene_generator: DiffusersTextToImageGenerator | None = None

    def load(self) -> None:
        if self._scene_generator is None:
            model_id = str(
                self.model_config.get("scene_model_id")
                or self.model_config.get("anchor_bank_model_id")
                or self.model_config.get("model_id")
                or ""
            ).strip()
            if not model_id or model_id == "storydiffusion_direct_placeholder":
                model_id = "stabilityai/sdxl-turbo"
            scene_model_config = {
                **self.model_config,
                "backend": "diffusers_text2img",
                "granularity": "scene",
                "model_id": model_id,
                "enable_attention_slicing": False,
            }
            self._scene_generator = DiffusersTextToImageGenerator(scene_model_config, self.runtime_config)
        self._scene_generator.load()

    def generate_story(self, request: StoryGenerationRequest) -> StoryGenerationResult:
        self.load()
        assert self._scene_generator is not None

        panel_outputs: list[PanelGenerationOutput] = []
        for plan in request.scene_plans:
            identity_plan = dict(plan.identity_plan or {})
            reference_image_path = identity_plan.get("identity_anchor_path") if identity_plan.get("identity_conditioning_enabled") else None
            if not reference_image_path:
                reference_image_path = plan.anchor_paths.get(plan.anchor_characters[0]) if plan.anchor_characters else None
            generation_request = GenerationRequest(
                scene_id=plan.scene_id,
                candidate_index=0,
                seed=int(request.seed + plan.scene_index),
                prompt_spec=plan.prompt_spec,
                width=request.width,
                height=request.height,
                guidance_scale=0.0,
                num_inference_steps=int(self.model_config.get("num_inference_steps", 4)),
                reference_image_path=reference_image_path,
                previous_selected_image_path=None,
                extra_options={
                    "generation_mode": "text2img",
                    "story_backend": "storydiffusion_direct",
                    "identity_conditioning_enabled": bool(reference_image_path),
                    "identity_anchor_character_id": identity_plan.get("identity_anchor_character_id"),
                    "identity_anchor_type": identity_plan.get("identity_anchor_type"),
                    "identity_anchor_path": reference_image_path,
                    "identity_conditioning_reason": identity_plan.get("identity_conditioning_reason"),
                    "ip_adapter_scale": identity_plan.get("ip_adapter_scale", 0.6),
                    "ip_adapter_model_id": identity_plan.get("ip_adapter_model_id"),
                    "ip_adapter_subfolder": identity_plan.get("ip_adapter_subfolder"),
                    "ip_adapter_weight_name": identity_plan.get("ip_adapter_weight_name"),
                    "identity_apply_to_modes": identity_plan.get("identity_apply_to_modes", ["text2img"]),
                    "anchor_paths": dict(plan.anchor_paths or {}),
                    "anchor_characters": list(plan.anchor_characters or []),
                    "route_hint": dict(plan.route_hint or {}),
                },
            )
            candidate = self._scene_generator.generate_scene(generation_request)
            panel_outputs.append(
                PanelGenerationOutput(
                    scene_id=plan.scene_id,
                    panel_index=plan.scene_index,
                    prompt=plan.generation_prompt,
                    image=candidate.image,
                    image_path=candidate.image_path,
                    metadata={
                        "identity_plan": identity_plan,
                        "anchor_paths": dict(plan.anchor_paths or {}),
                        "anchor_characters": list(plan.anchor_characters or []),
                        "route_hint": dict(plan.route_hint or {}),
                        "scene_backend_metadata": candidate.metadata,
                    },
                )
            )

        return StoryGenerationResult(
            backend="storydiffusion_direct",
            seed=request.seed,
            panel_outputs=panel_outputs,
            metadata={
                "implemented": True,
                "message": "storydiffusion_direct delegated to scene-level diffusers generation",
                "scene_plan_count": len(request.scene_plans),
                "anchor_bank_enabled": bool(request.anchor_bank_summary.get("enabled", False)),
                "character_specs": request.character_specs,
                "scene_plans": [plan.metadata | {"scene_id": plan.scene_id, "scene_index": plan.scene_index} for plan in request.scene_plans],
            },
        )
