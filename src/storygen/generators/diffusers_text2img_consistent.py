from __future__ import annotations

from typing import Any

from storygen.generators.consistent_attention import ConsistentAttentionState, ConsistentSelfAttentionProcessor
from storygen.types import GenerationCandidate, GenerationRequest

from .diffusers_text2img import DiffusersTextToImageGenerator


class DiffusersTextToImageConsistentAttentionGenerator(DiffusersTextToImageGenerator):
    """Opt-in diffusers backend for StoryDiffusion-style consistent attention."""

    def __init__(self, model_config: dict[str, Any], runtime_config: dict[str, Any]) -> None:
        super().__init__(model_config, runtime_config)
        self.consistent_attention_config = dict(model_config.get("consistent_attention", {}))
        self.consistent_attention_enabled = bool(self.consistent_attention_config.get("enabled", False))
        self.consistent_attention_state = ConsistentAttentionState(
            id_length=int(self.consistent_attention_config.get("id_length", 4)),
            write_mode=bool(self.consistent_attention_config.get("write_mode", False)),
        )
        self._consistent_attention_attached = False

    def load(self) -> None:
        super().load()
        if not self.consistent_attention_enabled or self._consistent_attention_attached:
            return
        if self.pipeline is None:
            return
        unet = getattr(self.pipeline, "unet", None)
        if unet is None or not hasattr(unet, "attn_processors") or not hasattr(unet, "set_attn_processor"):
            return
        processors = {}
        for name, processor in unet.attn_processors.items():
            if name.endswith("attn1.processor"):
                processors[name] = ConsistentSelfAttentionProcessor(state=self.consistent_attention_state)
            else:
                processors[name] = processor
        unet.set_attn_processor(processors)
        self._consistent_attention_attached = True

    def _configure_consistent_attention_state(self, request: GenerationRequest) -> None:
        self.consistent_attention_state.write_mode = bool(
            request.extra_options.get("consistent_attention_write_mode", self.consistent_attention_state.write_mode)
        )
        self.consistent_attention_state.id_length = int(
            request.extra_options.get("consistent_attention_id_length", self.consistent_attention_state.id_length)
        )
        character_order = request.extra_options.get("consistent_attention_character_order", [])
        if isinstance(character_order, list):
            self.consistent_attention_state.character_order = [str(item) for item in character_order]
        self.consistent_attention_state.total_count = int(request.extra_options.get("consistent_attention_total_count", 0))
        self.consistent_attention_state.cur_step = 0
        self.consistent_attention_state.attn_count = 0

    def generate_scene(self, request: GenerationRequest) -> GenerationCandidate:
        if self.consistent_attention_enabled:
            self._configure_consistent_attention_state(request)
        candidate = super().generate_scene(request)
        candidate.metadata = {
            **candidate.metadata,
            "consistent_attention_backend": True,
            "consistent_attention_enabled": self.consistent_attention_enabled,
            "consistent_attention_config": self.consistent_attention_config,
            "consistent_attention_state": {
                "id_length": self.consistent_attention_state.id_length,
                "write_mode": self.consistent_attention_state.write_mode,
                "character_order": list(self.consistent_attention_state.character_order),
                "total_count": self.consistent_attention_state.total_count,
            },
        }
        return candidate
