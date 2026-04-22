from __future__ import annotations

from typing import Any

from storygen.types import StoryGenerationRequest, StoryGenerationResult

from .base import BaseStoryGenerator


class StoryDiffusionDirectGenerator(BaseStoryGenerator):
    """Placeholder for a future story-level StoryDiffusion backend."""

    def __init__(self, model_config: dict[str, Any], runtime_config: dict[str, Any]) -> None:
        self.model_config = model_config
        self.runtime_config = runtime_config

    def load(self) -> None:
        return None

    def generate_story(self, request: StoryGenerationRequest) -> StoryGenerationResult:
        raise NotImplementedError(
            "model.backend='storydiffusion_direct' is a placeholder. "
            "Implement the story-level StoryDiffusion backend before using it."
        )
