from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from storygen.prompt_builder import PromptBuilder
from storygen.types import PromptBundle, Story


class BasePromptPipeline(ABC):
    @abstractmethod
    def build(self, story: Story) -> PromptBundle:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class RuleBasedPromptPipeline(BasePromptPipeline):
    def __init__(self, prompt_config: dict[str, Any]) -> None:
        self.prompt_config = prompt_config
        self.builder = PromptBuilder(prompt_config)

    def build(self, story: Story) -> PromptBundle:
        return PromptBundle(
            scene_prompts=self.builder.build_story_prompts(story),
            story_prompt=None,
            metadata=self.metadata(),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "pipeline": "rule_based",
            "implemented": True,
            "rewriter_type": self.prompt_config.get("rewriter", {}).get("type", "rule_based"),
        }


class ApiPromptPipeline(BasePromptPipeline):
    def __init__(self, prompt_config: dict[str, Any]) -> None:
        self.prompt_config = prompt_config

    def build(self, story: Story) -> PromptBundle:
        raise NotImplementedError(
            "prompt.pipeline='api' is a placeholder. Implement the API prompt pipeline before using it."
        )

    def metadata(self) -> dict[str, Any]:
        api_config = self.prompt_config.get("api", {})
        return {
            "pipeline": "api",
            "implemented": False,
            "provider": api_config.get("provider"),
            "model": api_config.get("model"),
        }


def build_prompt_pipeline(prompt_config: dict[str, Any]) -> BasePromptPipeline:
    pipeline_type = prompt_config.get("pipeline", "rule_based")
    if pipeline_type == "rule_based":
        return RuleBasedPromptPipeline(prompt_config)
    if pipeline_type == "api":
        return ApiPromptPipeline(prompt_config)
    raise ValueError(f"Unsupported prompt pipeline: {pipeline_type}")
