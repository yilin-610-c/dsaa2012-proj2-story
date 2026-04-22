from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from storygen.llm_assisted_prompt_builder import LLMAssistedPromptBuilder
from storygen.llm_client import BaseLLMClient
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


class LLMAssistedPromptPipeline(BasePromptPipeline):
    def __init__(
        self,
        prompt_config: dict[str, Any],
        *,
        llm_client: BaseLLMClient | None = None,
        event_logger=None,
    ) -> None:
        self.prompt_config = prompt_config
        self.builder = LLMAssistedPromptBuilder(
            prompt_config,
            llm_client=llm_client,
            event_logger=event_logger,
        )

    def build(self, story: Story) -> PromptBundle:
        prompts = self.builder.build_story_prompts(story)
        return PromptBundle(
            scene_prompts=prompts,
            story_prompt=None,
            metadata=self.metadata(),
        )

    def metadata(self) -> dict[str, Any]:
        return self.builder.metadata()


ApiPromptPipeline = LLMAssistedPromptPipeline


def build_prompt_pipeline(prompt_config: dict[str, Any], *, event_logger=None) -> BasePromptPipeline:
    pipeline_type = prompt_config.get("pipeline", "rule_based")
    if pipeline_type == "rule_based":
        return RuleBasedPromptPipeline(prompt_config)
    if pipeline_type in {"llm_assisted", "api"}:
        return LLMAssistedPromptPipeline(prompt_config, event_logger=event_logger)
    raise ValueError(f"Unsupported prompt pipeline: {pipeline_type}")
