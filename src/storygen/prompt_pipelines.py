from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from storygen.character_specs import build_rule_based_character_specs
from storygen.llm_assisted_prompt_builder import LLMAssistedPromptBuilder
from storygen.llm_client import BaseLLMClient
from storygen.native_prompting import LLMDirectPromptBuilder, build_anchor_prompt_bundle
from storygen.prompt_stack.factory import build_rule_prompt_builder
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
        self.builder = build_rule_prompt_builder(prompt_config)
        self.last_character_specs: dict[str, dict[str, Any]] = {}

    def build(self, story: Story) -> PromptBundle:
        self.last_character_specs = build_rule_based_character_specs(story)
        return PromptBundle(
            scene_prompts=self.builder.build_story_prompts(story),
            story_prompt=None,
            metadata=self.metadata(),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "pipeline": "rule_based",
            "implemented": True,
            "prompt_builder": str(self.prompt_config.get("builder", "legacy")),
            "rewriter_type": self.prompt_config.get("rewriter", {}).get("type", "rule_based"),
            "character_specs": self.last_character_specs,
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


class LLMDirectPromptPipeline(BasePromptPipeline):
    def __init__(
        self,
        prompt_config: dict[str, Any],
        *,
        llm_client: BaseLLMClient | None = None,
        event_logger=None,
    ) -> None:
        self.prompt_config = prompt_config
        self.builder = LLMDirectPromptBuilder(
            prompt_config,
            llm_client=llm_client,
            event_logger=event_logger,
        )
        self.last_bundle: PromptBundle | None = None

    def build(self, story: Story) -> PromptBundle:
        payload = self.builder.build(story)
        if "anchor" not in payload.target_backends:
            raise ValueError("prompt.pipeline=llm_direct requires prompt.llm_direct.targets to include 'anchor'")
        self.last_bundle = build_anchor_prompt_bundle(payload, story, self.prompt_config)
        self.last_bundle.metadata.update(self.metadata())
        return self.last_bundle

    def metadata(self) -> dict[str, Any]:
        return self.builder.metadata()


ApiPromptPipeline = LLMAssistedPromptPipeline


def build_prompt_pipeline(prompt_config: dict[str, Any], *, event_logger=None) -> BasePromptPipeline:
    pipeline_type = prompt_config.get("pipeline", "rule_based")
    if pipeline_type == "rule_based":
        return RuleBasedPromptPipeline(prompt_config)
    if pipeline_type in {"llm_assisted", "api"}:
        return LLMAssistedPromptPipeline(prompt_config, event_logger=event_logger)
    if pipeline_type == "llm_direct":
        return LLMDirectPromptPipeline(prompt_config, event_logger=event_logger)
    raise ValueError(f"Unsupported prompt pipeline: {pipeline_type}")
