from __future__ import annotations

from typing import Any

from storygen.prompt_builder import PromptBuilder
from storygen.prompt_stack.chain import SceneSpecsStep, StoryContextStep, default_rule_steps
from storygen.prompt_stack.factory import merge_prompt_config_with_pack
from storygen.prompt_stack.renderers.storydiffusion import sanitize_prompt_specs_for_storydiffusion
from storygen.prompt_stack.state import PromptState
from storygen.types import PromptSpec, Scene, Story


class ModularPromptBuilder:
    """
    Opt-in prompt stack: structured steps + optional StoryDiffusion post-filter.
    SDXL path matches legacy strings by delegating to PromptBuilder on merged config.
    """

    def __init__(
        self,
        prompt_config: dict[str, Any],
        *,
        backend: str = "sdxl",
        template_pack: str | None = None,
    ) -> None:
        self.prompt_config = prompt_config
        self.backend = str(backend or "sdxl").strip().lower()
        modular = prompt_config.get("modular") if isinstance(prompt_config.get("modular"), dict) else {}
        pack = template_pack if template_pack is not None else modular.get("template_pack", "default")
        self._template_pack = str(pack or "default")
        self._merged_prompt_config, self._post_rules = merge_prompt_config_with_pack(
            dict(prompt_config),
            self._template_pack,
        )
        self._inner = PromptBuilder(self._merged_prompt_config)
        self._steps = default_rule_steps(self._inner)

    def build_story_context(self, story: Story) -> dict[str, Any]:
        return self._inner.build_story_context(story)

    def build_prompt_for_scene(
        self,
        story: Story,
        scene: Scene,
        story_context: dict[str, str | list[str] | None] | None = None,
    ) -> PromptSpec:
        return self._inner.build_prompt_for_scene(story, scene, story_context=story_context)

    def build_story_prompts(self, story: Story) -> dict[str, PromptSpec]:
        state = PromptState(story=story)
        for step in self._steps:
            step.run(state)
        specs = dict(state.scene_specs)
        if self.backend == "storydiffusion" and self._post_rules:
            specs = sanitize_prompt_specs_for_storydiffusion(specs, self._post_rules)
        return specs
